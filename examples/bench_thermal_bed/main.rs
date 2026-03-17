//! Packed bed thermal conductivity benchmark.
//!
//! Validates DEM contact-based heat conduction against expected behavior
//! for a packed bed of monodisperse spheres conducting heat through contacts.
//!
//! **Stage 1 ("settling")**: Velocity Verlet with gravity settles particles
//! into a packed bed on the bottom wall. Top wall is placed high to avoid
//! interfering with settling.
//!
//! **Stage 2 ("thermal")**: The top wall is moved down to contact the bed
//! surface. Wall temperatures are set (bottom = 400 K, top = 300 K). Heat
//! conducts through contacts until steady state.
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --example bench_thermal_bed -- examples/bench_thermal_bed/config.toml
//! cd examples/bench_thermal_bed && python3 validate.py && python3 plot.py
//! ```

use mddem::prelude::*;
use mddem::dem_atom::DemAtom;
use mddem::dem_granular::{
    GranularTempPlugin, HertzMindlinContactPlugin, RotationalDynamicsPlugin,
};
use mddem::dem_thermal::{ThermalAtom, ThermalPlugin};

use std::any::TypeId;
use std::fs::{self, OpenOptions};
use std::io::Write;

fn main() {
    let mut app = App::new();

    app.add_plugins(CorePlugins);

    // DEM physics
    app.add_plugins(DemAtomPlugin)
        .add_plugins(DemAtomInsertPlugin)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(HertzMindlinContactPlugin)
        .add_plugins(RotationalDynamicsPlugin)
        .add_plugins(GranularTempPlugin);

    // Walls and thermal
    app.add_plugins(WallPlugin);
    app.add_plugins(ThermalPlugin);
    app.add_plugins(GravityPlugin);

    // Register temperature as dump scalar
    {
        let dump_reg = app
            .get_mut_resource(TypeId::of::<DumpRegistry>())
            .expect("DumpRegistry not found");
        dump_reg
            .borrow_mut()
            .downcast_mut::<DumpRegistry>()
            .expect("DumpRegistry downcast failed")
            .register_scalar("temperature", |atoms, registry| {
                let thermal = registry.expect::<ThermalAtom>("dump_temperature");
                (0..atoms.nlocal as usize)
                    .map(|i| thermal.temperature[i])
                    .collect()
            });
    }

    // One-shot: configure thermal stage (move top wall, set temperatures, freeze)
    app.add_update_system(
        setup_thermal_stage.run_if(in_stage("thermal")),
        ScheduleSet::PreInitialIntegration,
    );

    // Write thermal output data
    app.add_update_system(
        write_thermal_data.run_if(in_stage("thermal")),
        ScheduleSet::PreExchange,
    );

    app.start();
}

/// One-shot system at the start of the thermal stage:
/// 1. Find the highest particle and move top wall just above it
/// 2. Set wall temperatures (bottom=400K, top=300K)
/// 3. Zero all velocities
/// 4. Reset all temperatures to initial value (undo any drift from settling)
fn setup_thermal_stage(
    mut atoms: ResMut<Atom>,
    mut walls: ResMut<Walls>,
    registry: Res<AtomDataRegistry>,
    comm: Res<CommResource>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    *done = true;

    let nlocal = atoms.nlocal as usize;
    let dem = registry.expect::<DemAtom>("setup_thermal");
    let mut thermal = registry.expect_mut::<ThermalAtom>("setup_thermal");

    // Find the highest particle center (use negative min trick for max)
    let mut local_z_max = f64::NEG_INFINITY;
    let mut radius = 0.001_f64; // default
    for i in 0..nlocal {
        if atoms.pos[i][2] > local_z_max {
            local_z_max = atoms.pos[i][2];
            radius = dem.radius[i];
        }
    }
    // For max: negate, use min, negate back
    let global_z_max = -comm.all_reduce_min_f64(-local_z_max);
    let r = radius;

    // Place top wall so it overlaps slightly with the top particle
    // Wall at z_max + r * 0.9 (slight overlap for good thermal contact)
    let top_wall_z = global_z_max + r * 0.9;

    for wall in walls.planes.iter_mut() {
        if wall.normal_z > 0.5 {
            // Bottom wall — hot
            wall.temperature = Some(400.0);
        } else if wall.normal_z < -0.5 {
            // Top wall — move down and set cold
            wall.point_z = top_wall_z;
            wall.origin[2] = top_wall_z;
            wall.temperature = Some(300.0);
        }
    }

    // Zero velocities and reset temperatures
    for i in 0..nlocal {
        atoms.vel[i] = [0.0, 0.0, 0.0];
        atoms.force[i] = [0.0, 0.0, 0.0];
        thermal.temperature[i] = 350.0; // Reset to initial
    }

    if comm.rank() == 0 {
        println!(
            "Thermal stage setup: top wall at z={:.4}m, bottom=400K, top=300K",
            top_wall_z
        );
    }
}

/// Write thermal output data at each thermo interval.
fn write_thermal_data(
    atoms: Res<Atom>,
    walls: Res<Walls>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    input: Res<Input>,
    config: Res<ThermalConfig>,
    comm: Res<CommResource>,
) {
    let index = scheduler_manager.index;
    let thermo_interval = run_config.current_stage(index).thermo;
    if !run_state.total_cycle.is_multiple_of(thermo_interval) {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let thermal = registry.expect::<ThermalAtom>("write_thermal_data");
    let dem = registry.expect::<DemAtom>("write_thermal_data");
    let k = config.conductivity;

    // Compute heat flux from bottom wall (normal_z > 0)
    let mut local_wall_flux = 0.0_f64;
    for wall in walls.planes.iter() {
        if wall.normal_z < 0.5 {
            continue;
        }
        let wall_temp = match wall.temperature {
            Some(t) => t,
            None => continue,
        };
        for i in 0..nlocal {
            let px = atoms.pos[i][0];
            let py = atoms.pos[i][1];
            let pz = atoms.pos[i][2];
            if !wall.in_bounds(px, py, pz) {
                continue;
            }
            let dz = pz - wall.point_z;
            let distance = dz * wall.normal_z;
            if distance <= 0.0 {
                continue;
            }
            let radius = dem.radius[i];
            let delta = radius - distance;
            if delta <= 0.0 {
                continue;
            }
            let r_eff = radius;
            let a = (r_eff * delta).sqrt();
            let q = k * 2.0 * a * (wall_temp - thermal.temperature[i]);
            local_wall_flux += q;
        }
    }

    let global_wall_flux = comm.all_reduce_sum_f64(local_wall_flux);

    // Compute average temperature
    let mut local_temp_sum = 0.0_f64;
    for i in 0..nlocal {
        local_temp_sum += thermal.temperature[i];
    }
    let global_temp_sum = comm.all_reduce_sum_f64(local_temp_sum);
    let global_count = comm.all_reduce_sum_f64(nlocal as f64);

    // Also compute and save the top wall z position for validate.py
    let mut top_wall_z = 0.0_f64;
    let mut bottom_wall_z = 0.0_f64;
    for wall in walls.planes.iter() {
        if wall.normal_z > 0.5 {
            bottom_wall_z = wall.point_z;
        } else if wall.normal_z < -0.5 {
            top_wall_z = wall.point_z;
        }
    }

    if comm.rank() != 0 {
        return;
    }

    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/data", dir),
        None => "data".to_string(),
    };
    fs::create_dir_all(&base_dir).ok();

    let physical_time = run_state.total_cycle as f64 * atoms.dt;
    let avg_temp = if global_count > 0.0 {
        global_temp_sum / global_count
    } else {
        0.0
    };

    // Write wall heat flux time series
    let flux_path = format!("{}/WallHeatFlux.txt", base_dir);
    let is_first = run_state.cycle_count[index] == 0;
    let mut file = if is_first {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&flux_path)
            .expect("failed to create WallHeatFlux.txt");
        writeln!(
            f,
            "# step time wall_heat_flux avg_temperature bottom_wall_z top_wall_z"
        )
        .unwrap();
        f
    } else {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&flux_path)
            .expect("failed to open WallHeatFlux.txt")
    };
    writeln!(
        file,
        "{} {:.6e} {:.10e} {:.6e} {:.6e} {:.6e}",
        run_state.total_cycle,
        physical_time,
        global_wall_flux,
        avg_temp,
        bottom_wall_z,
        top_wall_z
    )
    .unwrap();

    // Write temperature profile — overwrite each time (final = steady state)
    let profile_path = format!("{}/ThermalProfile.csv", base_dir);
    let mut pfile = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&profile_path)
        .expect("failed to create ThermalProfile.csv");
    writeln!(pfile, "z,temperature").unwrap();
    for i in 0..nlocal {
        writeln!(
            pfile,
            "{:.8e},{:.8e}",
            atoms.pos[i][2], thermal.temperature[i]
        )
        .unwrap();
    }
}
