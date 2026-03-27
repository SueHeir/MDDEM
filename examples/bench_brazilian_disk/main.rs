//! Brazilian disk (indirect tensile) test benchmark.
//!
//! A circular disk of bonded particles is compressed between two flat platens.
//! The disk fails by a vertical tensile crack and the measured tensile
//! strength is validated against the analytical Brazilian test formula:
//! `sigma_t = 2P / (pi D t)`.
//!
//! Bond stiffness is chosen to be compatible with the auto-computed Rayleigh
//! timestep, avoiding the need for manual dt override. Platens start just
//! outside the disk surface to avoid initial overlaps.
//!
//! ```bash
//! cargo run --release --example bench_brazilian_disk --no-default-features \
//!     -- examples/bench_brazilian_disk/config.toml
//! ```

use std::f64::consts::PI;
use std::fs::{self, File, OpenOptions};
use std::io::Write as IoWrite;

use mddem::dem_atom::DemAtom;
use mddem::dem_bond::BondMetrics;
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(DemBondPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(FixesPlugin);

    app.add_setup_system(
        setup_disk.run_if(first_stage_only()),
        ScheduleSetupSet::Setup,
    );
    app.add_update_system(record_load_displacement, ScheduleSet::PostForce);

    app.start();
}

// ---------------------------------------------------------------------------
// Setup: hexagonal close-packed disk
// ---------------------------------------------------------------------------

/// Create a circular disk of particles on a hexagonal lattice, centered at
/// the origin in the xz-plane.  The y-direction is thin (quasi-2D).
fn setup_disk(
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let radius: f64 = 0.0005;  // particle radius [m]
    let density: f64 = 2500.0; // density [kg/m^3]
    let disk_radius: f64 = 0.010; // disk outer radius [m]

    let mat_idx = material_table
        .find_material("rock")
        .expect("material 'rock' not found");

    let mut dem = registry.expect_mut::<DemAtom>("setup_disk");

    let spacing = 2.0 * radius;
    let row_height = spacing * (3.0_f64).sqrt() / 2.0;

    let n_rows = (2.0 * disk_radius / row_height).ceil() as i32 + 1;
    let n_cols = (2.0 * disk_radius / spacing).ceil() as i32 + 1;

    let mut count = 0u32;

    for row in -n_rows..=n_rows {
        let z = row as f64 * row_height;
        let x_offset = if row.unsigned_abs() % 2 == 1 {
            radius
        } else {
            0.0
        };

        for col in -n_cols..=n_cols {
            let x = col as f64 * spacing + x_offset;

            let dist_from_center = (x * x + z * z).sqrt();
            if dist_from_center + radius > disk_radius {
                continue;
            }

            let mass = density * (4.0 / 3.0) * PI * radius.powi(3);
            let tag = atom.get_max_tag() + 1;

            atom.natoms += 1;
            atom.nlocal += 1;
            atom.tag.push(tag);
            atom.origin_index.push(0);
            atom.cutoff_radius.push(radius);
            atom.is_ghost.push(false);
            atom.pos.push([x, 0.0, z]);
            atom.vel.push([0.0; 3]);
            atom.force.push([0.0; 3]);
            atom.mass.push(mass);
            atom.inv_mass.push(1.0 / mass);
            atom.atom_type.push(mat_idx);

            dem.radius.push(radius);
            dem.density.push(density);
            dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
            dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
            dem.omega.push([0.0; 3]);
            dem.ang_mom.push([0.0; 3]);
            dem.torque.push([0.0; 3]);
            dem.body_id.push(0.0);

            count += 1;
        }
    }

    println!(
        "BrazilianDisk: placed {} particles in disk of radius {:.4} m",
        count, disk_radius
    );
}

// ---------------------------------------------------------------------------
// Data recording
// ---------------------------------------------------------------------------

/// Record wall forces and platen displacement to a CSV file each thermo step.
fn record_load_displacement(
    walls: Res<Walls>,
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    run_config: Res<RunConfig>,
    input: Res<Input>,
    comm: Res<CommResource>,
    bond_metrics: Res<BondMetrics>,
) {
    if comm.rank() != 0 {
        return;
    }
    let stage = run_config.current_stage(0);
    let thermo_interval = stage.thermo;
    if thermo_interval == 0 || run_state.total_cycle % thermo_interval != 0 {
        return;
    }

    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/data", dir),
        None => "data".to_string(),
    };
    fs::create_dir_all(&base_dir).expect("failed to create data directory");
    let path = format!("{}/load_displacement.csv", base_dir);

    let step = run_state.total_cycle;
    let time = step as f64 * atoms.dt;

    let f_bottom = walls.planes[0].force_accumulator;
    let f_top = walls.planes[1].force_accumulator;
    let load = (f_bottom.abs() + f_top.abs()) / 2.0;

    let z_bottom = walls.planes[0].point_z;
    let z_top = walls.planes[1].point_z;
    let gap = z_top - z_bottom;

    let bonds_broken = bond_metrics.total_bonds_broken;
    let bond_count = bond_metrics.bond_count;

    let mut file = if step == 0 {
        let mut f = File::create(&path).expect("failed to create load_displacement.csv");
        writeln!(
            f,
            "step,time,load,gap,z_bottom,z_top,f_bottom,f_top,bonds_broken,bond_count"
        )
        .expect("write header");
        f
    } else {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("failed to open load_displacement.csv")
    };

    writeln!(
        &mut file,
        "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{},{}",
        step, time, load, gap, z_bottom, z_top, f_bottom, f_top, bonds_broken, bond_count
    )
    .expect("write data line");
}
