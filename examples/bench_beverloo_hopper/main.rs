//! Beverloo hopper discharge benchmark: validates mass flow rate against the
//! Beverloo correlation for 2D hopper discharge.
//!
//! Uses a flat-bottom rectangular hopper (periodic in y for quasi-2D) with a
//! central orifice. Particles fill under gravity, settle, then discharge.
//! A custom system tracks remaining particle count over time to compute the
//! steady-state mass flow rate.
//!
//! The orifice width `D` is parameterized in config.toml via wall bounds.
//! Run multiple times with different `D` values using `run_sweep.sh` and
//! use `validate.py` / `plot.py` to compare against Beverloo's equation.
//!
//! ```bash
//! cargo run --release --example bench_beverloo_hopper --no-default-features \
//!     -- examples/bench_beverloo_hopper/config.toml
//! ```

use dem_measure_plane::MeasurePlanePlugin;
use mddem::prelude::*;
use std::fs::{self, OpenOptions};
use std::io::Write;

#[derive(Clone, Debug, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("filling")]
    Filling,
    #[stage("discharge")]
    Discharge,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(MeasurePlanePlugin)
        .add_plugins(StatesPlugin {
            initial: Phase::Filling,
        })
        .add_plugins(StageAdvancePlugin::<Phase>::new());

    app.add_update_system(
        check_settled.run_if(in_state(Phase::Filling)),
        ScheduleSet::PostFinalIntegration,
    );

    app.add_update_system(
        track_particle_count.run_if(in_state(Phase::Discharge)),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

/// Check if particles have settled using per-particle KE threshold.
/// With many particles under gravity, total KE can remain high even when
/// particles are nearly at rest due to residual vibrations. Per-particle
/// KE avoids this scaling issue.
fn check_settled(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut walls: ResMut<Walls>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    // Wait at least 5000 steps for particles to fall and begin settling,
    // then check every 200 steps
    if step < 5000 || step % 200 != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let local_ke: f64 = (0..nlocal)
        .map(|i| {
            let vx = atoms.vel[i][0];
            let vy = atoms.vel[i][1];
            let vz = atoms.vel[i][2];
            0.5 * atoms.mass[i] * (vx * vx + vy * vy + vz * vz)
        })
        .sum();
    let global_ke = comm.all_reduce_sum_f64(local_ke);
    let global_n = comm.all_reduce_sum_f64(nlocal as f64);

    // Per-particle KE threshold: ~1e-7 J corresponds to v ≈ 0.14 m/s
    // for a 2mm glass sphere (mass ~1.05e-5 kg). This is slow enough
    // to indicate settling while being achievable in reasonable time.
    let ke_per_particle = if global_n > 0.0 {
        global_ke / global_n
    } else {
        0.0
    };

    if ke_per_particle < 1e-7 {
        walls.deactivate_by_name("blocker");
        next_state.set(Phase::Discharge);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE/particle = {:.3e} J (total KE = {:.3e} J, N = {:.0}) — settled, removing blocker",
                step, ke_per_particle, global_ke, global_n
            );
        }
    }
}

/// Track particle count and total mass remaining in the hopper over time.
/// Writes data to `data/particle_count.txt` for post-processing.
fn track_particle_count(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    // Record every 200 steps during discharge for good time resolution
    if step % 200 != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;

    // Count particles still above the floor (z > threshold, i.e., still in hopper)
    let z_threshold = -0.005; // slightly below the orifice floor
    let local_count: f64 = (0..nlocal)
        .filter(|&i| atoms.pos[i][2] > z_threshold)
        .count() as f64;
    let local_mass: f64 = (0..nlocal)
        .filter(|&i| atoms.pos[i][2] > z_threshold)
        .map(|i| atoms.mass[i])
        .sum();

    let global_count = comm.all_reduce_sum_f64(local_count);
    let global_mass = comm.all_reduce_sum_f64(local_mass);

    if comm.rank() == 0 {
        let base_dir = input
            .output_dir
            .as_deref()
            .unwrap_or("examples/bench_beverloo_hopper");
        let data_dir = format!("{}/data", base_dir);
        let _ = fs::create_dir_all(&data_dir);
        let data_path = format!("{}/particle_count.txt", data_dir);

        let dt = atoms.dt;
        let time = step as f64 * dt;

        // Truncate on first write (discharge just started), append thereafter
        let file_exists = std::path::Path::new(&data_path).exists();
        let mut file = if !file_exists {
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&data_path)
                .expect("failed to create particle_count.txt")
        } else {
            OpenOptions::new()
                .append(true)
                .open(&data_path)
                .expect("failed to open particle_count.txt for append")
        };

        writeln!(
            file,
            "{} {:.10e} {:.0} {:.10e}",
            step, time, global_count, global_mass
        )
        .expect("failed to write particle_count.txt");
    }
}
