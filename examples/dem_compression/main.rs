//! Uniaxial compression: insert → relax → compress with 100× gravity.
//!
//! Three-stage DEM simulation demonstrating per-stage config overrides
//! and KE-based automatic stage advancement.
//!
//! ```bash
//! cargo run --example dem_compression --no-default-features -- examples/dem_compression/config.toml
//! ```

use mddem::prelude::*;

#[derive(Clone, Debug, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("insert")]
    Insert,
    #[stage("relax")]
    Relax,
    #[stage("compress")]
    Compress,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(StatesPlugin {
            initial: Phase::Insert,
        })
        .add_plugins(StageAdvancePlugin::<Phase>::new());

    app.add_update_system(
        check_insert_settled.run_if(in_state(Phase::Insert)),
        ScheduleSet::PostFinalIntegration,
    );
    app.add_update_system(
        check_relaxed.run_if(in_state(Phase::Relax)),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

/// During insert stage, wait for KE to drop below threshold then advance to relax.
fn check_insert_settled(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    if step < 1000 || step % 100 != 0 {
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

    if global_ke < 1e-5 {
        next_state.set(Phase::Relax);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE = {:.3e} J — particles settled, advancing to relax stage",
                step, global_ke
            );
        }
    }
}

/// During relax stage, wait for KE to drop further then advance to compress.
fn check_relaxed(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    if step % 100 != 0 {
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

    if global_ke < 1e-7 {
        next_state.set(Phase::Compress);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE = {:.3e} J — fully relaxed, advancing to compress stage",
                step, global_ke
            );
        }
    }
}
