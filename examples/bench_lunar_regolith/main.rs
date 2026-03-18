//! Lunar Regolith Cohesive Angle of Repose Benchmark
//!
//! Validates JKR adhesion in reduced gravity using a draining-box method:
//! 1. **Settle**: Pack particles in a walled box, let them settle under gravity.
//! 2. **Drain**: Remove the "gate" wall, particles flow out the open side.
//! 3. **Measure**: Let remaining pile settle completely, final dump captures angle.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --example bench_lunar_regolith \
//!   -- examples/bench_lunar_regolith/config.toml
//! ```

use mddem::prelude::*;

#[derive(Clone, Debug, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("settle")]
    Settle,
    #[stage("drain")]
    Drain,
    #[stage("measure")]
    Measure,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(StatesPlugin::new(Phase::Settle, ScheduleSet::PostFinalIntegration))
        .add_plugins(StageAdvancePlugin::<Phase>::new(ScheduleSet::PostFinalIntegration));

    // During settle: check KE and advance early if settled
    app.add_update_system(
        check_settled.run_if(in_state(Phase::Settle)),
        ScheduleSet::PostFinalIntegration,
    );

    // On entering drain: ensure gate is removed (works whether settle
    // ended early via check_settled or ran out of steps)
    app.add_update_system(
        open_gate.run_if(on_enter_state(Phase::Drain)),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

/// Check if particles have settled by monitoring per-particle KE.
/// When settled, deactivate the "gate" wall and advance to Drain phase.
fn check_settled(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut walls: ResMut<Walls>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    // Wait at least 5000 steps, then check every 200 steps
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

    let ke_per_particle = if global_n > 0.0 {
        global_ke / global_n
    } else {
        0.0
    };

    if ke_per_particle < 1e-6 {
        walls.deactivate_by_name("gate");
        next_state.set(Phase::Drain);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE/particle = {:.3e} J (N = {:.0}) — settled, removing gate",
                step, ke_per_particle, global_n
            );
        }
    }
}

/// Remove the gate wall when entering the drain phase.
/// This fires exactly once via on_enter_state, regardless of how the
/// transition happened (early KE threshold or stage step exhaustion).
fn open_gate(
    comm: Res<CommResource>,
    mut walls: ResMut<Walls>,
) {
    walls.deactivate_by_name("gate");
    if comm.rank() == 0 {
        println!("Drain phase entered — gate removed");
    }
}
