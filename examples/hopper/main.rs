//! Slot hopper discharge: particles fill a funnel, then flow through an exit.
//!
//! Demonstrates multi-stage simulation with runtime wall control (removing a
//! blocker wall when particles settle) and KE-based stage transitions.
//!
//! ```bash
//! cargo run --example hopper --no-default-features -- examples/hopper/config.toml
//! ```

use mddem::prelude::*;

#[derive(Clone, Debug, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("filling")]
    Filling,
    #[stage("flowing")]
    Flowing,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(StatesPlugin::new(Phase::Filling, ParticleSimScheduleSet::PostFinalIntegration))
        .add_plugins(StageAdvancePlugin::<Phase>::new(ParticleSimScheduleSet::PostFinalIntegration));

    app.add_update_system(
        check_settled.run_if(in_state(Phase::Filling)),
        ParticleSimScheduleSet::PostFinalIntegration,
    );

    app.start();
}

/// Check if particles have settled (KE near zero) and remove the blocker wall.
fn check_settled(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut walls: ResMut<Walls>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    // Wait at least 1000 steps for particles to start moving, then check every 100 steps
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
        walls.deactivate_by_name("blocker");
        next_state.set(Phase::Flowing);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE = {:.3e} J — particles settled, removing blocker wall",
                step, global_ke
            );
        }
    }
}
