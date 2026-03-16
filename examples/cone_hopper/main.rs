//! Conical hopper discharge: particles drain through a cone using region-surface walls.
//!
//! Demonstrates the `type = "region"` wall with a `Cone` region.
//! Particles are inserted above the cone, settle under gravity,
//! then a blocker wall is removed to allow flow through the narrow end.
//!
//! ```bash
//! cargo run --example cone_hopper --no-default-features -- examples/cone_hopper/config.toml
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
        .add_plugins(StatesPlugin {
            initial: Phase::Filling,
        })
        .add_plugins(StageAdvancePlugin::<Phase>::new());

    app.add_update_system(
        check_settled.run_if(in_state(Phase::Filling)),
        ScheduleSet::PostFinalIntegration,
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
