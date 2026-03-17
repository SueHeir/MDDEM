//! Lunar Regolith Cohesive Angle of Repose Benchmark
//!
//! Validates JKR adhesion in reduced gravity by simulating a column collapse
//! of cohesive particles. Particles settle in a narrow column (stage 1),
//! then the column walls are removed and the pile collapses (stage 2).
//! The final pile angle depends on friction, cohesion, and gravity.
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --example bench_lunar_regolith \
//!   -- examples/bench_lunar_regolith/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin);

    // Remove column walls at the start of the collapse stage
    app.add_setup_system(
        remove_column_walls.run_if(in_stage("collapse")),
        ScheduleSetupSet::PostSetup,
    );

    app.start();
}

/// Remove the column containment walls at the start of the collapse stage.
fn remove_column_walls(comm: Res<CommResource>, mut walls: ResMut<Walls>) {
    walls.deactivate_by_name("column_left");
    walls.deactivate_by_name("column_right");

    if comm.rank() == 0 {
        println!("==> Column walls removed — pile collapsing");
    }
}
