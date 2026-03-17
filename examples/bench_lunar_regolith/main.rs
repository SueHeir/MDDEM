//! Lunar Regolith Cohesive Angle of Repose Benchmark
//!
//! Validates JKR adhesion in reduced gravity by simulating a funnel-pour
//! of cohesive particles. Particles are inserted from a narrow slot above
//! the center of a flat floor and naturally form a conical/triangular pile.
//! The pile angle depends on friction, cohesion, and gravity.
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

    app.start();
}
