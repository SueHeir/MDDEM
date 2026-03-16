//! LJ performance benchmark: minimal plugin set for profiling.
//!
//! Runs LJ with only lattice, force, and thermostat plugins — no
//! measurements or extra output — to isolate core simulation performance.
//!
//! ```bash
//! cargo run --example lj_performance --release --no-default-features -- examples/lj_performance/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(NoseHooverPlugin);
    app.start();
}
