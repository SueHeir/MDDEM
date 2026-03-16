//! LJ Argon with Langevin thermostat (stochastic dynamics).
//!
//! Same setup as `lj_argon` but uses a Langevin thermostat instead of
//! Nose-Hoover. Demonstrates manual plugin selection without `LJDefaultPlugins`.
//!
//! ```bash
//! cargo run --example lj_langevin --no-default-features -- examples/lj_langevin/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(MeasurePlugin);
    app.start();
}
