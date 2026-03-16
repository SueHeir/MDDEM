//! Poiseuille flow: LJ fluid between frozen wall slabs with a body force.
//!
//! Demonstrates groups (wall/fluid), freeze fix, addforce fix, and
//! Langevin thermostat applied to a subset of atoms.
//!
//! ```bash
//! cargo run --example poiseuille_flow --no-default-features -- examples/poiseuille_flow/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(FixesPlugin)
        .add_plugins(MeasurePlugin);
    app.start();
}
