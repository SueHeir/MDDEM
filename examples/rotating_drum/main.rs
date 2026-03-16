//! Rotating drum DEM simulation — a classic industrial test case.
//!
//! A cylinder wall rotates at constant angular velocity with particles inside.
//! Friction between the drum wall and particles drags particles upward,
//! creating avalanching behavior and a dynamic angle of repose.
//!
//! This demonstrates:
//! - Cylinder walls with prescribed angular velocity
//! - Tangential friction from rotating walls
//! - Gravity-driven granular flow inside a drum
//!
//! ```bash
//! cargo run --example rotating_drum --no-default-features -- examples/rotating_drum/config.toml
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
