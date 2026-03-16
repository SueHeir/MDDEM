//! Simple shear flow example for granular rheology.
//!
//! Top and bottom walls move in opposite x-directions to create a linear
//! shear profile. Measures velocity distribution and stress response.
//!
//! ```bash
//! cargo run --example granular_shear --no-default-features -- examples/granular_shear/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(WallPlugin)
        .add_plugins(GravityPlugin)
        .add_plugins(VelocityDistributionPlugin);
    app.start();
}
