//! Granular gas with velocity distribution analysis.
//!
//! Runs a dissipative granular gas in a periodic box and measures the velocity
//! distribution at regular intervals, comparing against Maxwell-Boltzmann theory.
//!
//! ```bash
//! cargo run --example granular_gas_vdist --no-default-features -- examples/granular_gas_vdist/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(VelocityDistributionPlugin);
    app.start();
}
