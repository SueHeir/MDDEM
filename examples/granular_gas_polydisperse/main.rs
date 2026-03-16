//! Polydisperse granular gas — measures per-species granular temperature
//! and velocity distribution deviations from monodisperse Maxwell-Boltzmann.
//!
//! This example demonstrates energy equipartition breakdown in a binary
//! granular gas with two particle sizes. In polydisperse dissipative systems,
//! each species develops its own granular temperature, with lighter particles
//! typically running hotter than heavier ones.
//!
//! ```bash
//! cargo run --example granular_gas_polydisperse --no-default-features -- examples/granular_gas_polydisperse/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(VelocityDistributionPlugin);
    app.start();
}
