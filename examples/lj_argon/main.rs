//! LJ Argon: 108-atom FCC lattice with Nose-Hoover NVT thermostat.
//!
//! The simplest molecular dynamics example — uses `LJDefaultPlugins` for
//! a complete simulation with RDF and MSD measurements.
//!
//! ```bash
//! cargo run --example lj_argon --no-default-features -- examples/lj_argon/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
    app.start();
}
