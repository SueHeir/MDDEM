//! DEM performance benchmark — measures steps/second for Hertz-Mindlin contact
//! in a periodic granular gas. Used to measure optimization impact.
//!
//! ```bash
//! # Debug build (quick check):
//! cargo run --example dem_benchmark --no-default-features -- examples/dem_benchmark/config.toml
//!
//! # Release build (performance measurement):
//! cargo run --example dem_benchmark --release --no-default-features -- examples/dem_benchmark/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
