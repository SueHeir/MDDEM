//! Haff's cooling law benchmark — validates energy dissipation against
//! theoretical cooling curves for a granular gas in a periodic box.
//!
//! ```bash
//! cargo run --example granular_gas_benchmark --no-default-features -- examples/granular_gas_benchmark/run_debug.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
