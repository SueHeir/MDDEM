//! Haff's cooling law benchmark — validates energy dissipation against
//! theoretical cooling curves for a granular gas in a periodic box.
//!
//! ```bash
//! cargo run --example benchmark --no-default-features -- examples/benchmark/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
