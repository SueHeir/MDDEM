//! Basic 3D granular gas: 500 particles in a periodic box.
//!
//! Uses the standard plugin stack with a TOML config file.
//!
//! ```bash
//! cargo run --example granular_basic --no-default-features -- examples/granular_basic/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
