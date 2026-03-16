//! Shrink-wrap boundary demo: particles falling under gravity.
//!
//! The z-axis uses shrink-wrap boundaries, so the box top automatically
//! tracks the highest particle. As particles settle, the box shrinks.
//! X and Y axes are periodic.
//!
//! ```bash
//! cargo run --example shrink_wrap_demo --no-default-features -- examples/shrink_wrap_demo/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
