//! Group freeze: per-stage group redefinition with frozen atom slabs.
//!
//! Two-stage LJ simulation where the frozen region shrinks between stages,
//! demonstrating `[[run.group]]` per-stage overrides.
//!
//! ```bash
//! cargo run --example group_freeze --no-default-features -- examples/group_freeze/config.toml
//! ```

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(LJDefaultPlugins)
        .add_plugins(FixesPlugin);
    app.start();
}
