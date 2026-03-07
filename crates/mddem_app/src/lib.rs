mod app;
mod plugin;
mod sub_app;

pub use app::*;
pub use plugin::*;
pub use sub_app::*;

/// The app prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use crate::{
        app::App, app::ConfigSnippets, app::GenerateConfigFlag, sub_app::SubApp, Plugin,
        PluginGroup, PluginGroupBuilder, StatesPlugin,
    };
}
