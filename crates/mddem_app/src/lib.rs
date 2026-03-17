//! Plugin-based application framework for MDDEM.
//!
//! Provides [`App`] as the central container and the [`Plugin`] trait for modular registration
//! of resources and systems.

mod app;
mod plugin;
mod sub_app;

pub use app::*;
pub use plugin::*;
pub use sub_app::*;

/// The `mddem_app` prelude.
///
/// Re-exports the most commonly used types so plugins can import them with a
/// single `use mddem_app::prelude::*;`.
pub mod prelude {
    pub use crate::{
        app::App, app::ConfigSnippets, app::GenerateConfigFlag, sub_app::SubApp, Plugin,
        PluginGroup, PluginGroupBuilder, StageAdvancePlugin, StageNames, StatesPlugin,
    };
}
