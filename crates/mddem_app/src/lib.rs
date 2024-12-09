mod app;
mod sub_app;
mod plugin;

pub use app::*;
pub use sub_app::*;
pub use plugin::*;



/// The app prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use crate::{
        app::App,
        sub_app::SubApp,
        Plugin,
    };
}
