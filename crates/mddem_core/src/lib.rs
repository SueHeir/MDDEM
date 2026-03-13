//! Core crate for MDDEM.
//!
//! Contains all resource types, plugins, and systems for the base simulation
//! framework. DEM-specific extensions (`DemAtom`, force models) live in
//! separate crates (`dem_atom`, `dem_granular`).

pub mod atom;
pub mod bond;
pub mod comm;
pub mod domain;
pub mod group;
pub mod input;
pub mod run;
pub mod virial;

// Re-export all public types at crate root for convenience.
pub use atom::*;
pub use bond::*;
pub use comm::*;
pub use domain::*;
pub use group::{group_includes, Group, GroupDef, GroupPlugin, GroupRegistry};
pub use input::{load_toml, print_banner, Config, Input, InputPlugin};
pub use run::*;
pub use virial::*;

// Re-export toml so downstream users can build Config tables programmatically.
pub use toml;
