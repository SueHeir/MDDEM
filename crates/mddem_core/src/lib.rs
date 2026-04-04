//! Core crate for MDDEM.
//!
//! Contains all resource types, plugins, and systems for the base simulation
//! framework. DEM-specific extensions (`DemAtom`, force models) live in
//! separate crates (`dem_atom`, `dem_granular`).

pub mod angle;
pub mod atom;
pub mod bond;
pub mod comm;
pub mod domain;
pub mod group;
pub mod input;
pub mod pair_coeff;
pub mod region;
pub mod run;
pub mod schedule;
pub mod neighbor;
pub mod virial;

/// Internal state controlling the communication/rebuild path each timestep.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum CommState {
    /// Full rebuild: run pbc, exchange, full borders, neighbor build.
    #[default]
    FullRebuild,
    /// Lightweight path: skip pbc/exchange, only forward-comm ghost positions.
    CommunicateOnly,
}

// Re-export all public types at crate root for convenience.
pub use angle::*;
pub use atom::*;
pub use bond::*;
pub use comm::*;
pub use domain::*;
pub use group::{group_includes, Group, GroupDef, GroupPlugin, GroupRegistry};
pub use input::{load_toml, print_banner, Config, Input, InputPlugin};
pub use pair_coeff::{MixingRule, PairCoeffTable};
pub use region::{Axis, Region, SurfaceResult};
pub use run::*;
pub use schedule::*;
pub use neighbor::*;
pub use virial::*;

// Re-export toml so downstream users can build Config tables programmatically.
pub use toml;
