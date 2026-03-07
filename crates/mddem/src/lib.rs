pub use mddem_core;
pub use mddem_neighbor;
pub use mddem_verlet;
pub use mddem_print;
pub use dem_atom;
pub use dem_atom_insert;
pub use dem_granular;
pub use dem_gravity;
pub use dem_wall;

use mddem_app::prelude::*;

/// Core simulation infrastructure plugin group.
///
/// Includes, in registration order:
/// - [`InputPlugin`](mddem_core::InputPlugin) — CLI parsing, banner printing, TOML config loading
///   (skipped if `Config` is already present)
/// - [`CommunicationPlugin`](mddem_core::CommunicationPlugin) / [`SingleProcessCommPlugin`](mddem_core::SingleProcessCommPlugin) —
///   MPI or single-process communication backend (selected by `mpi_backend` feature)
/// - [`DomainPlugin`](mddem_core::DomainPlugin) — Cartesian domain decomposition
/// - [`NeighborPlugin`](mddem_neighbor::NeighborPlugin) — sweep-and-prune neighbor lists
/// - [`RunPlugin`](mddem_core::RunPlugin) — run/cycle management
/// - [`VelocityVerletPlugin`](mddem_verlet::VelocityVerletPlugin) — translational Velocity Verlet integration
/// - [`PrintPlugin`](mddem_print::PrintPlugin) — thermo output, dump files, restart files
///
/// MPI finalization is registered as a cleanup callback and runs automatically
/// at the end of [`App::start()`].
///
/// # Usage
/// ```rust,ignore
/// use mddem::prelude::*;
///
/// let mut app = App::new();
/// app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
/// app.start();
/// ```
pub struct CorePlugins;

impl PluginGroup for CorePlugins {
    fn build(self) -> PluginGroupBuilder {
        let builder = PluginGroupBuilder::start::<Self>()
            .add(mddem_core::InputPlugin);

        #[cfg(feature = "mpi_backend")]
        let builder = builder.add(mddem_core::CommunicationPlugin);
        #[cfg(not(feature = "mpi_backend"))]
        let builder = builder.add(mddem_core::SingleProcessCommPlugin);

        builder
            .add(mddem_core::DomainPlugin::default())
            .add(mddem_neighbor::NeighborPlugin { style: mddem_neighbor::NeighborStyle::SweepAndPrune })
            .add(mddem_core::RunPlugin)
            .add(mddem_verlet::VelocityVerletPlugin)
            .add(mddem_print::PrintPlugin)
    }
}

pub mod prelude {
    pub use mddem_app::prelude::*;
    pub use mddem_scheduler::prelude::*;
    pub use mddem_core::*;
    pub use mddem_neighbor::*;
    pub use mddem_verlet::*;
    pub use mddem_print::*;
    pub use dem_atom::{DemAtomPlugin, DemConfig, MaterialTable};
    pub use dem_atom_insert::{DemAtomInsertPlugin, ParticlesConfig};
    pub use dem_granular::GranularDefaultPlugins;
    pub use dem_gravity::{GravityPlugin, GravityConfig};
    pub use dem_wall::{WallPlugin, Walls, WallDef, WallPlane};
    pub use crate::CorePlugins;
}
