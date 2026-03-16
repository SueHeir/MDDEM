//! Top-level MDDEM crate. Re-exports all sub-crates and provides [`CorePlugins`],
//! [`GranularDefaultPlugins`](dem_granular::GranularDefaultPlugins), and [`LJDefaultPlugins`]
//! plugin groups for quick setup.

pub use dem_atom;
pub use dem_bond;
pub use dem_measure_plane;
pub use dem_granular;
pub use dem_thermal;
pub use dem_velocity_distribution;
pub use dem_wall;
pub use md_lattice;
pub use md_lj;
pub use md_measure;
pub use md_type_rdf;
pub use md_thermostat;
pub use mddem_core;
pub use mddem_fixes;
pub use mddem_neighbor;
pub use mddem_print;
pub use mddem_verlet;

use mddem_app::prelude::*;

/// Core simulation infrastructure plugin group.
///
/// Includes, in registration order:
/// - [`InputPlugin`](mddem_core::InputPlugin) — CLI parsing, banner printing, TOML config loading
///   (skipped if `Config` is already present)
/// - [`CommunicationPlugin`](mddem_core::CommunicationPlugin) —
///   Unified MPI or single-process communication backend (selected by `mpi_backend` feature)
/// - [`DomainPlugin`](mddem_core::DomainPlugin) — Cartesian domain decomposition
/// - [`NeighborPlugin`](mddem_neighbor::NeighborPlugin) — sweep-and-prune neighbor lists
/// - [`RunPlugin`](mddem_core::RunPlugin) — run/cycle management
/// - [`PrintPlugin`](mddem_print::PrintPlugin) — thermo output, dump files, restart files
///
/// **Note:** Velocity Verlet integration is **not** included here. Use
/// [`VelocityVerletPlugin`](mddem_verlet::VelocityVerletPlugin) directly, or rely on
/// [`NoseHooverPlugin`](md_thermostat::NoseHooverPlugin) which provides fused integration.
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
        let builder = PluginGroupBuilder::start::<Self>().add(mddem_core::InputPlugin);

        builder
            .add(mddem_core::CommunicationPlugin)
            .add(mddem_core::DomainPlugin::default())
            .add(mddem_neighbor::NeighborPlugin {
                style: mddem_neighbor::NeighborStyle::Bin,
            })
            .add(mddem_core::GroupPlugin)
            .add(mddem_core::RunPlugin)
            .add(mddem_print::PrintPlugin)
    }
}

/// Lennard-Jones simulation plugin group.
///
/// Includes:
/// - [`LatticePlugin`](md_lattice::LatticePlugin) — FCC lattice initialization
/// - [`LJForcePlugin`](md_lj::LJForcePlugin) — LJ 12-6 pair force + virial
/// - [`NoseHooverPlugin`](md_thermostat::NoseHooverPlugin) — NVT thermostat with fused Velocity Verlet integration
/// - [`MeasurePlugin`](md_measure::MeasurePlugin) — RDF, MSD, pressure measurements
pub struct LJDefaultPlugins;

impl PluginGroup for LJDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(md_lattice::LatticePlugin)
            .add(md_lj::LJForcePlugin)
            .add(md_thermostat::NoseHooverPlugin)
            .add(md_measure::MeasurePlugin)
    }
}

pub mod prelude {
    pub use crate::{CorePlugins, LJDefaultPlugins};
    pub use dem_atom::{DemAtomPlugin, DemConfig, MaterialTable};
    pub use dem_atom::{DemAtomInsertPlugin, ParticlesConfig};
    pub use dem_bond::DemBondPlugin;
    pub use dem_granular::GranularDefaultPlugins;
    pub use dem_velocity_distribution::VelocityDistributionPlugin;
    pub use mddem_fixes::{GravityConfig, GravityPlugin};
    pub use dem_thermal::{ThermalConfig, ThermalPlugin};
    pub use dem_measure_plane::{MeasurePlaneDef, MeasurePlanePlugin, MeasurePlanes};
    pub use dem_wall::{WallDef, WallPlane, WallPlugin, Walls};
    pub use md_lattice::{LatticeConfig, LatticePlugin};
    pub use md_lj::{LJConfig, LJForcePlugin, LJPairTable, LJTailCorrections};
    pub use md_measure::{MeasureConfig, MeasurePlugin};
    pub use md_type_rdf::{TypeRdfConfig, TypeRdfPlugin};
    pub use md_thermostat::{LangevinConfig, LangevinPlugin, LangevinState, NoseHooverPlugin, NoseHooverState, ThermostatConfig};
    pub use mddem_fixes::{AddForceDef, FixesPlugin, FixesRegistry, FreezeDef, MoveLinearDef, SetForceDef, ViscousDef};
    pub use mddem_app::prelude::*;
    pub use mddem_derive::StageEnum;
    pub use mddem_core::*;
    pub use mddem_neighbor::*;
    pub use mddem_print::*;
    pub use mddem_scheduler::prelude::*;
    pub use mddem_verlet::*;
}
