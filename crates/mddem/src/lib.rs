//! # MDDEM — Molecular Dynamics / Discrete Element Method Simulation Framework
//!
//! MDDEM is a plugin-based simulation framework for particle simulations,
//! supporting both **Discrete Element Method (DEM)** for granular materials and
//! **Molecular Dynamics (MD)** for atomistic systems.
//!
//! This crate is the main entry point. It re-exports every plugin crate and
//! provides convenience plugin groups ([`CorePlugins`], [`LJDefaultPlugins`])
//! so that a complete simulation can be set up in just a few lines.
//!
//! # Quick Start
//!
//! ## DEM granular simulation
//!
//! ```rust,ignore
//! use mddem::prelude::*;
//!
//! fn main() {
//!     let mut app = App::new();
//!     app.add_plugins(CorePlugins)
//!        .add_plugins(GranularDefaultPlugins);
//!     app.start();
//! }
//! ```
//!
//! ## MD Lennard-Jones simulation
//!
//! ```rust,ignore
//! use mddem::prelude::*;
//!
//! fn main() {
//!     let mut app = App::new();
//!     app.add_plugins(CorePlugins)
//!        .add_plugins(LJDefaultPlugins);
//!     app.start();
//! }
//! ```
//!
//! Both examples read their parameters from a TOML config file passed as a
//! command-line argument (e.g. `cargo run -- config.toml`).
//!
//! # Architecture
//!
//! Every feature is implemented as a [`Plugin`](sim_app::Plugin) that
//! registers systems into a schedule. Plugins are grouped into **plugin groups**
//! for convenience:
//!
//! | Plugin Group | Purpose |
//! |---|---|
//! | [`CorePlugins`] | Infrastructure: input, communication, domain, neighbor lists, run loop, output |
//! | [`GranularDefaultPlugins`](dem_granular::GranularDefaultPlugins) | DEM: Hertz–Mindlin contact, rotational dynamics, particle insertion |
//! | [`LJDefaultPlugins`] | MD: LJ 12-6 pair force, Nose–Hoover thermostat, lattice init, measurements |
//!
//! For finer control, add individual plugins instead of a plugin group.
//!
//! # Crate Organization
//!
//! ## Infrastructure crates (`mddem_*`)
//!
//! | Crate | Description |
//! |---|---|
//! | [`sim_app`] | Application framework: [`App`](sim_app::App), [`Plugin`](sim_app::Plugin) trait, ECS-style resources |
//! | [`mddem_core`] | Core simulation types: [`Atom`](mddem_core::Atom), [`Config`](mddem_core::Config), domain, communication, regions |
//! | [`sim_scheduler`] | System scheduler with [`ScheduleSet`](sim_scheduler::ScheduleSet) ordering |
//! | [`mddem_neighbor`] | Sweep-and-prune neighbor list construction |
//! | [`mddem_verlet`] | Velocity Verlet time integration |
//! | [`mddem_print`] | Thermo output, dump files (CSV/binary/VTP), restart files |
//! | [`mddem_derive`] | Derive macros: `#[derive(AtomData)]`, `#[derive(StageEnum)]` |
//! | [`mddem_deform`] | Box deformation: engineering strain rate, velocity, target size |
//! | [`mddem_fixes`] | General-purpose fixes: gravity, addforce, setforce, freeze, movelinear, viscous |
//! | [`mddem_velocity_distribution`] | Initial velocity distributions (Gaussian, uniform) |
//!
//! ## DEM crates (`dem_*`)
//!
//! | Crate | Description |
//! |---|---|
//! | [`dem_atom`] | DEM per-atom data (`DemAtom`), material table, particle insertion, size distributions |
//! | [`dem_granular`] | Hertz normal, Mindlin tangential, rotational dynamics, granular temperature |
//! | [`dem_bond`] | Inter-particle bonds: normal/tangential/bending, auto-bonding, breakage |
//! | [`dem_wall`] | Wall definitions: plane, cylinder, sphere, cone, region surfaces; wall motion |
//! | [`dem_thermal`] | Heat conduction between contacting particles |
//! | [`dem_contact_analysis`] | Contact statistics: coordination number, fabric tensor, rattlers, per-contact CSV |
//! | [`dem_measure_plane`] | Measurement planes for flux and profile sampling |
//!
//! ## MD crates (`md_*`)
//!
//! | Crate | Description |
//! |---|---|
//! | [`md_lj`] | Lennard-Jones 12-6 pair potential with tail corrections |
//! | [`md_thermostat`] | Nose–Hoover NVT and Langevin thermostats |
//! | [`md_lattice`] | FCC / BCC / SC lattice initialization |
//! | [`md_measure`] | Pressure, temperature, and energy measurements |
//! | [`md_type_rdf`] | Type-filtered radial distribution function |
//! | [`md_msd`] | Mean squared displacement per atom type |
//! | [`md_bond`] | FENE and harmonic bond potentials |
//! | [`md_polymer`] | Polymer chain initialization, end-to-end distance, radius of gyration |
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---|---|---|
//! | `mpi_backend` | **yes** | Enable MPI-based parallel communication. Disable with `--no-default-features` for single-process runs. |
//!
//! # The Prelude
//!
//! The [`prelude`] module re-exports the most commonly used types so you can
//! write `use mddem::prelude::*;` and have everything you need for a typical
//! simulation. See the [`prelude`] module docs for the full list.

// ---------------------------------------------------------------------------
// Sub-crate re-exports
// ---------------------------------------------------------------------------

// --- DEM crates ---

/// DEM per-atom data, material properties, particle insertion, and size distributions.
pub use dem_atom;

/// Inter-particle bonds for DEM: normal, tangential, bending modes with auto-bonding and breakage.
pub use dem_bond;
/// Rigid clump (multisphere) composites for non-spherical DEM particles.
pub use dem_clump;

/// Contact analysis: coordination number, fabric tensor, rattler detection, per-contact CSV output.
pub use dem_contact_analysis;

/// Measurement planes for sampling particle flux and spatial profiles.
pub use dem_measure_plane;

/// DEM granular contact models: Hertz normal, Mindlin tangential, rotational dynamics.
pub use dem_granular;

/// Heat conduction between contacting DEM particles.
pub use dem_thermal;

/// Wall definitions (plane, cylinder, sphere, cone, region surfaces) and wall motion.
pub use dem_wall;

// --- MD crates ---

/// FCC / BCC / SC lattice initialization for molecular dynamics.
pub use md_lattice;

/// Lennard-Jones 12-6 pair potential with optional tail corrections.
pub use md_lj;

/// FENE and harmonic bond potentials for molecular dynamics.
pub use md_bond;

/// Pressure, temperature, and energy measurement systems.
pub use md_measure;

/// Mean squared displacement tracking per atom type.
pub use md_msd;

/// Type-filtered radial distribution function.
pub use md_type_rdf;

/// Polymer chain initialization, end-to-end distance, and radius of gyration.
pub use md_polymer;

/// Nose–Hoover NVT and Langevin thermostats with fused Velocity Verlet integration.
pub use md_thermostat;

// --- Infrastructure crates ---

/// Core simulation types: atoms, config, domain decomposition, communication, regions, groups.
pub use mddem_core;

/// Box deformation: engineering strain rate, velocity, and target-size modes.
pub use mddem_deform;

/// General-purpose fixes: gravity, addforce, setforce, freeze, movelinear, viscous.
pub use mddem_fixes;

/// Sweep-and-prune neighbor list construction.
pub use mddem_neighbor;

/// Thermo output, dump files (CSV/binary/VTP), and restart file I/O.
pub use mddem_print;

/// Velocity Verlet time integration for translational degrees of freedom.
pub use mddem_verlet;

/// Initial velocity distributions (Gaussian, uniform) for particle initialization.
pub use mddem_velocity_distribution;

use sim_app::prelude::*;

/// Core simulation infrastructure plugin group.
///
/// Includes, in registration order:
/// - [`InputPlugin`](mddem_core::InputPlugin) — CLI parsing, banner printing, TOML config loading
///   (skipped if `Config` is already present)
/// - [`CommunicationPlugin`](mddem_core::CommunicationPlugin) —
///   Unified MPI or single-process communication backend (selected by `mpi_backend` feature)
/// - [`DomainPlugin`](mddem_core::DomainPlugin) — Cartesian domain decomposition
/// - [`NeighborPlugin`](mddem_neighbor::NeighborPlugin) — sweep-and-prune neighbor lists
/// - [`GroupPlugin`](mddem_core::GroupPlugin) — atom group definitions and filtering
/// - [`RunPlugin`](mddem_core::RunPlugin) — run/cycle management
/// - [`PrintPlugin`](mddem_print::PrintPlugin) — thermo output, dump files, restart files
///
/// **Note:** Velocity Verlet integration is **not** included here. Use
/// [`VelocityVerletPlugin`](mddem_verlet::VelocityVerletPlugin) directly, or rely on
/// [`NoseHooverPlugin`](md_thermostat::NoseHooverPlugin) which provides fused integration.
/// The DEM plugin group [`GranularDefaultPlugins`](dem_granular::GranularDefaultPlugins)
/// includes Velocity Verlet automatically.
///
/// MPI finalization is registered as a cleanup callback and runs automatically
/// at the end of [`App::start()`](sim_app::App::start).
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
            .add(|app: &mut App| {
                app.set_warning_fn(mddem_core::verlet_schedule_warnings);
            })
            .add(mddem_core::InputPlugin);

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
/// A convenience group for molecular dynamics simulations using the
/// Lennard-Jones 12-6 pair potential. Includes:
///
/// - [`LatticePlugin`](md_lattice::LatticePlugin) — FCC / BCC / SC lattice initialization
/// - [`LJForcePlugin`](md_lj::LJForcePlugin) — LJ 12-6 pair force with virial contribution
/// - [`NoseHooverPlugin`](md_thermostat::NoseHooverPlugin) — NVT thermostat with fused
///   Velocity Verlet integration
/// - [`MeasurePlugin`](md_measure::MeasurePlugin) — pressure, temperature, and energy measurements
///
/// For Langevin dynamics, replace this group with individual plugins and use
/// [`LangevinPlugin`](md_thermostat::LangevinPlugin) instead of
/// [`NoseHooverPlugin`](md_thermostat::NoseHooverPlugin).
///
/// # Usage
/// ```rust,ignore
/// use mddem::prelude::*;
///
/// let mut app = App::new();
/// app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
/// app.start();
/// ```
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

/// The MDDEM prelude — import everything you need for a typical simulation.
///
/// ```rust,ignore
/// use mddem::prelude::*;
/// ```
///
/// This re-exports:
///
/// ## Plugin groups
/// - [`CorePlugins`] — infrastructure (input, comm, domain, neighbor, run, print)
/// - [`LJDefaultPlugins`] — LJ 12-6 + Nose–Hoover + lattice + measurements
/// - [`GranularDefaultPlugins`] — Hertz–Mindlin + rotational dynamics + particle insertion
///
/// ## DEM types
/// - [`DemAtomPlugin`], [`DemConfig`], [`MaterialTable`] — DEM atom data and materials
/// - [`DemAtomInsertPlugin`], [`ParticlesConfig`] — particle insertion
/// - [`DemBondPlugin`] — inter-particle bonds
/// - [`WallPlugin`], [`Walls`], [`WallDef`], [`WallPlane`] — wall definitions
/// - [`ThermalPlugin`], [`ThermalConfig`] — heat conduction
/// - [`ContactAnalysisPlugin`], [`ContactAnalysisConfig`] — contact statistics
/// - [`MeasurePlanePlugin`], [`MeasurePlanes`], [`MeasurePlaneDef`] — measurement planes
///
/// ## MD types
/// - [`LatticePlugin`], [`LatticeConfig`] — lattice initialization
/// - [`LJForcePlugin`], [`LJConfig`], [`LJPairTable`], [`LJTailCorrections`] — LJ potential
/// - [`MdBondPlugin`], [`MdBondConfig`] — FENE and harmonic bonds
/// - [`MeasurePlugin`], [`MeasureConfig`] — pressure/energy measurements
/// - [`TypeMsdPlugin`], [`MsdConfig`] — mean squared displacement
/// - [`TypeRdfPlugin`], [`TypeRdfConfig`] — radial distribution function
/// - [`PolymerPlugin`], [`PolymerInitPlugin`], [`ChainStatsPlugin`] — polymer tools
/// - [`NoseHooverPlugin`], [`LangevinPlugin`] — thermostats
///
/// ## Shared types
/// - [`DeformPlugin`], [`DeformConfig`], [`DeformState`] — box deformation
/// - [`FixesPlugin`], [`GravityPlugin`] — general-purpose fixes
/// - [`VelocityDistributionPlugin`] — initial velocity assignment
/// - [`StageEnum`] — derive macro for multi-stage runs
///
/// ## Core framework (via glob re-exports)
/// - [`App`](sim_app::App), [`Plugin`](sim_app::Plugin),
///   [`PluginGroup`](sim_app::PluginGroup) — application framework
/// - [`Atom`](mddem_core::Atom), [`Config`](mddem_core::Config),
///   [`RunState`](mddem_core::RunState) — core simulation types
/// - [`Res`](sim_scheduler::Res), [`ResMut`](sim_scheduler::ResMut) — resource accessors
/// - [`ScheduleSet`](sim_scheduler::ScheduleSet) — system ordering
pub mod prelude {
    // Plugin groups defined in this crate
    pub use crate::{CorePlugins, LJDefaultPlugins};

    // DEM plugins and config types
    pub use dem_atom::{DemAtomPlugin, DemConfig, MaterialTable};
    pub use dem_atom::{DemAtomInsertPlugin, ParticlesConfig};
    pub use dem_bond::DemBondPlugin;
    pub use dem_clump::{ClumpPlugin, ClumpRegistry, ClumpAtom, ClumpDef};
    pub use dem_granular::GranularDefaultPlugins;
    pub use dem_thermal::{ThermalConfig, ThermalPlugin};
    pub use dem_contact_analysis::{ContactAnalysisConfig, ContactAnalysisPlugin};
    pub use dem_measure_plane::{MeasurePlaneDef, MeasurePlanePlugin, MeasurePlanes};
    pub use dem_wall::{WallDef, WallMotion, WallPlane, WallPlugin, Walls};

    // MD plugins and config types
    pub use md_lattice::{LatticeConfig, LatticePlugin};
    pub use md_lj::{LJConfig, LJForcePlugin, LJPairTable, LJTailCorrections};
    pub use md_bond::{MdBondConfig, MdBondPlugin};
    pub use md_measure::{MeasureConfig, MeasurePlugin};
    pub use md_msd::{MsdConfig, TypeMsdPlugin};
    pub use md_type_rdf::{TypeRdfConfig, TypeRdfPlugin};
    pub use md_polymer::{ChainStatsConfig, ChainStatsPlugin, PolymerConfig, PolymerInitConfig, PolymerInitPlugin, PolymerPlugin};
    pub use md_thermostat::{LangevinConfig, LangevinPlugin, LangevinState, NoseHooverPlugin, NoseHooverState, ThermostatConfig};

    // Shared infrastructure plugins
    pub use mddem_deform::{DeformConfig, DeformPlugin, DeformState};
    pub use mddem_fixes::{AddForceDef, FixesPlugin, FixesRegistry, FreezeDef, GravityConfig, GravityPlugin, MoveLinearDef, SetForceDef, ViscousDef};
    pub use mddem_velocity_distribution::VelocityDistributionPlugin;

    // Derive macros
    pub use mddem_derive::{SchedulePhase, StageEnum};

    // Core framework re-exports (glob)
    pub use sim_app::prelude::*;
    pub use mddem_core::*;
    pub use mddem_neighbor::*;
    pub use mddem_print::*;
    pub use sim_scheduler::prelude::*;
    pub use mddem_verlet::*;
}
