//! DEM granular contact models for spherical particle simulations.
//!
//! This crate provides the core physics for Discrete Element Method (DEM) simulations
//! of granular materials. It implements contact force models, rotational dynamics,
//! and granular temperature output.
//!
//! # Contact models
//!
//! ## Normal contact
//! - **Hertz** (default) — nonlinear elastic contact: `F_n = 4/3 E* √(R* δ) · δ`
//!   with viscoelastic damping proportional to `β √(S_n m_r)`
//! - **Hooke** — linear spring contact: `F_n = k_n δ` with linear damping `γ_n v_n`
//!
//! ## Tangential contact
//! - **Mindlin** — incremental spring-history model with Coulomb friction cap `μ |F_n|`.
//!   Spring displacement is stored per-contact and rotated to stay in the tangent plane
//!   each step. Damping: `γ_t = 2 β √(5/3) √(k_t m_r)`.
//!
//! ## Rolling resistance
//! - **Constant torque** (default) — `τ_r = μ_r |F_n| R*` opposing relative rolling
//! - **SDS** (spring-dashpot-slider) — incremental rolling displacement with spring
//!   stiffness, viscous damping, and Coulomb-style slider cap
//!
//! ## Twisting friction
//! - **Constant torque** (default) — `τ_tw = μ_tw |F_n| R*` opposing relative twisting
//! - **SDS** (spring-dashpot-slider) — incremental twist angle with spring, damping, cap
//!
//! ## Adhesion / cohesion
//! - **JKR** — Johnson-Kendall-Roberts adhesion with extended interaction range beyond
//!   geometric contact; pull-off force `F = 3/2 π γ R*`
//! - **DMT** — Derjaguin-Muller-Toporov adhesion with constant attractive force
//!   `F = 2π γ R*` during contact only
//! - **SJKR** — simplified cohesion proportional to contact area: `F = k_coh π δ R*`
//!
//! # TOML configuration
//!
//! Contact model parameters are set per-material in the `[[materials]]` array:
//!
//! ```toml
//! [[materials]]
//! name = "glass"
//! youngs_modulus = 8.7e9      # Pa — Young's modulus E
//! poisson_ratio = 0.3         # dimensionless — Poisson's ratio ν
//! restitution = 0.95          # dimensionless — coefficient of restitution (0–1)
//! friction = 0.4              # dimensionless — sliding friction coefficient μ
//! rolling_friction = 0.1      # dimensionless — rolling friction coefficient μ_r
//! cohesion_energy = 0.0       # J/m² — SJKR cohesion energy density (0 = disabled)
//! surface_energy = 0.0        # J/m² — JKR/DMT surface energy γ (0 = disabled)
//! ```
//!
//! Global model selection:
//!
//! ```toml
//! [materials]
//! contact_model = "hertz"     # "hertz" (default) or "hooke"
//! adhesion_model = "jkr"      # "jkr" (default) or "dmt" (only when surface_energy > 0)
//! rolling_model = "constant"  # "constant" (default) or "sds"
//! twisting_model = "constant" # "constant" (default) or "sds"
//! ```
//!
//! # Modules
//!
//! - [`contact`] — Fused Hertz-Mindlin + Hooke contact force (primary code path)
//! - [`normal`] — Standalone Hertz normal-only force (for use without tangential)
//! - [`tangential`] — Standalone Mindlin tangential force (for use without fused contact)
//! - [`rotational`] — Quaternion-based velocity Verlet for angular degrees of freedom
//! - [`granular_temp`] — Granular temperature (velocity fluctuation) output

pub mod granular_temp;
pub mod normal;
pub mod rotational;
pub mod tangential;

pub use granular_temp::GranularTempPlugin;
pub use normal::HertzNormalForcePlugin;
pub use rotational::RotationalDynamicsPlugin;
pub use tangential::MindlinTangentialForcePlugin;

pub mod contact;

use grass_app::prelude::*;

use dem_atom::DemAtomPlugin;
use dem_atom::DemAtomInsertPlugin;
use mddem_verlet::VelocityVerletPlugin;

pub use contact::HertzMindlinContactPlugin;

/// Re-export from [`dem_atom`] for convenience.
pub use dem_atom::SQRT_5_3;
/// Small epsilon to avoid division by zero when normalizing tangential,
/// rolling, or twisting spring displacements.
pub const TANGENTIAL_EPSILON: f64 = 1e-30;

/// Warn when `distance / (r1 + r2)` falls below this threshold.
///
/// A ratio near 0.0 means nearly full overlap, which indicates an unstable
/// simulation (timestep too large or bad initial packing). Contacts with
/// overlap exceeding this threshold trigger a warning but still compute
/// forces (capped at half the smaller radius) to prevent runaway penetration.
pub const LARGE_OVERLAP_WARN_THRESHOLD: f64 = 0.90;

/// Maximum overlap warnings per timestep before the simulation panics.
///
/// If more than this many pairs exceed [`LARGE_OVERLAP_WARN_THRESHOLD`],
/// the simulation aborts with an actionable error message suggesting the
/// user reduce the timestep or fix the initial configuration.
pub const MAX_OVERLAP_WARNINGS: usize = 500;

/// Default DEM granular physics plugin group.
///
/// Includes, in registration order:
/// - [`DemAtomPlugin`] — per-atom material properties (radius, density) and
///   `MaterialTable` for per-material Young's modulus, Poisson ratio, restitution,
///   friction with geometric-mean mixing
/// - [`DemAtomInsertPlugin`] — random particle insertion from `[[particles.insert]]` config
/// - [`HertzNormalForcePlugin`] — Hertz elastic contact + viscoelastic normal damping
/// - [`MindlinTangentialForcePlugin`] — Mindlin spring-history tangential force with
///   Coulomb friction cap and torque accumulation; ordered after Hertz via `.after("hertz_normal")`
/// - [`RotationalDynamicsPlugin`] — quaternion Velocity Verlet for angular degrees of freedom
///   (I = 2/5 m r² for solid spheres)
/// - [`GranularTempPlugin`] — granular temperature output to file
///
/// Does **not** include infrastructure plugins (input, comm, domain, neighbor,
/// run, print). Use [`CorePlugins`] to get all infrastructure.
///
/// # Usage
/// ```rust,ignore
/// use mddem::prelude::*;
///
/// let mut app = App::new();
/// app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
/// app.start();
/// ```
pub struct GranularDefaultPlugins;

impl PluginGroup for GranularDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(DemAtomPlugin)
            .add(DemAtomInsertPlugin)
            .add(VelocityVerletPlugin::new())
            .add(HertzMindlinContactPlugin)
            .add(RotationalDynamicsPlugin)
            .add(GranularTempPlugin)
    }
}
