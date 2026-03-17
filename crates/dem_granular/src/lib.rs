//! DEM granular contact models: Hertz normal, Mindlin tangential, and rotational dynamics.

pub mod granular_temp;
pub mod normal;
pub mod rotational;
pub mod rotational_viscous;
pub mod tangential;

pub use granular_temp::GranularTempPlugin;
pub use normal::HertzNormalForcePlugin;
pub use rotational::RotationalDynamicsPlugin;
pub use rotational_viscous::RotationalViscousPlugin;
pub use tangential::MindlinTangentialForcePlugin;

pub mod contact;

use mddem_app::prelude::*;

use dem_atom::DemAtomPlugin;
use dem_atom::DemAtomInsertPlugin;
use mddem_verlet::VelocityVerletPlugin;

pub use contact::HertzMindlinContactPlugin;

/// Re-export from [`dem_atom`] for convenience.
pub use dem_atom::SQRT_5_3;
/// Epsilon to avoid division by zero in tangential force
pub const TANGENTIAL_EPSILON: f64 = 1e-30;
/// Large overlap warning threshold (ratio of distance to sum of radii)
pub const LARGE_OVERLAP_WARN_THRESHOLD: f64 = 0.90;
/// Max overlap warnings per step before panic
pub const MAX_OVERLAP_WARNINGS: usize = 100;

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
