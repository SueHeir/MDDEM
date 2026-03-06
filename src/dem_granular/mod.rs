pub mod normal;
pub mod tangential;
pub mod rotational;

pub use normal::HertzNormalForcePlugin;
pub use tangential::MindlinTangentialForcePlugin;
pub use rotational::RotationalDynamicsPlugin;

use mddem_app::prelude::*;

use crate::dem_atom::DemAtomPlugin;

/// Default DEM granular physics plugin group.
///
/// Includes, in registration order:
/// - [`DemAtomPlugin`] — per-atom material properties (radius, Young's modulus, Poisson ratio,
///   restitution coefficient) and stable timestep computation
/// - [`HertzNormalForcePlugin`] — Hertz elastic contact + viscoelastic normal damping;
///   also clears forces and torques at the start of each step
/// - [`MindlinTangentialForcePlugin`] — Mindlin spring-history tangential force with
///   Coulomb friction cap and torque accumulation; ordered after Hertz via `.after("hertz_normal")`
/// - [`RotationalDynamicsPlugin`] — quaternion Velocity Verlet for angular degrees of freedom
///   (I = 2/5 m r² for solid spheres)
///
/// Does **not** include `VerletPlugin` (translational integration) or `PrintPlugin`
/// (output) — those are simulation infrastructure, not DEM physics.
///
/// # Usage
/// ```rust,ignore
/// App::new()
///     .add_plugins(InputPlugin)
///     .add_plugins(CommincationPlugin)
///     .add_plugins(DomainPlugin)
///     .add_plugins(NeighborPlugin { brute_force: false })
///     .add_plugins(GranularDefaultPlugins)
///     .add_plugins(VerletPlugin)
///     .add_plugins(PrintPlugin)
///     .start();
/// ```
pub struct GranularDefaultPlugins;

impl PluginGroup for GranularDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(DemAtomPlugin)
            .add(HertzNormalForcePlugin)
            .add(MindlinTangentialForcePlugin)
            .add(RotationalDynamicsPlugin)
    }
}
