//! Quaternion-based rotational dynamics for DEM spheres.
//!
//! Implements velocity Verlet integration for angular degrees of freedom
//! assuming solid spheres with moment of inertia `I = 2/5 m r²`.
//!
//! # Integration scheme
//!
//! **Initial integration** (before force computation):
//! 1. Half-step angular velocity: `ω += ½ dt τ / I`
//! 2. Update quaternion: `q = Δq · q` where `Δq = (cos(θ/2), sin(θ/2) ω̂)` and `θ = |ω| dt`
//!
//! **Final integration** (after force computation):
//! 1. Half-step angular velocity: `ω += ½ dt τ / I`
//!
//! This mirrors the standard velocity Verlet for translational motion,
//! applied to the rotational degrees of freedom with quaternion orientation tracking.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use dem_atom::DemAtom;
use mddem_core::{Atom, AtomDataRegistry, ParticleSimScheduleSet};

/// Construct a unit quaternion `[w, x, y, z]` from a unit rotation axis and angle (radians).
#[inline]
fn quat_from_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let half = angle * 0.5;
    let s = half.sin();
    [half.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Multiply two quaternions `[w, x, y, z]` using the Hamilton product.
#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ]
}

/// Quaternion-based velocity Verlet integrator for angular degrees of freedom.
///
/// Assumes solid spheres with moment of inertia `I = 2/5 m r²`. Registers
/// two systems:
/// - [`initial_rotation`] at [`ParticleSimScheduleSet::InitialIntegration`] — half-step ω, update quaternion
/// - [`final_rotation`] at [`ParticleSimScheduleSet::FinalIntegration`] — half-step ω after new torques
pub struct RotationalDynamicsPlugin;

impl Plugin for RotationalDynamicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_rotation, ParticleSimScheduleSet::InitialIntegration)
            .add_update_system(final_rotation, ParticleSimScheduleSet::FinalIntegration);
    }
}

/// Initial half-step: advance angular velocity and update quaternion orientation.
pub fn initial_rotation(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("initial_rotation");
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inv_inertia = dem.inv_inertia[i];
        if inv_inertia == 0.0 { continue; } // Skip clump sub-spheres

        dem.omega[i][0] += 0.5 * dt * dem.torque[i][0] * inv_inertia;
        dem.omega[i][1] += 0.5 * dt * dem.torque[i][1] * inv_inertia;
        dem.omega[i][2] += 0.5 * dt * dem.torque[i][2] * inv_inertia;

        let ox = dem.omega[i][0];
        let oy = dem.omega[i][1];
        let oz = dem.omega[i][2];
        let omega_mag = (ox*ox + oy*oy + oz*oz).sqrt();
        let angle = omega_mag * dt;
        if angle > 1e-14 {
            let inv = 1.0 / omega_mag;
            let axis = [ox * inv, oy * inv, oz * inv];
            let dq = quat_from_axis_angle(axis, angle);
            dem.quaternion[i] = quat_mul(dq, dem.quaternion[i]);
        }
    }
}

/// Final half-step: advance angular velocity using updated torques.
pub fn final_rotation(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("final_rotation");
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inv_inertia = dem.inv_inertia[i];
        if inv_inertia == 0.0 { continue; } // Skip clump sub-spheres

        dem.omega[i][0] += 0.5 * dt * dem.torque[i][0] * inv_inertia;
        dem.omega[i][1] += 0.5 * dt * dem.torque[i][1] * inv_inertia;
        dem.omega[i][2] += 0.5 * dt * dem.torque[i][2] * inv_inertia;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_test_utils::push_dem_test_atom;

    #[test]
    fn angular_acceleration_from_torque() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let dt = 1e-7;
        atom.dt = dt;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0; 3], radius);
        let mass = atom.mass[0];
        let inertia = 0.4 * mass * radius * radius;

        // Apply torque around z-axis
        dem.torque[0][2] = 1.0;
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_update_system(initial_rotation, ParticleSimScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let expected_omega_z = 0.5 * dt * 1.0 / inertia;
        assert!(
            (dem.omega[0][2] - expected_omega_z).abs() < 1e-20,
            "omega_z should be {}, got {}",
            expected_omega_z,
            dem.omega[0][2]
        );
    }

    #[test]
    fn quaternion_updates_from_angular_velocity() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-5;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0; 3], radius);
        dem.omega[0][2] = 100.0;
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_update_system(initial_rotation, ParticleSimScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let q = dem.quaternion[0];
        // Check quaternion is no longer identity [1,0,0,0]
        let dot = q[0]; // dot with identity = w component
        let angle = 2.0 * dot.clamp(-1.0, 1.0).acos();
        assert!(
            angle > 1e-10,
            "quaternion should have rotated, angle = {}",
            angle
        );
    }
}
