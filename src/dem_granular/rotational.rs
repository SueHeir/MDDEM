use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::UnitVector3;

use crate::{
    dem_atom::DemAtom,
    mddem_atom::{Atom, AtomDataRegistry},
};

pub struct RotationalDynamicsPlugin;

impl Plugin for RotationalDynamicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_rotation, ScheduleSet::InitalIntegration)
            .add_update_system(final_rotation, ScheduleSet::FinalIntegration);
    }
}

/// Half angular-velocity kick + full quaternion drift (first half of rotational Velocity Verlet).
///
/// Runs at `InitalIntegration` alongside the translational half-kick + position drift.
/// Only local atoms are integrated; ghost atoms do not need orientation tracking.
///
/// Moment of inertia for a solid sphere: I = 2/5 · m · r²
pub fn initial_rotation(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let dem = registry.get::<DemAtom>().unwrap();
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inertia = 0.4 * atoms.mass[i] * dem.radius[i] * dem.radius[i];

        // Half angular-velocity kick: ω += 0.5·dt·τ/I
        let torque = atoms.torque[i];
        atoms.omega[i] += 0.5 * dt * torque / inertia;

        // Full quaternion drift: rotate by ω·dt around the current rotation axis
        let angle = atoms.omega[i].norm() * dt;
        if angle > 1e-14 {
            let axis = UnitVector3::new_normalize(atoms.omega[i]);
            let dq = nalgebra::UnitQuaternion::from_axis_angle(&axis, angle);
            atoms.quaterion[i] = dq * atoms.quaterion[i];
        }
    }
}

/// Second half angular-velocity kick (second half of rotational Velocity Verlet).
///
/// Runs at `FinalIntegration` after force evaluation and ghost-force accumulation.
/// Torques have been updated with the new step's contact forces at this point.
pub fn final_rotation(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let dem = registry.get::<DemAtom>().unwrap();
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inertia = 0.4 * atoms.mass[i] * dem.radius[i] * dem.radius[i];

        // Half angular-velocity kick: ω += 0.5·dt·τ/I
        let torque = atoms.torque[i];
        atoms.omega[i] += 0.5 * dt * torque / inertia;
    }
}
