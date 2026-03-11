use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::{UnitVector3, Vector3};

use dem_atom::DemAtom;
use mddem_core::{Atom, AtomDataRegistry};

/// Quaternion-based Velocity Verlet for angular degrees of freedom (I = 2/5 mr²).
pub struct RotationalDynamicsPlugin;

impl Plugin for RotationalDynamicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration)
            .add_update_system(final_rotation, ScheduleSet::FinalIntegration);
    }
}

pub fn initial_rotation(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("initial_rotation");
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inv_inertia = dem.inv_inertia[i];

        dem.omega[i][0] += 0.5 * dt * dem.torque[i][0] * inv_inertia;
        dem.omega[i][1] += 0.5 * dt * dem.torque[i][1] * inv_inertia;
        dem.omega[i][2] += 0.5 * dt * dem.torque[i][2] * inv_inertia;

        let omega = Vector3::new(dem.omega[i][0], dem.omega[i][1], dem.omega[i][2]);
        let angle = omega.norm() * dt;
        if angle > 1e-14 {
            let axis = UnitVector3::new_normalize(omega);
            let dq = nalgebra::UnitQuaternion::from_axis_angle(&axis, angle);
            dem.quaternion[i] = dq * dem.quaternion[i];
        }
    }
}

pub fn final_rotation(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("final_rotation");
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inv_inertia = dem.inv_inertia[i];

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
    use nalgebra::{UnitQuaternion, Vector3};
    use std::f64::consts::PI;

    fn push_test_atom(atom: &mut Atom, dem: &mut DemAtom, tag: u32, radius: f64) {
        let density = 2500.0;
        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
        atom.push_test_atom(tag, Vector3::zeros(), radius, mass);
        dem.radius.push(radius);
        dem.density.push(density);
        dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
        dem.quaternion.push(UnitQuaternion::identity());
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
    }

    #[test]
    fn angular_acceleration_from_torque() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let dt = 1e-7;
        atom.dt = dt;

        push_test_atom(&mut atom, &mut dem, 0, radius);
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
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration);
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

        push_test_atom(&mut atom, &mut dem, 0, radius);
        dem.omega[0][2] = 100.0;
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let q = dem.quaternion[0];
        let identity = UnitQuaternion::identity();
        let angle = q.angle_to(&identity);
        assert!(
            angle > 1e-10,
            "quaternion should have rotated, angle = {}",
            angle
        );
    }
}
