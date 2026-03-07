use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::{UnitVector3, Vector3};

use mddem_core::{Atom, AtomDataRegistry};
use dem_atom::DemAtom;

pub struct RotationalDynamicsPlugin;

impl Plugin for RotationalDynamicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration)
            .add_update_system(final_rotation, ScheduleSet::FinalIntegration);
    }
}

pub fn initial_rotation(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let dem = registry.get::<DemAtom>().unwrap();
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inertia = 0.4 * atoms.mass[i] * dem.radius[i] * dem.radius[i];

        atoms.omega_x[i] += 0.5 * dt * atoms.torque_x[i] / inertia;
        atoms.omega_y[i] += 0.5 * dt * atoms.torque_y[i] / inertia;
        atoms.omega_z[i] += 0.5 * dt * atoms.torque_z[i] / inertia;

        let omega = Vector3::new(atoms.omega_x[i], atoms.omega_y[i], atoms.omega_z[i]);
        let angle = omega.norm() * dt;
        if angle > 1e-14 {
            let axis = UnitVector3::new_normalize(omega);
            let dq = nalgebra::UnitQuaternion::from_axis_angle(&axis, angle);
            atoms.quaterion[i] = dq * atoms.quaterion[i];
        }
    }
}

pub fn final_rotation(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let dem = registry.get::<DemAtom>().unwrap();
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        let inertia = 0.4 * atoms.mass[i] * dem.radius[i] * dem.radius[i];

        atoms.omega_x[i] += 0.5 * dt * atoms.torque_x[i] / inertia;
        atoms.omega_y[i] += 0.5 * dt * atoms.torque_y[i] / inertia;
        atoms.omega_z[i] += 0.5 * dt * atoms.torque_z[i] / inertia;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{Atom, AtomDataRegistry};
    use dem_atom::DemAtom;
    use nalgebra::UnitQuaternion;
    use std::f64::consts::PI;

    fn push_test_atom(atom: &mut Atom, dem: &mut DemAtom, tag: u32, radius: f64) {
        let density = 2500.0;
        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(0.0); atom.pos_y.push(0.0); atom.pos_z.push(0.0);
        atom.vel_x.push(0.0); atom.vel_y.push(0.0); atom.vel_z.push(0.0);
        atom.force_x.push(0.0); atom.force_y.push(0.0); atom.force_z.push(0.0);
        atom.torque_x.push(0.0); atom.torque_y.push(0.0); atom.torque_z.push(0.0);
        atom.mass.push(mass);
        atom.skin.push(radius);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0); atom.omega_y.push(0.0); atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0); atom.ang_mom_y.push(0.0); atom.ang_mom_z.push(0.0);
        dem.radius.push(radius);
        dem.density.push(density);
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
        atom.torque_z[0] = 1.0;
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let expected_omega_z = 0.5 * dt * 1.0 / inertia;
        assert!((atom.omega_z[0] - expected_omega_z).abs() < 1e-20,
            "omega_z should be {}, got {}", expected_omega_z, atom.omega_z[0]);
    }

    #[test]
    fn quaternion_updates_from_angular_velocity() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-5;

        push_test_atom(&mut atom, &mut dem, 0, radius);
        atom.omega_z[0] = 100.0;
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_update_system(initial_rotation, ScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let q = atom.quaterion[0];
        let identity = UnitQuaternion::identity();
        let angle = q.angle_to(&identity);
        assert!(angle > 1e-10, "quaternion should have rotated, angle = {}", angle);
    }
}
