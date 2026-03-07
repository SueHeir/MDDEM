use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use mddem_core::Atom;

pub struct VelocityVerletPlugin;

impl Plugin for VelocityVerletPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_integration, ScheduleSet::InitialIntegration)
            .add_update_system(final_integration, ScheduleSet::FinalIntegration);
    }
}

pub fn initial_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    for i in 0..atoms.len() {
        let half_dt_over_m = 0.5 * dt / atoms.mass[i];
        atoms.vel_x[i] += half_dt_over_m * atoms.force_x[i];
        atoms.vel_y[i] += half_dt_over_m * atoms.force_y[i];
        atoms.vel_z[i] += half_dt_over_m * atoms.force_z[i];
        atoms.pos_x[i] += atoms.vel_x[i] * dt;
        atoms.pos_y[i] += atoms.vel_y[i] * dt;
        atoms.pos_z[i] += atoms.vel_z[i] * dt;
    }
}

pub fn final_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    for i in 0..atoms.len() {
        let half_dt_over_m = 0.5 * dt / atoms.mass[i];
        atoms.vel_x[i] += half_dt_over_m * atoms.force_x[i];
        atoms.vel_y[i] += half_dt_over_m * atoms.force_y[i];
        atoms.vel_z[i] += half_dt_over_m * atoms.force_z[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    use nalgebra::UnitQuaternion;

    fn make_atom() -> Atom {
        let mut atom = Atom::new();
        atom.dt = 0.01;
        atom.pos_x.push(0.0);
        atom.pos_y.push(0.0);
        atom.pos_z.push(0.0);
        atom.vel_x.push(1.0);
        atom.vel_y.push(0.0);
        atom.vel_z.push(0.0);
        atom.force_x.push(2.0);
        atom.force_y.push(0.0);
        atom.force_z.push(0.0);
        atom.torque_x.push(0.0);
        atom.torque_y.push(0.0);
        atom.torque_z.push(0.0);
        atom.mass.push(1.0);
        atom.tag.push(0);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.skin.push(0.001);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0);
        atom.omega_y.push(0.0);
        atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0);
        atom.ang_mom_y.push(0.0);
        atom.ang_mom_z.push(0.0);
        atom.nlocal = 1;
        atom.natoms = 1;
        atom
    }

    #[test]
    fn initial_integration_updates_position_and_velocity() {
        let mut app = App::new();
        app.add_resource(make_atom());
        app.add_update_system(initial_integration, ScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // v += 0.5 * 0.01 * 2.0 / 1.0 = 0.01 → v = 1.01
        // x += 1.01 * 0.01 = 0.0101
        assert!((atom.vel_x[0] - 1.01).abs() < 1e-10);
        assert!((atom.pos_x[0] - 0.0101).abs() < 1e-10);
    }

    #[test]
    fn final_integration_updates_velocity_only() {
        let mut app = App::new();
        app.add_resource(make_atom());
        app.add_update_system(final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.vel_x[0] - 1.01).abs() < 1e-10);
        // Position should be unchanged
        assert!((atom.pos_x[0] - 0.0).abs() < 1e-10);
    }
}
