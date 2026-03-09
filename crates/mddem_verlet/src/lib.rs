//! Translational Velocity Verlet time integration (half-step kick-drift-kick).

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use mddem_core::Atom;

/// Registers initial and final integration systems for translational Velocity Verlet.
pub struct VelocityVerletPlugin;

impl Plugin for VelocityVerletPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(initial_integration, ScheduleSet::InitialIntegration)
            .add_update_system(final_integration, ScheduleSet::FinalIntegration);
    }
}

pub fn initial_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;
    // SAFETY: i < nlocal <= len for all arrays (inv_mass, force, vel, pos).
    // Use raw pointers to avoid borrow checker conflicts between different fields.
    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    let pos_ptr = atoms.pos.as_mut_ptr();
    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            let p = &mut *pos_ptr.add(i);
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        }
    }
}

pub fn final_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;
    // SAFETY: i < nlocal <= len for all arrays (inv_mass, force, vel).
    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    use nalgebra::Vector3;

    fn make_atom() -> Atom {
        let mut atom = Atom::new();
        atom.dt = 0.01;
        atom.push_test_atom(0, Vector3::zeros(), 0.001, 1.0);
        atom.vel[0][0] = 1.0;
        atom.force[0][0] = 2.0;
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
        assert!((atom.vel[0][0] - 1.01).abs() < 1e-10);
        assert!((atom.pos[0][0] - 0.0101).abs() < 1e-10);
    }

    #[test]
    fn final_integration_updates_velocity_only() {
        let mut app = App::new();
        app.add_resource(make_atom());
        app.add_update_system(final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.vel[0][0] - 1.01).abs() < 1e-10);
        // Position should be unchanged
        assert!((atom.pos[0][0] - 0.0).abs() < 1e-10);
    }
}
