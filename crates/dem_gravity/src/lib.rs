use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, Config};

#[derive(Deserialize, Clone)]
pub struct GravityConfig {
    #[serde(default)]
    pub gx: f64,
    #[serde(default)]
    pub gy: f64,
    #[serde(default = "default_gz")]
    pub gz: f64,
}

impl Default for GravityConfig {
    fn default() -> Self {
        GravityConfig { gx: 0.0, gy: 0.0, gz: -9.81 }
    }
}

fn default_gz() -> f64 {
    -9.81
}

pub struct GravityPlugin;

impl Plugin for GravityPlugin {
    fn build(&self, app: &mut App) {
        Config::load::<GravityConfig>(app, "gravity");
        app.add_update_system(apply_gravity, ScheduleSet::Force);
    }
}

pub fn apply_gravity(mut atoms: ResMut<Atom>, gravity: Res<GravityConfig>) {
    for i in 0..atoms.nlocal as usize {
        atoms.force_x[i] += atoms.mass[i] * gravity.gx;
        atoms.force_y[i] += atoms.mass[i] * gravity.gy;
        atoms.force_z[i] += atoms.mass[i] * gravity.gz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    use nalgebra::UnitQuaternion;

    fn make_atom(mass: f64) -> Atom {
        let mut atom = Atom::new();
        atom.dt = 1e-6;
        atom.pos_x.push(0.0); atom.pos_y.push(0.0); atom.pos_z.push(0.0);
        atom.vel_x.push(0.0); atom.vel_y.push(0.0); atom.vel_z.push(0.0);
        atom.force_x.push(0.0); atom.force_y.push(0.0); atom.force_z.push(0.0);
        atom.torque_x.push(0.0); atom.torque_y.push(0.0); atom.torque_z.push(0.0);
        atom.mass.push(mass);
        atom.tag.push(0);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.skin.push(0.001);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0); atom.omega_y.push(0.0); atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0); atom.ang_mom_y.push(0.0); atom.ang_mom_z.push(0.0);
        atom.nlocal = 1;
        atom.natoms = 1;
        atom
    }

    #[test]
    fn gravity_applies_force_equal_to_mg() {
        let mass = 0.5;
        let gz = -9.81;

        let mut app = App::new();
        app.add_resource(make_atom(mass));
        app.add_resource(GravityConfig { gx: 0.0, gy: 0.0, gz });
        app.add_update_system(apply_gravity, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force_x[0]).abs() < 1e-15);
        assert!((atom.force_y[0]).abs() < 1e-15);
        assert!((atom.force_z[0] - mass * gz).abs() < 1e-15);
    }

    #[test]
    fn gravity_skips_ghost_atoms() {
        let mass = 1.0;
        let gz = -9.81;

        let mut atom = make_atom(mass);
        // Add a ghost atom
        atom.pos_x.push(0.0); atom.pos_y.push(0.0); atom.pos_z.push(0.0);
        atom.vel_x.push(0.0); atom.vel_y.push(0.0); atom.vel_z.push(0.0);
        atom.force_x.push(0.0); atom.force_y.push(0.0); atom.force_z.push(0.0);
        atom.torque_x.push(0.0); atom.torque_y.push(0.0); atom.torque_z.push(0.0);
        atom.mass.push(mass);
        atom.tag.push(1);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.is_ghost.push(true);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.skin.push(0.001);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0); atom.omega_y.push(0.0); atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0); atom.ang_mom_y.push(0.0); atom.ang_mom_z.push(0.0);
        // nlocal stays 1, ghost is index 1

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(GravityConfig { gx: 0.0, gy: 0.0, gz });
        app.add_update_system(apply_gravity, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Local atom gets force
        assert!((atom.force_z[0] - mass * gz).abs() < 1e-15);
        // Ghost atom does not
        assert!((atom.force_z[1]).abs() < 1e-15);
    }
}
