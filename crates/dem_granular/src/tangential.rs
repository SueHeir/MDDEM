use std::collections::{HashMap, HashSet};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry};
use mddem_neighbor::Neighbor;

// sqrt(5/3) — damping coefficient constant shared with Hertz normal model
const SQRT_5_3: f64 = 0.9128709291752768;

pub struct MindlinTangentialForcePlugin;

impl Plugin for MindlinTangentialForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(
            mindlin_tangential_force
                .label("mindlin_tangential")
                .after("hertz_normal"),
            ScheduleSet::Force,
        );
    }
}

pub fn mindlin_tangential_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    mut history: Local<HashMap<(u32, u32), Vector3<f64>>>,
) {
    let dem = registry.get::<DemAtom>().unwrap();
    let dt = atoms.dt;

    let mut active: HashSet<(u32, u32)> = HashSet::new();
    for &(i, j) in &neighbor.neighbor_list {
        let dx = atoms.pos_x[j] - atoms.pos_x[i];
        let dy = atoms.pos_y[j] - atoms.pos_y[i];
        let dz = atoms.pos_z[j] - atoms.pos_z[i];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < dem.radius[i] + dem.radius[j] {
            let ti = atoms.tag[i].min(atoms.tag[j]);
            let tj = atoms.tag[i].max(atoms.tag[j]);
            active.insert((ti, tj));
        }
    }
    history.retain(|k, _| active.contains(k));

    for &(i, j) in &neighbor.neighbor_list {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let diff = Vector3::new(
            atoms.pos_x[j] - atoms.pos_x[i],
            atoms.pos_y[j] - atoms.pos_y[i],
            atoms.pos_z[j] - atoms.pos_z[i],
        );
        let distance = diff.norm();

        if distance >= r1 + r2 || distance == 0.0 {
            continue;
        }

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;
        let mu = material_table.friction_ij[mat_i][mat_j];
        let beta = material_table.beta_ij[mat_i][mat_j];

        let n = diff / distance;
        let delta = (r1 + r2) - distance;

        let r_eff = (r1 * r2) / (r1 + r2);
        let e_eff = 1.0
            / ((1.0 - material_table.poisson_ratio[mat_i].powi(2))
                / material_table.youngs_mod[mat_i]
                + (1.0 - material_table.poisson_ratio[mat_j].powi(2))
                    / material_table.youngs_mod[mat_j]);
        let g_eff = 1.0
            / (2.0
                * (2.0 - material_table.poisson_ratio[mat_i])
                * (1.0 + material_table.poisson_ratio[mat_i])
                / material_table.youngs_mod[mat_i]
                + 2.0
                    * (2.0 - material_table.poisson_ratio[mat_j])
                    * (1.0 + material_table.poisson_ratio[mat_j])
                    / material_table.youngs_mod[mat_j]);

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;
        let k_t = 8.0 * g_eff * sqrt_dr;

        let m_r = (atoms.mass[i] * atoms.mass[j]) / (atoms.mass[i] + atoms.mass[j]);

        let vel_i = Vector3::new(atoms.vel_x[i], atoms.vel_y[i], atoms.vel_z[i]);
        let vel_j = Vector3::new(atoms.vel_x[j], atoms.vel_y[j], atoms.vel_z[j]);
        let omega_i = Vector3::new(atoms.omega_x[i], atoms.omega_y[i], atoms.omega_z[i]);
        let omega_j = Vector3::new(atoms.omega_x[j], atoms.omega_y[j], atoms.omega_z[j]);

        let v_contact_i = vel_i + omega_i.cross(&(r1 * n));
        let v_contact_j = vel_j + omega_j.cross(&(-r2 * n));
        let v_rel = v_contact_j - v_contact_i;
        let v_n_scalar = v_rel.dot(&n);
        let v_t = v_rel - v_n_scalar * n;

        let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n_scalar;
        let f_n_mag = (k_n * delta - f_diss_n).max(0.0);

        let tag_i = atoms.tag[i];
        let tag_j = atoms.tag[j];
        let canonical_key = (tag_i.min(tag_j), tag_i.max(tag_j));
        let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

        let stored = history.entry(canonical_key).or_insert_with(Vector3::zeros);
        let mut s = sign * *stored;

        s -= s.dot(&n) * n;

        s += v_t * dt;

        let f_t_spring_mag = k_t * s.norm();
        let f_t_max = mu * f_n_mag;
        if f_t_spring_mag > f_t_max && f_t_spring_mag > 1e-30 {
            s *= f_t_max / f_t_spring_mag;
        }

        let gamma_t = -2.0 * SQRT_5_3 * beta * (k_t * m_r).sqrt();
        let mut f_t = k_t * s - gamma_t * v_t;
        let f_t_mag = f_t.norm();
        if f_t_mag > f_t_max && f_t_mag > 1e-30 {
            f_t *= f_t_max / f_t_mag;
        }

        let torque_i = (r1 * n).cross(&f_t);
        let torque_j = (-r2 * n).cross(&(-f_t));

        let scale = if atoms.is_ghost[i] || atoms.is_ghost[j] {
            0.5
        } else {
            1.0
        };
        atoms.force_x[i] += f_t.x * scale;
        atoms.force_y[i] += f_t.y * scale;
        atoms.force_z[i] += f_t.z * scale;
        atoms.force_x[j] -= f_t.x * scale;
        atoms.force_y[j] -= f_t.y * scale;
        atoms.force_z[j] -= f_t.z * scale;
        atoms.torque_x[i] += torque_i.x * scale;
        atoms.torque_y[i] += torque_i.y * scale;
        atoms.torque_z[i] += torque_i.z * scale;
        atoms.torque_x[j] += torque_j.x * scale;
        atoms.torque_y[j] += torque_j.y * scale;
        atoms.torque_z[j] += torque_j.z * scale;

        *stored = sign * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::{DemAtom, MaterialTable};
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use nalgebra::{UnitQuaternion, Vector3};
    use std::f64::consts::PI;

    fn make_material_table() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
        mt.build_pair_tables();
        mt
    }

    fn push_test_atom(
        atom: &mut Atom,
        dem: &mut DemAtom,
        tag: u32,
        pos: Vector3<f64>,
        radius: f64,
    ) {
        let density = 2500.0;
        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(pos.x);
        atom.pos_y.push(pos.y);
        atom.pos_z.push(pos.z);
        atom.vel_x.push(0.0);
        atom.vel_y.push(0.0);
        atom.vel_z.push(0.0);
        atom.force_x.push(0.0);
        atom.force_y.push(0.0);
        atom.force_z.push(0.0);
        atom.torque_x.push(0.0);
        atom.torque_y.push(0.0);
        atom.torque_z.push(0.0);
        atom.mass.push(mass);
        atom.skin.push(radius);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0);
        atom.omega_y.push(0.0);
        atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0);
        atom.ang_mom_y.push(0.0);
        atom.ang_mom_z.push(0.0);
        dem.radius.push(radius);
        dem.density.push(density);
    }

    #[test]
    fn spring_history_accumulates_tangential_force() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            1,
            Vector3::new(0.0019, 0.0, 0.0),
            radius,
        );
        atom.vel_y[1] = 0.1;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_list.push((0, 1));

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force_y[0].abs() > 0.0,
            "atom 0 should have nonzero tangential force y"
        );
        assert!(
            atom.force_y[1].abs() > 0.0,
            "atom 1 should have nonzero tangential force y"
        );
        assert!(
            (atom.force_y[0] + atom.force_y[1]).abs() < 1e-10,
            "tangential forces should be equal and opposite"
        );
        let torque_mag_0 =
            (atom.torque_x[0].powi(2) + atom.torque_y[0].powi(2) + atom.torque_z[0].powi(2)).sqrt();
        assert!(torque_mag_0 > 0.0, "atom 0 should have nonzero torque");
    }

    #[test]
    fn coulomb_friction_caps_tangential_force() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            1,
            Vector3::new(0.0019, 0.0, 0.0),
            radius,
        );
        atom.vel_y[1] = 1000.0;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_list.push((0, 1));

        let mt = make_material_table();

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_t_mag = (atom.force_y[0].powi(2) + atom.force_z[0].powi(2)).sqrt();
        assert!(f_t_mag > 0.0, "tangential force should be nonzero");
        assert!(
            f_t_mag < 1e6,
            "tangential force should be capped to a reasonable value"
        );
    }

    #[test]
    fn no_tangential_force_for_separated_atoms() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            1,
            Vector3::new(0.003, 0.0, 0.0),
            radius,
        );
        atom.vel_y[1] = 0.1;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_list.push((0, 1));

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag_0 =
            (atom.force_x[0].powi(2) + atom.force_y[0].powi(2) + atom.force_z[0].powi(2)).sqrt();
        let f_mag_1 =
            (atom.force_x[1].powi(2) + atom.force_y[1].powi(2) + atom.force_z[1].powi(2)).sqrt();
        assert!(f_mag_0 < 1e-20, "no force for separated atoms");
        assert!(f_mag_1 < 1e-20, "no force for separated atoms");
        let t_mag_0 =
            (atom.torque_x[0].powi(2) + atom.torque_y[0].powi(2) + atom.torque_z[0].powi(2)).sqrt();
        assert!(t_mag_0 < 1e-20, "no torque for separated atoms");
    }
}
