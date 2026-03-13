use std::any::{Any, TypeId};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomData, AtomDataRegistry};
use mddem_neighbor::Neighbor;

use crate::{SQRT_5_3, TANGENTIAL_EPSILON};

// ── ContactHistoryStore ─────────────────────────────────────────────────────

/// Stores per-atom tangential spring displacement history for Mindlin contacts.
pub struct ContactHistoryStore {
    /// Per-atom list of (partner_tag, spring_displacement, active_flag).
    /// The spring displacement is stored in canonical form (lower tag's perspective).
    /// The bool flag is transient — set to false at start of step, true when contact is touched.
    pub contacts: Vec<Vec<(u32, Vector3<f64>, bool)>>,
}

impl ContactHistoryStore {
    pub fn new() -> Self {
        ContactHistoryStore {
            contacts: Vec::new(),
        }
    }
}

impl AtomData for ContactHistoryStore {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn truncate(&mut self, n: usize) {
        // Grow if needed (atoms may have been inserted without going through unpack)
        self.contacts.resize_with(n, Vec::new);
        self.contacts.truncate(n);
    }

    fn swap_remove(&mut self, i: usize) {
        if i < self.contacts.len() {
            self.contacts.swap_remove(i);
        }
    }

    fn apply_permutation(&mut self, perm: &[usize], n: usize) {
        let new_contacts: Vec<Vec<(u32, Vector3<f64>, bool)>> =
            perm.iter().map(|&p| self.contacts[p].clone()).collect();
        self.contacts[..n].clone_from_slice(&new_contacts);
    }

    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.contacts.len() {
            let list = &self.contacts[i];
            buf.push(list.len() as f64);
            for &(tag, ref s, _) in list {
                buf.push(tag as f64);
                buf.push(s.x);
                buf.push(s.y);
                buf.push(s.z);
            }
        } else {
            buf.push(0.0); // no contacts
        }
    }

    fn unpack(&mut self, buf: &[f64]) -> usize {
        let count = buf[0] as usize;
        let mut list = Vec::with_capacity(count);
        let mut pos = 1;
        for _ in 0..count {
            let tag = buf[pos] as u32;
            let s = Vector3::new(buf[pos + 1], buf[pos + 2], buf[pos + 3]);
            list.push((tag, s, false));
            pos += 4;
        }
        self.contacts.push(list);
        pos
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Mindlin spring-history tangential force with Coulomb friction cap and torque accumulation.
pub struct MindlinTangentialForcePlugin;

impl Plugin for MindlinTangentialForcePlugin {
    fn build(&self, app: &mut App) {
        if let Some(registry_option) = app.get_mut_resource(TypeId::of::<AtomDataRegistry>()) {
            let mut registry_binder = registry_option.borrow_mut();
            let registry = registry_binder.downcast_mut::<AtomDataRegistry>().unwrap();
            registry.register(ContactHistoryStore::new());
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }
        app.add_update_system(
            mindlin_tangential_force
                .label("mindlin_tangential")
                .after(crate::normal::hertz_normal_force),
            ScheduleSet::Force,
        );
    }
}

// ── Force system ────────────────────────────────────────────────────────────

pub fn mindlin_tangential_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let mut dem = registry.expect_mut::<DemAtom>("mindlin_tangential_force");
    let mut history = registry.expect_mut::<ContactHistoryStore>("mindlin_tangential_force");
    let dt = atoms.dt;

    // Ensure contacts vec covers all atoms (local + ghost).
    // New atoms from insertion or ghost unpacking may not have entries yet.
    while history.contacts.len() < atoms.len() {
        history.contacts.push(Vec::new());
    }

    let nlocal = atoms.nlocal as usize;

    // Reset all active flags before pair loop
    for i in 0..nlocal {
        for entry in &mut history.contacts[i] {
            entry.2 = false;
        }
    }

    for (i, j) in neighbor.pairs(nlocal) {
            let r1 = dem.radius[i];
            let r2 = dem.radius[j];

            let diff = Vector3::new(
                atoms.pos[j][0] - atoms.pos[i][0],
                atoms.pos[j][1] - atoms.pos[i][1],
                atoms.pos[j][2] - atoms.pos[i][2],
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
            let e_eff = material_table.e_eff_ij[mat_i][mat_j];
            let g_eff = material_table.g_eff_ij[mat_i][mat_j];

            let sqrt_dr = (delta * r_eff).sqrt();
            let s_n = 2.0 * e_eff * sqrt_dr;
            let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;
            let k_t = 8.0 * g_eff * sqrt_dr;

            let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

            let vel_i = Vector3::new(atoms.vel[i][0], atoms.vel[i][1], atoms.vel[i][2]);
            let vel_j = Vector3::new(atoms.vel[j][0], atoms.vel[j][1], atoms.vel[j][2]);
            let omega_i = Vector3::new(dem.omega[i][0], dem.omega[i][1], dem.omega[i][2]);
            let omega_j = Vector3::new(dem.omega[j][0], dem.omega[j][1], dem.omega[j][2]);

            let v_contact_i = vel_i + omega_i.cross(&(r1 * n));
            let v_contact_j = vel_j + omega_j.cross(&(-r2 * n));
            let v_rel = v_contact_j - v_contact_i;
            let v_n_scalar = v_rel.dot(&n);
            let v_t = v_rel - v_n_scalar * n;

            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n_scalar;
            let f_n_mag = (k_n * delta - f_diss_n).max(0.0);

            let tag_i = atoms.tag[i];
            let tag_j = atoms.tag[j];
            let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

            // Look up existing spring (single search, reused for write-back)
            let entry_idx = history.contacts[i]
                .iter()
                .position(|(t, _, _)| *t == tag_j);
            let stored_spring = match entry_idx {
                Some(idx) => history.contacts[i][idx].1,
                None => Vector3::zeros(),
            };

            let mut s = sign * stored_spring;

            s -= s.dot(&n) * n;
            s += v_t * dt;

            let f_t_spring_mag = k_t * s.norm();
            let f_t_max = mu * f_n_mag;
            if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
                s *= f_t_max / f_t_spring_mag;
            }

            let gamma_t = 2.0 * SQRT_5_3 * beta * (k_t * m_r).sqrt();
            let mut f_t = k_t * s - gamma_t * v_t;
            let f_t_mag = f_t.norm();
            if f_t_mag > f_t_max && f_t_mag > TANGENTIAL_EPSILON {
                f_t *= f_t_max / f_t_mag;
            }

            let torque_i = (r1 * n).cross(&f_t);
            let torque_j = (-r2 * n).cross(&(-f_t));

            atoms.force[i][0] += f_t.x;
            atoms.force[i][1] += f_t.y;
            atoms.force[i][2] += f_t.z;
            atoms.force[j][0] -= f_t.x;
            atoms.force[j][1] -= f_t.y;
            atoms.force[j][2] -= f_t.z;
            dem.torque[i][0] += torque_i.x;
            dem.torque[i][1] += torque_i.y;
            dem.torque[i][2] += torque_i.z;
            dem.torque[j][0] += torque_j.x;
            dem.torque[j][1] += torque_j.y;
            dem.torque[j][2] += torque_j.z;

            // Store updated spring back (canonical form) and mark active
            let new_spring = sign * s;
            match entry_idx {
                Some(idx) => {
                    history.contacts[i][idx].1 = new_spring;
                    history.contacts[i][idx].2 = true;
                }
                None => history.contacts[i].push((tag_j, new_spring, true)),
            }
    }

    // Prune stale contacts for local atoms (remove entries not touched this step)
    for i in 0..nlocal {
        history.contacts[i].retain(|(_, _, active)| *active);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::{DemAtom, MaterialTable};
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use nalgebra::Vector3;
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
        history: &mut ContactHistoryStore,
        tag: u32,
        pos: Vector3<f64>,
        radius: f64,
    ) {
        let density = 2500.0;
        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
        atom.push_test_atom(tag, pos, radius, mass);
        dem.radius.push(radius);
        dem.density.push(density);
        dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
        dem.quaternion.push(nalgebra::UnitQuaternion::identity());
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
        history.contacts.push(Vec::new());
    }

    #[test]
    fn spring_history_accumulates_tangential_force() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, &mut hist, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            Vector3::new(0.0019, 0.0, 0.0),
            radius,
        );
        atom.vel[1][1] = 0.1;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][1].abs() > 0.0,
            "atom 0 should have nonzero tangential force y"
        );
        assert!(
            atom.force[1][1].abs() > 0.0,
            "atom 1 should have nonzero tangential force y"
        );
        assert!(
            (atom.force[0][1] + atom.force[1][1]).abs() < 1e-10,
            "tangential forces should be equal and opposite"
        );
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let torque_mag_0 =
            (dem.torque[0][0].powi(2) + dem.torque[0][1].powi(2) + dem.torque[0][2].powi(2)).sqrt();
        assert!(torque_mag_0 > 0.0, "atom 0 should have nonzero torque");
    }

    #[test]
    fn coulomb_friction_caps_tangential_force() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, &mut hist, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            Vector3::new(0.0019, 0.0, 0.0),
            radius,
        );
        atom.vel[1][1] = 1000.0;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mt = make_material_table();

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_t_mag = (atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
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
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(&mut atom, &mut dem, &mut hist, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            Vector3::new(0.003, 0.0, 0.0),
            radius,
        );
        atom.vel[1][1] = 0.1;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(mindlin_tangential_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag_0 =
            (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        let f_mag_1 =
            (atom.force[1][0].powi(2) + atom.force[1][1].powi(2) + atom.force[1][2].powi(2)).sqrt();
        assert!(f_mag_0 < 1e-20, "no force for separated atoms");
        assert!(f_mag_1 < 1e-20, "no force for separated atoms");
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let t_mag_0 =
            (dem.torque[0][0].powi(2) + dem.torque[0][1].powi(2) + dem.torque[0][2].powi(2)).sqrt();
        assert!(t_mag_0 < 1e-20, "no torque for separated atoms");
    }
}
