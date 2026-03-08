use std::any::{Any, TypeId};
use std::collections::HashSet;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomData, AtomDataRegistry};
use mddem_neighbor::Neighbor;

// sqrt(5/3) — damping coefficient constant shared with Hertz normal model
const SQRT_5_3: f64 = 0.9128709291752768;

// ── ContactHistoryStore ─────────────────────────────────────────────────────

/// Stores per-atom tangential spring displacement history for Mindlin contacts.
pub struct ContactHistoryStore {
    /// Per-atom list of (partner_tag, spring_displacement).
    /// The spring displacement is stored in canonical form (lower tag's perspective).
    pub contacts: Vec<Vec<(u32, Vector3<f64>)>>,
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
        let new_contacts: Vec<Vec<(u32, Vector3<f64>)>> =
            perm.iter().map(|&p| self.contacts[p].clone()).collect();
        self.contacts[..n].clone_from_slice(&new_contacts);
    }

    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.contacts.len() {
            let list = &self.contacts[i];
            buf.push(list.len() as f64);
            for &(tag, ref s) in list {
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
            list.push((tag, s));
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
                .after("hertz_normal"),
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
    let dem = registry.expect::<DemAtom>("mindlin_tangential_force");
    let mut history = registry.expect_mut::<ContactHistoryStore>("mindlin_tangential_force");
    let dt = atoms.dt;

    // Ensure contacts vec covers all atoms (local + ghost).
    // New atoms from insertion or ghost unpacking may not have entries yet.
    while history.contacts.len() < atoms.len() {
        history.contacts.push(Vec::new());
    }

    let nlocal = atoms.nlocal as usize;

    // Track which partner tags are active per local atom for pruning
    let mut active_partners: Vec<HashSet<u32>> = (0..nlocal).map(|_| HashSet::new()).collect();

    for (i, j) in neighbor.pairs(nlocal) {
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
            let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

            // Look up existing spring in atom i's contact list by partner tag.
            // The spring is stored in canonical form (from lower tag's perspective).
            let stored_spring = history.contacts[i]
                .iter()
                .find(|(t, _)| *t == tag_j)
                .map(|(_, s)| *s)
                .unwrap_or_else(Vector3::zeros);

            let mut s = sign * stored_spring;

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

            atoms.force_x[i] += f_t.x;
            atoms.force_y[i] += f_t.y;
            atoms.force_z[i] += f_t.z;
            atoms.force_x[j] -= f_t.x;
            atoms.force_y[j] -= f_t.y;
            atoms.force_z[j] -= f_t.z;
            atoms.torque_x[i] += torque_i.x;
            atoms.torque_y[i] += torque_i.y;
            atoms.torque_z[i] += torque_i.z;
            atoms.torque_x[j] += torque_j.x;
            atoms.torque_y[j] += torque_j.y;
            atoms.torque_z[j] += torque_j.z;

            // Store updated spring back (canonical form)
            let new_spring = sign * s;
            if let Some(entry) = history.contacts[i].iter_mut().find(|(t, _)| *t == tag_j) {
                entry.1 = new_spring;
            } else {
                history.contacts[i].push((tag_j, new_spring));
            }
            active_partners[i].insert(tag_j);
    }

    // Prune stale contacts for local atoms
    for i in 0..nlocal {
        let active = &active_partners[i];
        history.contacts[i].retain(|(partner_tag, _)| active.contains(partner_tag));
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
        atom.vel_y[1] = 0.1;
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
        atom.vel_y[1] = 1000.0;
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
        atom.vel_y[1] = 0.1;
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
