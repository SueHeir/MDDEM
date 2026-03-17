//! Standalone Mindlin tangential contact force with spring history.
//!
//! This module provides the tangential friction force as a separate plugin,
//! for use when normal and tangential are registered independently. For the
//! recommended fused implementation, use [`crate::contact::HertzMindlinContactPlugin`].
//!
//! # Physics
//!
//! The Mindlin no-slip tangential force uses an incremental spring displacement:
//!
//! 1. Rotate previous spring into current tangent plane (remove normal component)
//! 2. Integrate: `s += v_t · dt`
//! 3. Cap spring at Coulomb limit: `|k_t s| ≤ μ |F_n|`
//! 4. Tangential force: `F_t = k_t s - γ_t v_t`, capped at `μ |F_n|`
//!
//! where `k_t = 8 G* √(R* δ)` is the tangential stiffness, `G*` is the effective
//! shear modulus, and `γ_t = 2 β √(5/3) √(k_t m_r)` is the tangential damping.
//!
//! Spring displacements are stored in **canonical form** (from the perspective of the
//! particle with the lower tag) so that both particles in a pair see a consistent history
//! regardless of which is `i` vs `j` in the neighbor list.

use std::any::Any;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry};
use mddem_neighbor::Neighbor;

use crate::{SQRT_5_3, TANGENTIAL_EPSILON};

// ── ContactHistoryStore ─────────────────────────────────────────────────────

/// Per-contact spring displacement history for tangential, rolling, and twisting models.
///
/// Each contact entry is stored in **canonical form** — from the perspective of the
/// particle with the lower tag — so the spring is frame-consistent regardless of
/// neighbor list ordering. A `sign` factor of `+1` or `-1` converts between the
/// canonical frame and the local `(i, j)` frame each timestep.
///
/// # Storage layout
///
/// Each contact stores 7 `f64` displacement values:
///
/// | Indices | Model              | Description                             |
/// |---------|--------------------|-----------------------------------------|
/// | `[0..3]`| Mindlin tangential | Tangential spring displacement vector   |
/// | `[3..6]`| SDS rolling        | Rolling spring displacement vector      |
/// | `[6]`  | SDS twisting       | Twisting spring displacement (scalar)   |
///
/// Rolling and twisting slots are zero when the constant-torque model is used.
pub struct ContactHistoryStore {
    /// Per-atom list of `(partner_tag, spring_displacement[7], active_flag)`.
    ///
    /// `active_flag` is reset to `false` before each pair loop and set to `true`
    /// when a contact is touched. Stale entries (broken contacts) are pruned after
    /// the loop completes.
    pub contacts: Vec<Vec<(u32, [f64; 7], bool)>>,
}

impl ContactHistoryStore {
    /// Create an empty contact history store with no pre-allocated atoms.
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
        let new_contacts: Vec<Vec<(u32, [f64; 7], bool)>> =
            perm.iter().map(|&p| self.contacts[p].clone()).collect();
        self.contacts[..n].clone_from_slice(&new_contacts);
    }

    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.contacts.len() {
            let list = &self.contacts[i];
            buf.push(list.len() as f64);
            for &(tag, ref s, _) in list {
                buf.push(tag as f64);
                buf.push(s[0]);
                buf.push(s[1]);
                buf.push(s[2]);
                buf.push(s[3]);
                buf.push(s[4]);
                buf.push(s[5]);
                buf.push(s[6]);
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
            let s = [
                buf[pos + 1], buf[pos + 2], buf[pos + 3],
                buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7],
            ];
            list.push((tag, s, false));
            pos += 8;
        }
        self.contacts.push(list);
        pos
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Standalone Mindlin tangential force plugin with Coulomb friction cap and torque.
///
/// Registers [`ContactHistoryStore`] in the [`AtomDataRegistry`] and adds the
/// [`mindlin_tangential_force`] system at [`ScheduleSet::Force`], ordered after
/// the Hertz normal force.
///
/// For the recommended fused normal+tangential implementation, use
/// [`crate::contact::HertzMindlinContactPlugin`] instead.
pub struct MindlinTangentialForcePlugin;

impl Plugin for MindlinTangentialForcePlugin {
    fn build(&self, app: &mut App) {
        register_atom_data!(app, ContactHistoryStore::new());
        app.add_update_system(
            mindlin_tangential_force
                .label("mindlin_tangential")
                .after(crate::normal::hertz_normal_force),
            ScheduleSet::Force,
        );
    }
}

// ── Force system ────────────────────────────────────────────────────────────

/// Compute Mindlin tangential contact forces for all neighbor pairs.
///
/// For each contacting pair, computes the tangential friction force using an
/// incremental spring-history model, applies the Coulomb friction cap, and
/// accumulates both forces and torques with Newton's third law symmetry.
///
/// Contact history is stored in canonical form and pruned for stale contacts
/// at the end of each step.
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

            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let distance = (dx*dx + dy*dy + dz*dz).sqrt();

            if distance >= r1 + r2 || distance == 0.0 {
                continue;
            }

            let mat_i = atoms.atom_type[i] as usize;
            let mat_j = atoms.atom_type[j] as usize;
            let mu = material_table.friction_ij[mat_i][mat_j];
            let beta = material_table.beta_ij[mat_i][mat_j];

            let inv_dist = 1.0 / distance;
            let nx = dx * inv_dist;
            let ny = dy * inv_dist;
            let nz = dz * inv_dist;
            let delta = (r1 + r2) - distance;

            let r_eff = (r1 * r2) / (r1 + r2);
            let e_eff = material_table.e_eff_ij[mat_i][mat_j];
            let g_eff = material_table.g_eff_ij[mat_i][mat_j];

            let sqrt_dr = (delta * r_eff).sqrt();
            let s_n = 2.0 * e_eff * sqrt_dr;
            let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;
            let k_t = 8.0 * g_eff * sqrt_dr;

            let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

            // v_contact_i = vel_i + omega_i × (r1 * n)
            let r1nx = r1 * nx; let r1ny = r1 * ny; let r1nz = r1 * nz;
            let oix = dem.omega[i][0]; let oiy = dem.omega[i][1]; let oiz = dem.omega[i][2];
            let vc_ix = atoms.vel[i][0] + (oiy*r1nz - oiz*r1ny);
            let vc_iy = atoms.vel[i][1] + (oiz*r1nx - oix*r1nz);
            let vc_iz = atoms.vel[i][2] + (oix*r1ny - oiy*r1nx);

            // v_contact_j = vel_j + omega_j × (-r2 * n)
            let r2nx = r2 * nx; let r2ny = r2 * ny; let r2nz = r2 * nz;
            let ojx = dem.omega[j][0]; let ojy = dem.omega[j][1]; let ojz = dem.omega[j][2];
            let vc_jx = atoms.vel[j][0] + (-ojy*r2nz + ojz*r2ny);
            let vc_jy = atoms.vel[j][1] + (-ojz*r2nx + ojx*r2nz);
            let vc_jz = atoms.vel[j][2] + (-ojx*r2ny + ojy*r2nx);

            let vr_x = vc_jx - vc_ix;
            let vr_y = vc_jy - vc_iy;
            let vr_z = vc_jz - vc_iz;
            let v_n_scalar = vr_x*nx + vr_y*ny + vr_z*nz;
            let vt_x = vr_x - v_n_scalar * nx;
            let vt_y = vr_y - v_n_scalar * ny;
            let vt_z = vr_z - v_n_scalar * nz;

            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n_scalar;
            let f_n_mag = (k_n * delta - f_diss_n).max(0.0);

            let tag_i = atoms.tag[i];
            let tag_j = atoms.tag[j];
            let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

            // Look up existing spring (single search, reused for write-back)
            let entry_idx = history.contacts[i]
                .iter()
                .position(|(t, _, _)| *t == tag_j);
            let stored = match entry_idx {
                Some(idx) => history.contacts[i][idx].1,
                None => [0.0; 7],
            };

            let mut sx = sign * stored[0];
            let mut sy = sign * stored[1];
            let mut sz = sign * stored[2];

            let s_dot_n = sx*nx + sy*ny + sz*nz;
            sx -= s_dot_n * nx; sy -= s_dot_n * ny; sz -= s_dot_n * nz;
            sx += vt_x * dt; sy += vt_y * dt; sz += vt_z * dt;

            let s_mag = (sx*sx + sy*sy + sz*sz).sqrt();
            let f_t_spring_mag = k_t * s_mag;
            let f_t_max = mu * f_n_mag;
            if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
                let scale = f_t_max / f_t_spring_mag;
                sx *= scale; sy *= scale; sz *= scale;
            }

            let gamma_t = 2.0 * SQRT_5_3 * beta * (k_t * m_r).sqrt();
            let mut ft_x = k_t * sx - gamma_t * vt_x;
            let mut ft_y = k_t * sy - gamma_t * vt_y;
            let mut ft_z = k_t * sz - gamma_t * vt_z;
            let f_t_mag = (ft_x*ft_x + ft_y*ft_y + ft_z*ft_z).sqrt();
            if f_t_mag > f_t_max && f_t_mag > TANGENTIAL_EPSILON {
                let scale = f_t_max / f_t_mag;
                ft_x *= scale; ft_y *= scale; ft_z *= scale;
            }

            // Torques: τ_i = (r1 * n) × f_t, τ_j = (-r2 * n) × (-f_t) = (r2 * n) × f_t
            let ti_x = r1ny * ft_z - r1nz * ft_y;
            let ti_y = r1nz * ft_x - r1nx * ft_z;
            let ti_z = r1nx * ft_y - r1ny * ft_x;
            let tj_x = r2ny * ft_z - r2nz * ft_y;
            let tj_y = r2nz * ft_x - r2nx * ft_z;
            let tj_z = r2nx * ft_y - r2ny * ft_x;

            atoms.force[i][0] += ft_x;
            atoms.force[i][1] += ft_y;
            atoms.force[i][2] += ft_z;
            atoms.force[j][0] -= ft_x;
            atoms.force[j][1] -= ft_y;
            atoms.force[j][2] -= ft_z;
            dem.torque[i][0] += ti_x;
            dem.torque[i][1] += ti_y;
            dem.torque[i][2] += ti_z;
            dem.torque[j][0] += tj_x;
            dem.torque[j][1] += tj_y;
            dem.torque[j][2] += tj_z;

            // Store updated spring back (canonical form) and mark active
            // Only tangential spring is used in standalone mode; rolling/twisting slots zeroed.
            let new_spring = [sign * sx, sign * sy, sign * sz, 0.0, 0.0, 0.0, 0.0];
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
    use dem_atom::DemAtom;
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use mddem_test_utils::{make_material_table, push_dem_test_atom};

    fn push_test_atom_with_history(
        atom: &mut Atom,
        dem: &mut DemAtom,
        history: &mut ContactHistoryStore,
        tag: u32,
        pos: [f64; 3],
        radius: f64,
    ) {
        push_dem_test_atom(atom, dem, tag, pos, radius);
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

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            [0.0019, 0.0, 0.0],
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

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            [0.0019, 0.0, 0.0],
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

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom,
            &mut dem,
            &mut hist,
            1,
            [0.003, 0.0, 0.0],
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
