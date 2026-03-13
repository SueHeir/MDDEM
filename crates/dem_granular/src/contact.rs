//! Fused Hertz-Mindlin contact force: computes normal and tangential forces in a single
//! pair loop, eliminating redundant shared computation (distance, material lookups, Hertz
//! stiffness, normal force magnitude).

use std::any::TypeId;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry, BondStore, VirialStress, VirialStressPlugin};
use mddem_neighbor::Neighbor;

use crate::tangential::ContactHistoryStore;
use crate::{LARGE_OVERLAP_WARN_THRESHOLD, MAX_OVERLAP_WARNINGS, SQRT_5_3, TANGENTIAL_EPSILON};

/// Fused Hertz normal + Mindlin tangential contact force plugin.
///
/// Registers [`ContactHistoryStore`] in the [`AtomDataRegistry`] and a single
/// `hertz_mindlin_contact` system at [`ScheduleSet::Force`].
pub struct HertzMindlinContactPlugin;

impl Plugin for HertzMindlinContactPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(VirialStressPlugin);
        // Register ContactHistoryStore if not already registered
        if let Some(registry_option) = app.get_mut_resource(TypeId::of::<AtomDataRegistry>()) {
            let mut registry_binder = registry_option.borrow_mut();
            let registry = registry_binder.downcast_mut::<AtomDataRegistry>().unwrap();
            registry.register(ContactHistoryStore::new());
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }
        app.add_update_system(
            hertz_mindlin_contact_force.label("hertz_mindlin_contact"),
            ScheduleSet::Force,
        );
    }
}

pub fn hertz_mindlin_contact_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let mut dem = registry.expect_mut::<DemAtom>("hertz_mindlin_contact_force");
    let mut history =
        registry.expect_mut::<ContactHistoryStore>("hertz_mindlin_contact_force");
    let bond_store = registry.get::<BondStore>();
    let dt = atoms.dt;

    // Ensure contacts vec covers all atoms (local + ghost)
    while history.contacts.len() < atoms.len() {
        history.contacts.push(Vec::new());
    }

    let nlocal = atoms.nlocal as usize;
    let mut overlap_warnings = 0usize;

    // Reset all active flags before pair loop
    for i in 0..nlocal {
        for entry in &mut history.contacts[i] {
            entry.2 = false;
        }
    }

    for (i, j) in neighbor.pairs(nlocal) {
        if let Some(ref bonds) = bond_store {
            if bonds.are_excluded(i, j, &atoms.tag) {
                continue;
            }
        }

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = r1 + r2;

        if dist_sq >= sum_r * sum_r {
            continue;
        }

        let distance = dist_sq.sqrt();

        if distance == 0.0 {
            #[cfg(debug_assertions)]
            eprintln!(
                "WARNING: zero separation between tags {} {}",
                atoms.tag[i], atoms.tag[j]
            );
            continue;
        }

        if distance / sum_r < LARGE_OVERLAP_WARN_THRESHOLD {
            overlap_warnings += 1;
            #[cfg(debug_assertions)]
            eprintln!(
                "WARNING: large overlap tags {} {} ratio {:.3}",
                atoms.tag[i],
                atoms.tag[j],
                distance / sum_r
            );
            if overlap_warnings > MAX_OVERLAP_WARNINGS {
                panic!(
                    "Over {} excessive overlaps this step — aborting. \
                     Check timestep or initial configuration.",
                    MAX_OVERLAP_WARNINGS
                );
            }
            continue;
        }

        atoms.is_collision[i] = true;
        atoms.is_collision[j] = true;

        // ── Shared quantities (computed once) ────────────────────────────
        let inv_dist = 1.0 / distance;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let nz = dz * inv_dist;
        let delta = sum_r - distance;

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;

        let r_eff = (r1 * r2) / sum_r;
        let e_eff = material_table.e_eff_ij[mat_i][mat_j];
        let g_eff = material_table.g_eff_ij[mat_i][mat_j];

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;
        let k_t = 8.0 * g_eff * sqrt_dr;

        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

        let beta = material_table.beta_ij[mat_i][mat_j];
        let mu = material_table.friction_ij[mat_i][mat_j];

        // Full relative velocity (including angular contributions)
        let omega_ix = dem.omega[i][0];
        let omega_iy = dem.omega[i][1];
        let omega_iz = dem.omega[i][2];
        let omega_jx = dem.omega[j][0];
        let omega_jy = dem.omega[j][1];
        let omega_jz = dem.omega[j][2];

        // v_contact_i = vel_i + omega_i × (r1 * n)
        let r1n_x = r1 * nx;
        let r1n_y = r1 * ny;
        let r1n_z = r1 * nz;
        let vc_ix = atoms.vel[i][0] + (omega_iy * r1n_z - omega_iz * r1n_y);
        let vc_iy = atoms.vel[i][1] + (omega_iz * r1n_x - omega_ix * r1n_z);
        let vc_iz = atoms.vel[i][2] + (omega_ix * r1n_y - omega_iy * r1n_x);

        // v_contact_j = vel_j + omega_j × (-r2 * n)
        let r2n_x = r2 * nx;
        let r2n_y = r2 * ny;
        let r2n_z = r2 * nz;
        let vc_jx = atoms.vel[j][0] + (-omega_jy * r2n_z + omega_jz * r2n_y);
        let vc_jy = atoms.vel[j][1] + (-omega_jz * r2n_x + omega_jx * r2n_z);
        let vc_jz = atoms.vel[j][2] + (-omega_jx * r2n_y + omega_jy * r2n_x);

        let vr_x = vc_jx - vc_ix;
        let vr_y = vc_jy - vc_iy;
        let vr_z = vc_jz - vc_iz;

        let v_n = vr_x * nx + vr_y * ny + vr_z * nz;

        // ── Normal force ─────────────────────────────────────────────────
        let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
        let f_n_mag = (k_n * delta - f_diss_n).max(0.0);

        let fn_x = f_n_mag * nx;
        let fn_y = f_n_mag * ny;
        let fn_z = f_n_mag * nz;

        atoms.force[i][0] -= fn_x;
        atoms.force[i][1] -= fn_y;
        atoms.force[i][2] -= fn_z;
        atoms.force[j][0] += fn_x;
        atoms.force[j][1] += fn_y;
        atoms.force[j][2] += fn_z;

        // ── Tangential force ─────────────────────────────────────────────
        let vt_x = vr_x - v_n * nx;
        let vt_y = vr_y - v_n * ny;
        let vt_z = vr_z - v_n * nz;

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

        let n_vec = Vector3::new(nx, ny, nz);
        let mut s = sign * stored_spring;
        s -= s.dot(&n_vec) * n_vec;
        s.x += vt_x * dt;
        s.y += vt_y * dt;
        s.z += vt_z * dt;

        // Coulomb cap on spring
        let f_t_spring_mag = k_t * s.norm();
        let f_t_max = mu * f_n_mag;
        if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
            s *= f_t_max / f_t_spring_mag;
        }

        // Tangential force with damping (gamma_t > 0, opposes sliding velocity)
        let gamma_t = 2.0 * SQRT_5_3 * beta * (k_t * m_r).sqrt();
        let mut ft_x = k_t * s.x - gamma_t * vt_x;
        let mut ft_y = k_t * s.y - gamma_t * vt_y;
        let mut ft_z = k_t * s.z - gamma_t * vt_z;

        // Coulomb cap on total tangential force
        let f_t_mag = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z).sqrt();
        if f_t_mag > f_t_max && f_t_mag > TANGENTIAL_EPSILON {
            let scale = f_t_max / f_t_mag;
            ft_x *= scale;
            ft_y *= scale;
            ft_z *= scale;
        }

        // Torques: τ_i = (r1 * n) × f_t, τ_j = (-r2 * n) × (-f_t) = (r2 * n) × f_t
        let ti_x = r1n_y * ft_z - r1n_z * ft_y;
        let ti_y = r1n_z * ft_x - r1n_x * ft_z;
        let ti_z = r1n_x * ft_y - r1n_y * ft_x;
        let tj_x = r2n_y * ft_z - r2n_z * ft_y;
        let tj_y = r2n_z * ft_x - r2n_x * ft_z;
        let tj_z = r2n_x * ft_y - r2n_y * ft_x;

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

        // Virial: force on i from j = (-fn + ft)
        if let Some(ref mut v) = virial {
            if v.active {
                v.add_pair(dx, dy, dz, -fn_x + ft_x, -fn_y + ft_y, -fn_z + ft_z);
            }
        }

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

    // Debug: check total force + torque on all atoms (local + ghost).
    // In a correct Newton's 3rd law implementation, the sum of all forces
    // from pair interactions must be zero (each pair contributes +F to one atom
    // and -F to the other). A nonzero sum means a pair was counted asymmetrically.
    #[cfg(debug_assertions)]
    {
        let total = atoms.len();
        let mut sum_fx = 0.0;
        let mut sum_fy = 0.0;
        let mut sum_fz = 0.0;
        for i in 0..total {
            sum_fx += atoms.force[i][0];
            sum_fy += atoms.force[i][1];
            sum_fz += atoms.force[i][2];
        }
        let sum_f = (sum_fx * sum_fx + sum_fy * sum_fy + sum_fz * sum_fz).sqrt();
        if sum_f > 1e-6 {
            eprintln!(
                "WARNING: nonzero net force after contact: |F|={:.6e} ({:.6e},{:.6e},{:.6e})",
                sum_f, sum_fx, sum_fy, sum_fz
            );
        }
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

    fn make_material_table() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn fused_contact_repulsive_for_overlap() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(
            &mut atom, &mut dem, &mut hist, 0,
            Vector3::new(0.0, 0.0, 0.0), radius,
        );
        push_test_atom(
            &mut atom, &mut dem, &mut hist, 1,
            Vector3::new(0.0019, 0.0, 0.0), radius,
        );
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
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0] < 0.0, "particle 0 should have negative x force");
        assert!(atom.force[1][0] > 0.0, "particle 1 should have positive x force");
        assert!((atom.force[0][0] + atom.force[1][0]).abs() < 1e-10);
    }

    #[test]
    fn fused_contact_tangential_with_sliding() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(
            &mut atom, &mut dem, &mut hist, 0,
            Vector3::new(0.0, 0.0, 0.0), radius,
        );
        push_test_atom(
            &mut atom, &mut dem, &mut hist, 1,
            Vector3::new(0.0019, 0.0, 0.0), radius,
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
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Normal force present
        assert!(atom.force[0][0] < 0.0, "normal force on atom 0");
        assert!(atom.force[1][0] > 0.0, "normal force on atom 1");
        // Tangential force present
        assert!(atom.force[0][1].abs() > 0.0, "tangential force on atom 0");
        assert!(
            (atom.force[0][1] + atom.force[1][1]).abs() < 1e-10,
            "tangential forces equal and opposite"
        );
        // Torque present (stored in DemAtom via registry)
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let t_mag = (dem.torque[0][0].powi(2) + dem.torque[0][1].powi(2) + dem.torque[0][2].powi(2)).sqrt();
        assert!(t_mag > 0.0, "torque on atom 0");
    }

    #[test]
    fn fused_contact_no_force_for_gap() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom(
            &mut atom, &mut dem, &mut hist, 0,
            Vector3::new(0.0, 0.0, 0.0), radius,
        );
        push_test_atom(
            &mut atom, &mut dem, &mut hist, 1,
            Vector3::new(0.003, 0.0, 0.0), radius,
        );
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
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0].abs() < 1e-20);
        assert!(atom.force[1][0].abs() < 1e-20);
    }
}
