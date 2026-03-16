//! Fused Hertz-Mindlin contact force: computes normal and tangential forces in a single
//! pair loop, eliminating redundant shared computation (distance, material lookups, Hertz
//! stiffness, normal force magnitude).


use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{register_atom_data, Atom, AtomDataRegistry, BondStore, VirialStress, VirialStressPlugin};
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
        // Register ContactHistoryStore
        register_atom_data!(app, ContactHistoryStore::new());
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

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;
        let surface_energy = material_table.surface_energy_ij[mat_i][mat_j];

        // JKR: compute pull-off distance for extended interaction range
        let r_eff = (r1 * r2) / sum_r;
        let e_eff = material_table.e_eff_ij[mat_i][mat_j];
        let delta_pulloff = if surface_energy > 0.0 {
            let gamma = surface_energy;
            (std::f64::consts::PI * std::f64::consts::PI * gamma * gamma * r_eff
                / (4.0 * e_eff * e_eff))
                .cbrt()
        } else {
            0.0
        };

        // Check contact: geometric touch or within JKR adhesion range
        let interaction_r = sum_r + delta_pulloff;
        if dist_sq >= interaction_r * interaction_r {
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

        // delta > 0 means geometric overlap, delta < 0 means gap
        let delta = sum_r - distance;

        if delta > 0.0 && distance / sum_r < LARGE_OVERLAP_WARN_THRESHOLD {
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

        // For non-JKR, skip if no geometric overlap
        if delta <= 0.0 && surface_energy <= 0.0 {
            continue;
        }

        // ── Shared quantities (computed once) ────────────────────────────
        let inv_dist = 1.0 / distance;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let nz = dz * inv_dist;

        let g_eff = material_table.g_eff_ij[mat_i][mat_j];

        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

        let beta = material_table.beta_ij[mat_i][mat_j];
        let mu = material_table.friction_ij[mat_i][mat_j];
        let mu_r = material_table.rolling_friction_ij[mat_i][mat_j];
        let cohesion_energy = material_table.cohesion_energy_ij[mat_i][mat_j];

        // JKR adhesion-only regime: gap exists but within pull-off distance
        let jkr_adhesion_only = surface_energy > 0.0 && delta <= 0.0;

        // Hertz stiffness (only meaningful when delta > 0)
        let (s_n, k_n, k_t) = if delta > 0.0 {
            let sdr = (delta * r_eff).sqrt();
            let sn = 2.0 * e_eff * sdr;
            let kn = 4.0 / 3.0 * e_eff * sdr;
            let kt = 8.0 * g_eff * sdr;
            (sn, kn, kt)
        } else {
            (0.0, 0.0, 0.0)
        };

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
        let f_n_mag = if surface_energy > 0.0 {
            // JKR simplified explicit model
            let f_adhesion = 1.5 * std::f64::consts::PI * surface_energy * r_eff;
            if jkr_adhesion_only {
                // Adhesion-only regime: no Hertz, no damping
                -f_adhesion
            } else {
                // Full contact: Hertz + damping + adhesion
                let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                k_n * delta - f_diss_n - f_adhesion
            }
        } else if cohesion_energy > 0.0 {
            // SJKR cohesion: contact area A = pi * delta * r_eff
            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
            let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
            k_n * delta - f_diss_n - f_cohesion // can go negative (attractive)
        } else {
            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
            (k_n * delta - f_diss_n).max(0.0)
        };

        let fn_x = f_n_mag * nx;
        let fn_y = f_n_mag * ny;
        let fn_z = f_n_mag * nz;

        atoms.force[i][0] -= fn_x;
        atoms.force[i][1] -= fn_y;
        atoms.force[i][2] -= fn_z;
        atoms.force[j][0] += fn_x;
        atoms.force[j][1] += fn_y;
        atoms.force[j][2] += fn_z;

        // ── Tangential force (skip in JKR adhesion-only regime) ──────────
        if jkr_adhesion_only {
            // No tangential, rolling, or spring history in adhesion-only regime
            // Virial contribution from normal only
            if let Some(ref mut v) = virial {
                if v.active {
                    v.add_pair(dx, dy, dz, -fn_x, -fn_y, -fn_z);
                }
            }
            continue;
        }

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
            None => [0.0; 3],
        };

        let mut sx = sign * stored_spring[0];
        let mut sy = sign * stored_spring[1];
        let mut sz = sign * stored_spring[2];
        let s_dot_n = sx*nx + sy*ny + sz*nz;
        sx -= s_dot_n * nx; sy -= s_dot_n * ny; sz -= s_dot_n * nz;
        sx += vt_x * dt;
        sy += vt_y * dt;
        sz += vt_z * dt;

        // Coulomb cap on spring
        let s_mag = (sx*sx + sy*sy + sz*sz).sqrt();
        let f_t_spring_mag = k_t * s_mag;
        let f_t_max = mu * f_n_mag.abs();
        if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
            let scale = f_t_max / f_t_spring_mag;
            sx *= scale; sy *= scale; sz *= scale;
        }

        // Tangential force with damping (gamma_t > 0, opposes sliding velocity)
        let gamma_t = 2.0 * SQRT_5_3 * beta * (k_t * m_r).sqrt();
        let mut ft_x = k_t * sx - gamma_t * vt_x;
        let mut ft_y = k_t * sy - gamma_t * vt_y;
        let mut ft_z = k_t * sz - gamma_t * vt_z;

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

        // ── Rolling resistance torque ───────────────────────────────────
        if mu_r > 0.0 {
            // Relative angular velocity
            let or_x = omega_ix - omega_jx;
            let or_y = omega_iy - omega_jy;
            let or_z = omega_iz - omega_jz;
            // Remove twisting component (keep only rolling)
            let or_dot_n = or_x * nx + or_y * ny + or_z * nz;
            let roll_x = or_x - or_dot_n * nx;
            let roll_y = or_y - or_dot_n * ny;
            let roll_z = or_z - or_dot_n * nz;
            let roll_mag = (roll_x * roll_x + roll_y * roll_y + roll_z * roll_z).sqrt();
            if roll_mag > 1e-30 {
                let tau_mag = mu_r * f_n_mag.abs() * r_eff;
                let inv_roll = tau_mag / roll_mag;
                let tr_x = -inv_roll * roll_x;
                let tr_y = -inv_roll * roll_y;
                let tr_z = -inv_roll * roll_z;
                dem.torque[i][0] += tr_x;
                dem.torque[i][1] += tr_y;
                dem.torque[i][2] += tr_z;
                dem.torque[j][0] -= tr_x;
                dem.torque[j][1] -= tr_y;
                dem.torque[j][2] -= tr_z;
            }
        }

        // Virial: force on i from j = (-fn + ft)
        if let Some(ref mut v) = virial {
            if v.active {
                v.add_pair(dx, dy, dz, -fn_x + ft_x, -fn_y + ft_y, -fn_z + ft_z);
            }
        }

        // Store updated spring back (canonical form) and mark active
        let new_spring = [sign * sx, sign * sy, sign * sz];
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
    fn fused_contact_repulsive_for_overlap() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 0,
            [0.0, 0.0, 0.0], radius,
        );
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [0.0019, 0.0, 0.0], radius,
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

        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 0,
            [0.0, 0.0, 0.0], radius,
        );
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [0.0019, 0.0, 0.0], radius,
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

        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 0,
            [0.0, 0.0, 0.0], radius,
        );
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [0.003, 0.0, 0.0], radius,
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

    fn make_material_table_cohesion() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 1e9);
        mt.build_pair_tables();
        mt
    }

    fn make_material_table_rolling() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.3, 0.0);
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn cohesion_produces_attractive_force() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        // Very small overlap with high cohesion energy → cohesion dominates
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [0.00199999, 0.0, 0.0], radius, // delta = 1e-8 (tiny overlap)
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
        app.add_resource(make_material_table_cohesion());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // With cohesion and small overlap, normal force on atom 0 should be positive (attractive toward atom 1)
        assert!(
            atom.force[0][0] > 0.0,
            "cohesion should make force attractive on atom 0, got {}",
            atom.force[0][0]
        );
        // Newton's 3rd law
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "forces should be equal and opposite"
        );
    }

    #[test]
    fn zero_cohesion_matches_original() {
        // Two identical setups — one with default table, one with explicit 0.0 cohesion
        let radius = 0.001;
        let sep = 0.0019;

        let run = |mt: MaterialTable| -> [f64; 3] {
            let mut app = App::new();
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            let mut hist = ContactHistoryStore::new();
            atom.dt = 1e-7;
            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [sep, 0.0, 0.0], radius);
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
            app.add_resource(mt);
            app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            atom.force[0]
        };

        let f_default = run(make_material_table());
        let mut mt_zero = MaterialTable::new();
        mt_zero.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0);
        mt_zero.build_pair_tables();
        let f_zero = run(mt_zero);

        for d in 0..3 {
            assert!(
                (f_default[d] - f_zero[d]).abs() < 1e-15,
                "zero params should reproduce original, dim {} default={} zero={}",
                d, f_default[d], f_zero[d]
            );
        }
    }

    fn make_material_table_jkr() -> MaterialTable {
        let mut mt = MaterialTable::new();
        // Use high surface energy (1.0 J/m²) so adhesion clearly dominates at small overlaps
        mt.add_material_full("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0, 1.0);
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn jkr_pulloff_force_matches_theory() {
        // Test in adhesion-only regime (gap, not overlap) where force = -F_adhesion exactly
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        let gamma = 1.0;
        let r_eff = radius / 2.0;

        // Place particles with a tiny gap (adhesion-only regime)
        let gap = 1e-9;
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [2.0 * radius + gap, 0.0, 0.0], radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        let mt = make_material_table_jkr();
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let expected_pulloff = 1.5 * std::f64::consts::PI * gamma * r_eff;
        // In adhesion-only regime, force should be exactly -F_adhesion
        // Force on atom 0 should be positive (attracted toward atom 1)
        assert!(
            atom.force[0][0] > 0.0,
            "JKR should produce attractive force, got {}",
            atom.force[0][0]
        );
        // f_n_mag = -F_adhesion, force[0] -= f_n_mag * nx → force[0] += F_adhesion
        let f_mag = atom.force[0][0];
        assert!(
            (f_mag - expected_pulloff).abs() / expected_pulloff < 1e-6,
            "pull-off force should match theory {}, got {}",
            expected_pulloff, f_mag
        );
    }

    #[test]
    fn jkr_adhesion_only_regime() {
        // Two particles with a small gap (no geometric overlap) but within JKR range
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        // Gap of 1e-9 (very small, within JKR pull-off distance for gamma=1.0)
        let gap = 1e-9;
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [2.0 * radius + gap, 0.0, 0.0], radius,
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
        app.add_resource(make_material_table_jkr());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Should be attractive (atom 0 pulled toward atom 1 = positive x)
        assert!(
            atom.force[0][0] > 0.0,
            "JKR adhesion-only should attract, got {}",
            atom.force[0][0]
        );
        // Newton's 3rd law
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "forces should be equal and opposite"
        );
    }

    #[test]
    fn jkr_no_interaction_beyond_pulloff() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        // Large gap — well beyond JKR pull-off distance
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(
            &mut atom, &mut dem, &mut hist, 1,
            [0.003, 0.0, 0.0], radius, // gap = 0.001 >> delta_pulloff
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
        app.add_resource(make_material_table_jkr());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0].abs() < 1e-20, "no force beyond pull-off distance");
    }

    #[test]
    fn rolling_resistance_opposes_angular_velocity() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        // Give atom 0 a rolling angular velocity (around y-axis — perpendicular to contact normal x)
        dem.omega[0] = [0.0, 100.0, 0.0];
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
        app.add_resource(make_material_table_rolling());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // Rolling torque on atom 0 should oppose its angular velocity (negative y)
        assert!(
            dem.torque[0][1] < 0.0,
            "rolling torque should oppose omega_y, got {}",
            dem.torque[0][1]
        );
    }
}
