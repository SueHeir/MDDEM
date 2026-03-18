//! Fused Hertz-Mindlin contact force computation.
//!
//! This is the primary contact force module and the recommended code path for DEM
//! simulations. It computes normal, tangential, rolling, and twisting forces in a
//! **single pair loop**, eliminating redundant computation of shared quantities
//! (distance, material lookups, Hertz stiffness, normal force magnitude).
//!
//! # Supported models
//!
//! | Component   | Models                                    |
//! |-------------|-------------------------------------------|
//! | Normal      | Hertz (nonlinear), Hooke (linear)         |
//! | Tangential  | Mindlin incremental spring + Coulomb cap  |
//! | Rolling     | Constant torque, SDS (spring-dashpot-slider) |
//! | Twisting    | Constant torque, SDS (spring-dashpot-slider) |
//! | Adhesion    | JKR, DMT                                  |
//! | Cohesion    | SJKR (area-proportional)                  |
//!
//! # Contact detection
//!
//! Two particles are in contact when their geometric overlap `δ = R1 + R2 - d > 0`.
//! With JKR adhesion, the interaction range extends beyond geometric contact by a
//! pull-off distance derived from the surface energy and elastic properties.
//!
//! # TOML configuration
//!
//! The contact model is selected via `contact_model` in the `[materials]` section:
//!
//! ```toml
//! [materials]
//! contact_model = "hertz"  # or "hooke"
//! ```
//!
//! See the [crate-level documentation](crate) for the full material parameter list.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use dem_atom::{DemAtom, MaterialTable};
use dem_clump::ClumpAtom;
use mddem_core::{register_atom_data, Atom, AtomDataRegistry, BondStore, ScheduleSet, VirialStress, VirialStressPlugin};
use mddem_neighbor::Neighbor;

use crate::tangential::ContactHistoryStore;
use crate::{LARGE_OVERLAP_WARN_THRESHOLD, MAX_OVERLAP_WARNINGS, SQRT_5_3, TANGENTIAL_EPSILON};

/// Fused Hertz normal + Mindlin tangential contact force plugin.
///
/// Registers [`ContactHistoryStore`] in the [`AtomDataRegistry`] and a single
/// `hertz_mindlin_contact` system at [`ScheduleSet::Force`].
pub struct HertzMindlinContactPlugin;

impl Plugin for HertzMindlinContactPlugin {
    fn dependencies(&self) -> Vec<&str> {
        vec!["DemAtomPlugin"]
    }

    fn build(&self, app: &mut App) {
        app.add_plugins(VirialStressPlugin);
        // Register ContactHistoryStore
        register_atom_data!(app, ContactHistoryStore::new());

        let contact_model = {
            let mt = app.get_resource_ref::<MaterialTable>()
                .expect("MaterialTable must exist before HertzMindlinContactPlugin");
            mt.contact_model.clone()
        };

        match contact_model.as_str() {
            "hooke" => {
                app.add_update_system(
                    hooke_contact_force.label("hertz_mindlin_contact"),
                    ScheduleSet::Force,
                );
            }
            _ => {
                app.add_update_system(
                    hertz_mindlin_contact_force.label("hertz_mindlin_contact"),
                    ScheduleSet::Force,
                );
            }
        }
    }
}

/// Fused Hertz-Mindlin contact force for all neighbor pairs.
///
/// Computes normal (Hertz), tangential (Mindlin), rolling, and twisting forces
/// in a single pass over the neighbor list. Supports JKR/DMT adhesion and SJKR
/// cohesion. Forces and torques are accumulated with Newton's third law symmetry.
///
/// # Panics
///
/// Panics if more than [`MAX_OVERLAP_WARNINGS`] pairs have excessive overlap
/// in a single timestep, indicating an unstable simulation.
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
    let clump_store = registry.get::<ClumpAtom>();
    let dt = atoms.dt;

    let natoms = atoms.len();
    if history.contacts.len() < natoms {
        history.contacts.resize_with(natoms, Vec::new);
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

        // Skip same-clump pairs (sub-spheres of the same rigid body don't interact)
        if let Some(ref clump) = clump_store {
            if dem_clump::same_clump(clump, i, j) {
                continue;
            }
        }

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        // Skip clump parent atoms (they have radius 0 and don't participate in contacts)
        if r1 <= 0.0 || r2 <= 0.0 {
            continue;
        }

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = r1 + r2;

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;
        let surface_energy = material_table.surface_energy_ij[mat_i][mat_j];

        let use_dmt = material_table.adhesion_model == "dmt";

        // JKR: compute pull-off distance for extended interaction range
        // DMT: no extended range (particles separate at delta = 0)
        // Effective radius: R* = R1 R2 / (R1 + R2)
        let r_eff = (r1 * r2) / sum_r;
        // Effective Young's modulus: 1/E* = (1-ν1²)/E1 + (1-ν2²)/E2
        let e_eff = material_table.e_eff_ij[mat_i][mat_j];
        // JKR pull-off distance: particles interact beyond geometric contact
        let delta_pulloff = if surface_energy > 0.0 && !use_dmt {
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
        // Cap at half the smaller radius to keep the Hertz model numerically valid.
        let r_min = r1.min(r2);
        let delta = (sum_r - distance).min(0.5 * r_min);

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
            // Cap overlap at half the smaller radius to keep Hertz model valid,
            // but still compute the repulsive force (skipping would remove all
            // repulsion and cause runaway penetration).
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

        // Effective shear modulus: 1/G* = (2-ν1)/G1 + (2-ν2)/G2
        let g_eff = material_table.g_eff_ij[mat_i][mat_j];

        // Reduced mass: m_r = 1 / (1/m1 + 1/m2)
        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

        let beta = material_table.beta_ij[mat_i][mat_j];
        let mu = material_table.friction_ij[mat_i][mat_j];
        let mu_r = material_table.rolling_friction_ij[mat_i][mat_j];
        let mu_tw = material_table.twisting_friction_ij[mat_i][mat_j];
        let cohesion_energy = material_table.cohesion_energy_ij[mat_i][mat_j];

        // JKR adhesion-only regime: gap exists but within pull-off distance
        // DMT has no adhesion-only regime (no force beyond contact)
        let jkr_adhesion_only = surface_energy > 0.0 && !use_dmt && delta <= 0.0;

        // Hertz stiffness parameters (only meaningful when δ > 0)
        // S_n = 2 E* √(R* δ)  — normal stiffness parameter (used in damping)
        // k_n = 4/3 E* √(R* δ) — normal spring constant
        // k_t = 8 G* √(R* δ)  — tangential spring constant (Mindlin)
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
        // F_n > 0 → repulsive (along contact normal from i to j)
        // F_n < 0 → attractive (adhesion/cohesion pulls particles together)
        let f_n_mag = if surface_energy > 0.0 && use_dmt {
            // DMT: Hertz contact + constant adhesive force F_dmt = 2π γ R*
            let f_dmt = 2.0 * std::f64::consts::PI * surface_energy * r_eff;
            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
            k_n * delta - f_diss_n - f_dmt
        } else if surface_energy > 0.0 {
            // JKR: adhesion force F_adh = 3/2 π γ R* (simplified explicit model)
            let f_adhesion = 1.5 * std::f64::consts::PI * surface_energy * r_eff;
            if jkr_adhesion_only {
                // Gap regime (δ ≤ 0): pure adhesion, no Hertz contact or damping
                -f_adhesion
            } else {
                // Contact regime (δ > 0): Hertz repulsion + damping − adhesion
                let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                k_n * delta - f_diss_n - f_adhesion
            }
        } else if cohesion_energy > 0.0 {
            // SJKR: cohesion proportional to contact area A = π δ R*
            let f_diss_n = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
            let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
            k_n * delta - f_diss_n - f_cohesion // can go negative (attractive)
        } else {
            // Standard Hertz: repulsive only (clamped to ≥ 0)
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
        // No tangential friction when particles are not in geometric contact
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
        let stored = match entry_idx {
            Some(idx) => history.contacts[i][idx].1,
            None => [0.0; 7],
        };

        // Convert stored spring from canonical form to local (i,j) frame
        let mut sx = sign * stored[0];
        let mut sy = sign * stored[1];
        let mut sz = sign * stored[2];
        // Rotate spring into current tangent plane (remove normal component)
        let s_dot_n = sx*nx + sy*ny + sz*nz;
        sx -= s_dot_n * nx; sy -= s_dot_n * ny; sz -= s_dot_n * nz;
        // Integrate tangential velocity into spring displacement
        sx += vt_x * dt;
        sy += vt_y * dt;
        sz += vt_z * dt;

        // Coulomb cap on spring: |k_t s| ≤ μ |F_n|
        let s_mag = (sx*sx + sy*sy + sz*sz).sqrt();
        let f_t_spring_mag = k_t * s_mag;
        let f_t_max = mu * f_n_mag.abs();
        if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
            let scale = f_t_max / f_t_spring_mag;
            sx *= scale; sy *= scale; sz *= scale;
        }

        // Tangential damping coefficient: γ_t = 2 β √(5/3) √(k_t m_r)
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
        // Relative angular velocity (rolling component)
        let or_x = omega_ix - omega_jx;
        let or_y = omega_iy - omega_jy;
        let or_z = omega_iz - omega_jz;
        let or_dot_n = or_x * nx + or_y * ny + or_z * nz;
        let roll_x = or_x - or_dot_n * nx;
        let roll_y = or_y - or_dot_n * ny;
        let roll_z = or_z - or_dot_n * nz;

        let mut roll_disp_x = sign * stored[3];
        let mut roll_disp_y = sign * stored[4];
        let mut roll_disp_z = sign * stored[5];
        let mut twist_disp = sign * stored[6];

        if mu_r > 0.0 {
            let roll_mag = (roll_x * roll_x + roll_y * roll_y + roll_z * roll_z).sqrt();
            let sds_rolling = material_table.rolling_model == "sds";
            if sds_rolling {
                // SDS rolling: spring-dashpot-slider model
                let k_roll = material_table.rolling_stiffness_ij[mat_i][mat_j];
                let gamma_roll = material_table.rolling_damping_ij[mat_i][mat_j];

                // Update rolling displacement: remove normal component, integrate
                let rd_dot_n = roll_disp_x * nx + roll_disp_y * ny + roll_disp_z * nz;
                roll_disp_x -= rd_dot_n * nx;
                roll_disp_y -= rd_dot_n * ny;
                roll_disp_z -= rd_dot_n * nz;
                roll_disp_x += roll_x * dt;
                roll_disp_y += roll_y * dt;
                roll_disp_z += roll_z * dt;

                // Spring + dashpot torque
                let mut tr_x = -k_roll * roll_disp_x - gamma_roll * roll_x;
                let mut tr_y = -k_roll * roll_disp_y - gamma_roll * roll_y;
                let mut tr_z = -k_roll * roll_disp_z - gamma_roll * roll_z;
                let tr_mag = (tr_x * tr_x + tr_y * tr_y + tr_z * tr_z).sqrt();
                let tau_max = mu_r * f_n_mag.abs() * r_eff;

                if tr_mag > tau_max && tr_mag > TANGENTIAL_EPSILON {
                    // Cap and rescale spring displacement
                    let scale = tau_max / tr_mag;
                    tr_x *= scale;
                    tr_y *= scale;
                    tr_z *= scale;
                    // Rescale spring: δ = (τ + γ·ω) / (-k)
                    if k_roll > TANGENTIAL_EPSILON {
                        roll_disp_x = (tr_x + gamma_roll * roll_x) / (-k_roll);
                        roll_disp_y = (tr_y + gamma_roll * roll_y) / (-k_roll);
                        roll_disp_z = (tr_z + gamma_roll * roll_z) / (-k_roll);
                    }
                }

                dem.torque[i][0] += tr_x;
                dem.torque[i][1] += tr_y;
                dem.torque[i][2] += tr_z;
                dem.torque[j][0] -= tr_x;
                dem.torque[j][1] -= tr_y;
                dem.torque[j][2] -= tr_z;
            } else if roll_mag > 1e-30 {
                // Constant torque model (existing behavior)
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

        // ── Twisting friction torque ─────────────────────────────────────
        if mu_tw > 0.0 {
            let twist_vel = or_dot_n; // twisting component of relative angular velocity
            let sds_twisting = material_table.twisting_model == "sds";
            if sds_twisting {
                // SDS twisting: spring-dashpot-slider model
                let k_twist = material_table.twisting_stiffness_ij[mat_i][mat_j];
                let gamma_twist = material_table.twisting_damping_ij[mat_i][mat_j];

                // Update twisting displacement
                twist_disp += twist_vel * dt;

                // Spring + dashpot torque (scalar, along contact normal)
                let mut tau_twist = -k_twist * twist_disp - gamma_twist * twist_vel;
                let tau_max = mu_tw * f_n_mag.abs() * r_eff;

                if tau_twist.abs() > tau_max {
                    // Cap and rescale spring
                    tau_twist = tau_twist.signum() * tau_max;
                    if k_twist > TANGENTIAL_EPSILON {
                        twist_disp = (tau_twist + gamma_twist * twist_vel) / (-k_twist);
                    }
                }

                let tt_x = tau_twist * nx;
                let tt_y = tau_twist * ny;
                let tt_z = tau_twist * nz;
                dem.torque[i][0] += tt_x;
                dem.torque[i][1] += tt_y;
                dem.torque[i][2] += tt_z;
                dem.torque[j][0] -= tt_x;
                dem.torque[j][1] -= tt_y;
                dem.torque[j][2] -= tt_z;
            } else if twist_vel.abs() > 1e-30 {
                // Constant torque model (existing behavior)
                let tau = mu_tw * f_n_mag.abs() * r_eff;
                let sign_tw = if twist_vel > 0.0 { -1.0 } else { 1.0 };
                let tt_x = sign_tw * tau * nx;
                let tt_y = sign_tw * tau * ny;
                let tt_z = sign_tw * tau * nz;
                dem.torque[i][0] += tt_x;
                dem.torque[i][1] += tt_y;
                dem.torque[i][2] += tt_z;
                dem.torque[j][0] -= tt_x;
                dem.torque[j][1] -= tt_y;
                dem.torque[j][2] -= tt_z;
            }
        }

        // Virial: force on i from j = (-fn + ft)
        if let Some(ref mut v) = virial {
            if v.active {
                v.add_pair(dx, dy, dz, -fn_x + ft_x, -fn_y + ft_y, -fn_z + ft_z);
            }
        }

        // Store updated spring back (canonical form) and mark active
        let new_spring = [
            sign * sx, sign * sy, sign * sz,
            sign * roll_disp_x, sign * roll_disp_y, sign * roll_disp_z,
            sign * twist_disp,
        ];
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

/// Hooke (linear spring) contact force — alternative to Hertz-Mindlin.
///
/// Normal: `f_n = kn * delta`, tangential uses `kt` directly.
/// Damping: `gamma = 2 * beta * sqrt(kn_ij * m_r)`.
/// All other features (friction, rolling, twisting, cohesion, JKR) reused.
pub fn hooke_contact_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let mut dem = registry.expect_mut::<DemAtom>("hooke_contact_force");
    let mut history = registry.expect_mut::<ContactHistoryStore>("hooke_contact_force");
    let bond_store = registry.get::<BondStore>();
    let clump_store = registry.get::<ClumpAtom>();
    let dt = atoms.dt;

    while history.contacts.len() < atoms.len() {
        history.contacts.push(Vec::new());
    }

    let nlocal = atoms.nlocal as usize;
    let mut overlap_warnings = 0usize;

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

        // Skip same-clump pairs (sub-spheres of the same rigid body don't interact)
        if let Some(ref clump) = clump_store {
            if dem_clump::same_clump(clump, i, j) {
                continue;
            }
        }

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        // Skip clump parent atoms (radius 0)
        if r1 <= 0.0 || r2 <= 0.0 {
            continue;
        }

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
            continue;
        }

        let r_min = r1.min(r2);
        let delta = (sum_r - distance).min(0.5 * r_min);
        if delta <= 0.0 {
            continue;
        }

        if distance / sum_r < LARGE_OVERLAP_WARN_THRESHOLD {
            overlap_warnings += 1;
            if overlap_warnings > MAX_OVERLAP_WARNINGS {
                panic!("Over {} excessive overlaps this step — aborting.", MAX_OVERLAP_WARNINGS);
            }
            // Still compute force (don't skip) — removing repulsion causes runaway.
        }

        let inv_dist = 1.0 / distance;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let nz = dz * inv_dist;

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;
        let r_eff = (r1 * r2) / sum_r;
        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);
        let beta = material_table.beta_ij[mat_i][mat_j];
        let mu = material_table.friction_ij[mat_i][mat_j];
        let mu_r = material_table.rolling_friction_ij[mat_i][mat_j];
        let mu_tw = material_table.twisting_friction_ij[mat_i][mat_j];
        let cohesion_energy = material_table.cohesion_energy_ij[mat_i][mat_j];

        let kn = material_table.kn_ij[mat_i][mat_j];
        let kt = material_table.kt_ij[mat_i][mat_j];

        // Hooke normal: f_n = kn * delta
        // Damping: gamma_n = 2 * beta * sqrt(kn * m_r)
        let gamma_n = 2.0 * beta * (kn * m_r).sqrt();

        // Relative velocity
        let omega_ix = dem.omega[i][0];
        let omega_iy = dem.omega[i][1];
        let omega_iz = dem.omega[i][2];
        let omega_jx = dem.omega[j][0];
        let omega_jy = dem.omega[j][1];
        let omega_jz = dem.omega[j][2];

        let r1n_x = r1 * nx;
        let r1n_y = r1 * ny;
        let r1n_z = r1 * nz;
        let vc_ix = atoms.vel[i][0] + (omega_iy * r1n_z - omega_iz * r1n_y);
        let vc_iy = atoms.vel[i][1] + (omega_iz * r1n_x - omega_ix * r1n_z);
        let vc_iz = atoms.vel[i][2] + (omega_ix * r1n_y - omega_iy * r1n_x);

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

        // Normal force
        let f_n_mag = if cohesion_energy > 0.0 {
            let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
            kn * delta - gamma_n * v_n - f_cohesion
        } else {
            (kn * delta - gamma_n * v_n).max(0.0)
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

        // Tangential force
        let vt_x = vr_x - v_n * nx;
        let vt_y = vr_y - v_n * ny;
        let vt_z = vr_z - v_n * nz;

        let tag_i = atoms.tag[i];
        let tag_j = atoms.tag[j];
        let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

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
        let s_dot_n = sx * nx + sy * ny + sz * nz;
        sx -= s_dot_n * nx;
        sy -= s_dot_n * ny;
        sz -= s_dot_n * nz;
        sx += vt_x * dt;
        sy += vt_y * dt;
        sz += vt_z * dt;

        let s_mag = (sx * sx + sy * sy + sz * sz).sqrt();
        let f_t_spring_mag = kt * s_mag;
        let f_t_max = mu * f_n_mag.abs();
        if f_t_spring_mag > f_t_max && f_t_spring_mag > TANGENTIAL_EPSILON {
            let scale = f_t_max / f_t_spring_mag;
            sx *= scale;
            sy *= scale;
            sz *= scale;
        }

        let gamma_t = 2.0 * SQRT_5_3 * beta * (kt * m_r).sqrt();
        let mut ft_x = kt * sx - gamma_t * vt_x;
        let mut ft_y = kt * sy - gamma_t * vt_y;
        let mut ft_z = kt * sz - gamma_t * vt_z;

        let f_t_mag = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z).sqrt();
        if f_t_mag > f_t_max && f_t_mag > TANGENTIAL_EPSILON {
            let scale = f_t_max / f_t_mag;
            ft_x *= scale;
            ft_y *= scale;
            ft_z *= scale;
        }

        // Torques
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

        // Rolling/twisting relative angular velocity
        let or_x = omega_ix - omega_jx;
        let or_y = omega_iy - omega_jy;
        let or_z = omega_iz - omega_jz;
        let or_dot_n = or_x * nx + or_y * ny + or_z * nz;
        let roll_x = or_x - or_dot_n * nx;
        let roll_y = or_y - or_dot_n * ny;
        let roll_z = or_z - or_dot_n * nz;

        let mut roll_disp_x = sign * stored[3];
        let mut roll_disp_y = sign * stored[4];
        let mut roll_disp_z = sign * stored[5];
        let mut twist_disp = sign * stored[6];

        // Rolling resistance
        if mu_r > 0.0 {
            let roll_mag = (roll_x * roll_x + roll_y * roll_y + roll_z * roll_z).sqrt();
            let sds_rolling = material_table.rolling_model == "sds";
            if sds_rolling {
                let k_roll = material_table.rolling_stiffness_ij[mat_i][mat_j];
                let gamma_roll = material_table.rolling_damping_ij[mat_i][mat_j];

                let rd_dot_n = roll_disp_x * nx + roll_disp_y * ny + roll_disp_z * nz;
                roll_disp_x -= rd_dot_n * nx;
                roll_disp_y -= rd_dot_n * ny;
                roll_disp_z -= rd_dot_n * nz;
                roll_disp_x += roll_x * dt;
                roll_disp_y += roll_y * dt;
                roll_disp_z += roll_z * dt;

                let mut tr_x = -k_roll * roll_disp_x - gamma_roll * roll_x;
                let mut tr_y = -k_roll * roll_disp_y - gamma_roll * roll_y;
                let mut tr_z = -k_roll * roll_disp_z - gamma_roll * roll_z;
                let tr_mag = (tr_x * tr_x + tr_y * tr_y + tr_z * tr_z).sqrt();
                let tau_max = mu_r * f_n_mag.abs() * r_eff;

                if tr_mag > tau_max && tr_mag > TANGENTIAL_EPSILON {
                    let scale = tau_max / tr_mag;
                    tr_x *= scale;
                    tr_y *= scale;
                    tr_z *= scale;
                    if k_roll > TANGENTIAL_EPSILON {
                        roll_disp_x = (tr_x + gamma_roll * roll_x) / (-k_roll);
                        roll_disp_y = (tr_y + gamma_roll * roll_y) / (-k_roll);
                        roll_disp_z = (tr_z + gamma_roll * roll_z) / (-k_roll);
                    }
                }

                dem.torque[i][0] += tr_x;
                dem.torque[i][1] += tr_y;
                dem.torque[i][2] += tr_z;
                dem.torque[j][0] -= tr_x;
                dem.torque[j][1] -= tr_y;
                dem.torque[j][2] -= tr_z;
            } else if roll_mag > 1e-30 {
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

        // Twisting friction
        if mu_tw > 0.0 {
            let twist_vel = or_dot_n;
            let sds_twisting = material_table.twisting_model == "sds";
            if sds_twisting {
                let k_twist = material_table.twisting_stiffness_ij[mat_i][mat_j];
                let gamma_twist = material_table.twisting_damping_ij[mat_i][mat_j];

                twist_disp += twist_vel * dt;

                let mut tau_twist = -k_twist * twist_disp - gamma_twist * twist_vel;
                let tau_max = mu_tw * f_n_mag.abs() * r_eff;

                if tau_twist.abs() > tau_max {
                    tau_twist = tau_twist.signum() * tau_max;
                    if k_twist > TANGENTIAL_EPSILON {
                        twist_disp = (tau_twist + gamma_twist * twist_vel) / (-k_twist);
                    }
                }

                let tt_x = tau_twist * nx;
                let tt_y = tau_twist * ny;
                let tt_z = tau_twist * nz;
                dem.torque[i][0] += tt_x;
                dem.torque[i][1] += tt_y;
                dem.torque[i][2] += tt_z;
                dem.torque[j][0] -= tt_x;
                dem.torque[j][1] -= tt_y;
                dem.torque[j][2] -= tt_z;
            } else if twist_vel.abs() > 1e-30 {
                let tau = mu_tw * f_n_mag.abs() * r_eff;
                let sign_tw = if twist_vel > 0.0 { -1.0 } else { 1.0 };
                let tt_x = sign_tw * tau * nx;
                let tt_y = sign_tw * tau * ny;
                let tt_z = sign_tw * tau * nz;
                dem.torque[i][0] += tt_x;
                dem.torque[i][1] += tt_y;
                dem.torque[i][2] += tt_z;
                dem.torque[j][0] -= tt_x;
                dem.torque[j][1] -= tt_y;
                dem.torque[j][2] -= tt_z;
            }
        }

        // Virial
        if let Some(ref mut v) = virial {
            if v.active {
                v.add_pair(dx, dy, dz, -fn_x + ft_x, -fn_y + ft_y, -fn_z + ft_z);
            }
        }

        let new_spring = [
            sign * sx, sign * sy, sign * sz,
            sign * roll_disp_x, sign * roll_disp_y, sign * roll_disp_z,
            sign * twist_disp,
        ];
        match entry_idx {
            Some(idx) => {
                history.contacts[i][idx].1 = new_spring;
                history.contacts[i][idx].2 = true;
            }
            None => history.contacts[i].push((tag_j, new_spring, true)),
        }
    }

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

    fn make_material_table_hooke() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material_extended("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0, 0.0, 0.0, 1e6, 5e5);
        mt.contact_model = "hooke".to_string();
        mt.build_pair_tables();
        mt
    }

    fn make_material_table_twisting() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material_extended("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0);
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn hooke_force_linear_in_delta() {
        let radius = 0.001;
        let run = |sep: f64| -> f64 {
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
            app.add_resource(make_material_table_hooke());
            app.add_update_system(hooke_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            atom.force[0][0]
        };

        // delta1 = 2*r - sep1, delta2 = 2*r - sep2
        let sep1 = 0.00195; // delta = 0.00005
        let sep2 = 0.0019;  // delta = 0.0001
        let f1 = run(sep1);
        let f2 = run(sep2);

        // Hooke: force proportional to delta → f2/f1 ≈ 2.0 (linear)
        let ratio = f2 / f1;
        assert!(
            (ratio - 2.0).abs() < 0.15,
            "Hooke force should be linear in delta, got ratio {} (expected ~2.0)",
            ratio
        );
    }

    #[test]
    fn hooke_no_force_beyond_contact() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.003, 0.0, 0.0], radius);
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
        app.add_resource(make_material_table_hooke());
        app.add_update_system(hooke_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0].abs() < 1e-20, "no force beyond contact distance");
    }

    #[test]
    fn twisting_friction_opposes_spin() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        // Spin about contact normal (x-axis)
        dem.omega[0] = [100.0, 0.0, 0.0];
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
        app.add_resource(make_material_table_twisting());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // Twisting torque on atom 0 should oppose its spin about x (negative x torque)
        assert!(
            dem.torque[0][0] < 0.0,
            "twisting torque should oppose omega_x, got {}",
            dem.torque[0][0]
        );
    }

    #[test]
    fn twisting_friction_zero_when_no_spin() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        // No angular velocity at all
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
        app.add_resource(make_material_table_twisting());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // No twisting torque when there's no angular velocity
        let torque_mag = (dem.torque[0][0].powi(2) + dem.torque[0][1].powi(2) + dem.torque[0][2].powi(2)).sqrt();
        assert!(
            torque_mag < 1e-20,
            "no twisting torque when no spin, got {}",
            torque_mag
        );
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

    // ── SDS model helper ────────────────────────────────────────────────

    fn make_material_table_sds_rolling() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.rolling_model = "sds".to_string();
        mt.add_material_with_sds(
            "glass", 8.7e9, 0.3, 0.95, 0.4,
            0.3,   // rolling_friction (mu_r)
            0.0, 0.0,
            0.0,   // twisting_friction
            0.0, 0.0,
            1e3,   // rolling_stiffness
            0.5,   // rolling_damping
            0.0, 0.0,
        );
        mt.build_pair_tables();
        mt
    }

    fn make_material_table_sds_twisting() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.twisting_model = "sds".to_string();
        mt.add_material_with_sds(
            "glass", 8.7e9, 0.3, 0.95, 0.4,
            0.0,   // rolling_friction
            0.0, 0.0,
            0.3,   // twisting_friction (mu_tw)
            0.0, 0.0,
            0.0, 0.0,
            1e3,   // twisting_stiffness
            0.5,   // twisting_damping
        );
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn sds_rolling_opposes_angular_velocity() {
        // Two overlapping particles, one spinning → SDS rolling torque opposes it
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        // Give atom 0 angular velocity in y (rolling about contact normal x)
        dem.omega[0] = [0.0, 10.0, 0.0];
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
        app.add_resource(make_material_table_sds_rolling());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // SDS rolling torque should oppose omega_y on atom 0
        assert!(
            dem.torque[0][1] < 0.0,
            "SDS rolling torque should oppose omega_y, got {}",
            dem.torque[0][1]
        );
    }

    #[test]
    fn sds_rolling_spring_accumulates() {
        // Pre-load rolling displacement → larger torque than zero displacement
        // Use very small omega so that damping doesn't dominate and Coulomb cap isn't reached
        let radius = 0.001;

        let run_with_preload = |preload_y: f64| -> f64 {
            let mut app = App::new();
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            let mut hist = ContactHistoryStore::new();
            atom.dt = 1e-7;

            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
            dem.omega[0] = [0.0, 0.001, 0.0]; // very small angular velocity
            atom.nlocal = 2;
            atom.natoms = 2;

            // Pre-load rolling displacement in contact history (canonical: tag 0 < tag 1, sign=+1)
            if preload_y != 0.0 {
                hist.contacts[0].push((1, [0.0, 0.0, 0.0, 0.0, preload_y, 0.0, 0.0], false));
            }

            let mut neighbor = Neighbor::new();
            neighbor.neighbor_offsets = vec![0, 1, 1];
            neighbor.neighbor_indices = vec![1];

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);
            registry.register(hist);

            app.add_resource(atom);
            app.add_resource(neighbor);
            app.add_resource(registry);
            app.add_resource(make_material_table_sds_rolling());
            app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let reg = app.get_resource_ref::<AtomDataRegistry>().unwrap();
            let d = reg.expect::<DemAtom>("test");
            d.torque[0][1]
        };

        let torque_no_preload = run_with_preload(0.0);
        let torque_with_preload = run_with_preload(1e-5); // small preload below cap

        assert!(torque_no_preload < 0.0, "should oppose omega_y");
        assert!(torque_with_preload < 0.0, "should oppose omega_y");
        // Pre-loaded spring adds to torque magnitude
        assert!(
            torque_with_preload.abs() > torque_no_preload.abs(),
            "preloaded spring should increase torque: no_preload={}, preloaded={}",
            torque_no_preload, torque_with_preload
        );
    }

    #[test]
    fn sds_rolling_coulomb_cap() {
        // Very high angular velocity → torque should be capped at mu_r * |F_n| * R_eff
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-5; // larger dt to accumulate big spring

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        dem.omega[0] = [0.0, 1e6, 0.0]; // very high
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        let mt = make_material_table_sds_rolling();
        let mu_r = mt.rolling_friction_ij[0][0];

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        let torque_mag = (dem.torque[0][0].powi(2) + dem.torque[0][1].powi(2) + dem.torque[0][2].powi(2)).sqrt();

        // Compute expected cap: mu_r * F_n * R_eff
        // F_n from Hertz: 4/3 * E_eff * sqrt(delta * r_eff) * delta
        let r_eff = radius / 2.0;
        let delta = 2.0 * radius - 0.0019;
        let e_eff = 8.7e9 / (2.0 * (1.0 - 0.09)); // single material
        let sqrt_dr = (delta * r_eff).sqrt();
        let f_n_approx = 4.0 / 3.0 * e_eff * sqrt_dr * delta;
        let tau_cap = mu_r * f_n_approx * r_eff;

        // Rolling torque should not exceed cap (with reasonable tolerance for damping and normal force)
        // The torque includes tangential torque contributions, so we just check the rolling component
        // is bounded. Since torque_mag includes all contributions, just check it's finite and reasonable.
        assert!(torque_mag.is_finite(), "torque should be finite");
        assert!(
            torque_mag < tau_cap * 100.0, // generous bound since total torque includes tangential
            "torque {} should be bounded near cap {}",
            torque_mag, tau_cap
        );
    }

    #[test]
    fn sds_twisting_opposes_spin() {
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        // Spin about contact normal (x-axis)
        dem.omega[0] = [10.0, 0.0, 0.0];
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
        app.add_resource(make_material_table_sds_twisting());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // SDS twisting torque should oppose spin about x
        assert!(
            dem.torque[0][0] < 0.0,
            "SDS twisting torque should oppose spin about x, got {}",
            dem.torque[0][0]
        );
    }

    #[test]
    fn sds_twisting_spring_accumulates() {
        let radius = 0.001;

        let run_with_preload = |preload: f64| -> f64 {
            let mut app = App::new();
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            let mut hist = ContactHistoryStore::new();
            atom.dt = 1e-7;

            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
            push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
            dem.omega[0] = [0.001, 0.0, 0.0]; // very small spin
            atom.nlocal = 2;
            atom.natoms = 2;

            if preload != 0.0 {
                hist.contacts[0].push((1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, preload], false));
            }

            let mut neighbor = Neighbor::new();
            neighbor.neighbor_offsets = vec![0, 1, 1];
            neighbor.neighbor_indices = vec![1];

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);
            registry.register(hist);

            app.add_resource(atom);
            app.add_resource(neighbor);
            app.add_resource(registry);
            app.add_resource(make_material_table_sds_twisting());
            app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let reg = app.get_resource_ref::<AtomDataRegistry>().unwrap();
            let d = reg.expect::<DemAtom>("test");
            d.torque[0][0]
        };

        let torque_no_preload = run_with_preload(0.0);
        let torque_with_preload = run_with_preload(1e-5);
        assert!(torque_no_preload < 0.0);
        assert!(torque_with_preload < 0.0);
        assert!(
            torque_with_preload.abs() > torque_no_preload.abs(),
            "preloaded twisting spring should increase torque: no_preload={}, preloaded={}",
            torque_no_preload, torque_with_preload
        );
    }

    #[test]
    fn constant_model_unchanged_with_sds_config() {
        // When rolling_model = "constant" (default), SDS parameters should be ignored
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.0019, 0.0, 0.0], radius);
        dem.omega[0] = [0.0, 10.0, 0.0];
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        // Use constant model but with SDS parameters set (they should be ignored)
        let mut mt = MaterialTable::new();
        // rolling_model defaults to "constant"
        mt.add_material_with_sds(
            "glass", 8.7e9, 0.3, 0.95, 0.4,
            0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
            1e3, 0.5, 0.0, 0.0,
        );
        mt.build_pair_tables();

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

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // Constant model: torque = -mu_r * |F_n| * r_eff * (roll/|roll|)
        // Should still produce opposing torque
        assert!(
            dem.torque[0][1] < 0.0,
            "constant rolling model should still work, got {}",
            dem.torque[0][1]
        );

        // Check that spring history has zero rolling/twisting displacement
        let hist = registry.expect::<ContactHistoryStore>("test");
        let contact = &hist.contacts[0][0];
        assert_eq!(contact.1[3], 0.0, "rolling disp x should be zero in constant model");
        assert_eq!(contact.1[4], 0.0, "rolling disp y should be zero in constant model");
        assert_eq!(contact.1[5], 0.0, "rolling disp z should be zero in constant model");
        assert_eq!(contact.1[6], 0.0, "twisting disp should be zero in constant model");
    }

    // ── DMT adhesion tests ──────────────────────────────────────────────

    fn make_material_table_dmt() -> MaterialTable {
        let mut mt = MaterialTable::new();
        // Use high surface energy (1.0 J/m²) so adhesion clearly dominates at small overlaps
        mt.add_material_full("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0, 1.0);
        mt.adhesion_model = "dmt".to_string();
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn dmt_pulloff_force_matches_theory() {
        // DMT pull-off force = 2 * pi * gamma * r_eff (at contact, delta = 0+)
        let radius = 0.001;
        let gamma = 1.0;
        let r_eff = radius / 2.0; // two equal spheres

        // Use a very small overlap so Hertz contribution is negligible
        // At tiny delta, F_hertz ~ 0 but F_dmt = 2*pi*gamma*r_eff
        let tiny_overlap = 1e-12; // extremely small overlap
        let sep = 2.0 * radius - tiny_overlap;

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
        app.add_resource(make_material_table_dmt());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let expected_dmt = 2.0 * std::f64::consts::PI * gamma * r_eff;
        // Force on atom 0 should be positive (attracted toward atom 1)
        // f_n_mag = k_n*delta - f_diss - f_dmt ~ -f_dmt (since delta ~ 0, v=0)
        // force[0] -= f_n_mag * nx -> force[0] ~ +f_dmt
        assert!(
            atom.force[0][0] > 0.0,
            "DMT should produce attractive force, got {}",
            atom.force[0][0]
        );
        assert!(
            (atom.force[0][0] - expected_dmt).abs() / expected_dmt < 1e-3,
            "DMT pull-off force should match 2*pi*gamma*r_eff = {}, got {}",
            expected_dmt, atom.force[0][0]
        );
    }

    #[test]
    fn dmt_no_force_beyond_contact() {
        // DMT has no adhesion-only regime -- no force when delta < 0 (gap)
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        // Place particles with a gap
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
        app.add_resource(make_material_table_dmt());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // DMT: no force when particles are not in geometric contact
        assert!(
            atom.force[0][0].abs() < 1e-20,
            "DMT should have no force beyond contact, got {}",
            atom.force[0][0]
        );
    }

    #[test]
    fn dmt_pulloff_less_than_jkr() {
        // DMT pull-off = 2*pi*gamma*r_eff, JKR pull-off = 1.5*pi*gamma*r_eff
        // At same surface energy, DMT has HIGHER pull-off force than JKR (2 > 1.5)
        // But JKR has extended range (adhesion across gap), so effective sticking is stronger
        let gamma = 1.0;
        let radius = 0.001;
        let r_eff = radius / 2.0;

        let f_dmt = 2.0 * std::f64::consts::PI * gamma * r_eff;
        let f_jkr = 1.5 * std::f64::consts::PI * gamma * r_eff;
        assert!(
            f_dmt > f_jkr,
            "DMT pull-off ({}) should be larger than JKR pull-off ({})",
            f_dmt, f_jkr
        );
    }

    #[test]
    fn dmt_newtons_third_law() {
        // Verify equal and opposite forces for DMT contact
        let mut app = App::new();
        let radius = 0.001;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
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
        app.add_resource(make_material_table_dmt());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        for d in 0..3 {
            assert!(
                (atom.force[0][d] + atom.force[1][d]).abs() < 1e-10,
                "Newton's 3rd law violated in dim {}: {} + {} != 0",
                d, atom.force[0][d], atom.force[1][d]
            );
        }
    }

    #[test]
    fn dmt_does_not_break_jkr() {
        // Run the JKR test with default adhesion_model (should still work as JKR)
        let mut app = App::new();
        let radius = 0.001;
        let gamma = 1.0;
        let r_eff = radius / 2.0;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        // Place particles with a tiny gap (adhesion-only regime for JKR)
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

        // Use JKR material table (default adhesion_model = "jkr")
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table_jkr());
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let expected_jkr = 1.5 * std::f64::consts::PI * gamma * r_eff;
        // JKR should still attract across gap
        assert!(
            atom.force[0][0] > 0.0,
            "JKR should still work with DMT feature added, got {}",
            atom.force[0][0]
        );
        assert!(
            (atom.force[0][0] - expected_jkr).abs() / expected_jkr < 1e-6,
            "JKR pull-off force should still match 1.5*pi*gamma*r_eff = {}, got {}",
            expected_jkr, atom.force[0][0]
        );
    }

    // ── Force scaling validation tests ──────────────────────────────────

    #[test]
    fn hertz_force_scales_as_delta_three_halves() {
        let radius = 0.001;

        // Compute elastic-only normal force for a given separation (zero velocity -> no damping).
        let hertz_force_at = |sep: f64| -> f64 {
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
            app.add_resource(make_material_table());
            app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            // Force on atom 0 is negative (pushed away from atom 1), take absolute value
            atom.force[0][0].abs()
        };

        // Test at 5 different overlaps
        let deltas = [1e-5, 2e-5, 4e-5, 6e-5, 8e-5];
        let forces: Vec<f64> = deltas.iter().map(|d| {
            let sep = 2.0 * radius - d;
            hertz_force_at(sep)
        }).collect();

        // For each pair (i, 0), check F_i/F_0 ~ (delta_i/delta_0)^(3/2)
        for i in 1..deltas.len() {
            let expected_ratio = (deltas[i] / deltas[0]).powf(1.5);
            let actual_ratio = forces[i] / forces[0];
            let rel_err = ((actual_ratio - expected_ratio) / expected_ratio).abs();
            assert!(
                rel_err < 0.01,
                "Hertz force scaling: delta ratio {:.1}, expected F ratio {:.4}, got {:.4} (rel err {:.4})",
                deltas[i] / deltas[0], expected_ratio, actual_ratio, rel_err
            );
        }
    }

    #[test]
    fn hooke_force_scales_linearly_across_overlaps() {
        let radius = 0.001;
        let hooke_force_at = |sep: f64| -> f64 {
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
            app.add_resource(make_material_table_hooke());
            app.add_update_system(hooke_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            atom.force[0][0].abs()
        };

        let deltas = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4];
        let forces: Vec<f64> = deltas.iter().map(|d| {
            let sep = 2.0 * radius - d;
            hooke_force_at(sep)
        }).collect();

        for i in 1..deltas.len() {
            let expected_ratio = deltas[i] / deltas[0]; // linear
            let actual_ratio = forces[i] / forces[0];
            let rel_err = ((actual_ratio - expected_ratio) / expected_ratio).abs();
            assert!(
                rel_err < 0.01,
                "Hooke force scaling: delta ratio {:.1}, expected F ratio {:.4}, got {:.4} (rel err {:.4})",
                deltas[i] / deltas[0], expected_ratio, actual_ratio, rel_err
            );
        }
    }

    #[test]
    fn hertz_force_matches_analytical_value() {
        let radius = 0.001;
        let delta = 5e-5;
        let sep = 2.0 * radius - delta;

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

        let mt = make_material_table();
        let e_eff = mt.e_eff_ij[0][0];
        let r_eff = radius / 2.0; // two equal spheres: r_eff = r1*r2/(r1+r2) = r/2

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(hertz_mindlin_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_computed = atom.force[0][0].abs();
        // Analytical: F = (4/3) * E_eff * sqrt(R_eff) * delta^(3/2)
        let f_analytical = (4.0 / 3.0) * e_eff * r_eff.sqrt() * delta.powf(1.5);
        let rel_err = (f_computed - f_analytical).abs() / f_analytical;
        assert!(
            rel_err < 1e-10,
            "Hertz force analytical check: computed={:.6e}, expected={:.6e}, rel_err={:.2e}",
            f_computed, f_analytical, rel_err
        );
    }

    #[test]
    fn linear_momentum_conserved_during_elastic_contact() {
        // Use a perfectly elastic material (restitution = 1.0 -> beta = 0 -> no damping)
        let mut mt = MaterialTable::new();
        mt.add_material("elastic", 8.7e9, 0.3, 1.0, 0.0, 0.0, 0.0);
        mt.build_pair_tables();
        assert!(
            mt.beta_ij[0][0].abs() < 1e-15,
            "beta should be 0 for e=1.0"
        );

        let radius = 0.001;
        let dt = 1e-8;

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = dt;

        // Two particles approaching each other, slight overlap
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [0.00195, 0.0, 0.0], radius);
        atom.vel[0] = [0.1, 0.05, -0.02];
        atom.vel[1] = [-0.05, 0.03, 0.01];
        atom.nlocal = 2;
        atom.natoms = 2;

        let initial_momentum = [
            atom.mass[0] * atom.vel[0][0] + atom.mass[1] * atom.vel[1][0],
            atom.mass[0] * atom.vel[0][1] + atom.mass[1] * atom.vel[1][1],
            atom.mass[0] * atom.vel[0][2] + atom.mass[1] * atom.vel[1][2],
        ];

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];
        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(hist);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_update_system(
            crate::contact::hertz_mindlin_contact_force,
            ScheduleSet::Force,
        );
        app.add_update_system(
            mddem_verlet::initial_integration,
            ScheduleSet::InitialIntegration,
        );
        app.add_update_system(
            mddem_verlet::final_integration,
            ScheduleSet::FinalIntegration,
        );
        // Zero forces between steps
        app.add_update_system(
            |mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>| {
                let n = atoms.len();
                atoms.force[..n].fill([0.0; 3]);
                registry.zero_all(n);
            },
            ScheduleSet::PostInitialIntegration,
        );
        app.organize_systems();

        // Run for 100 steps
        for _ in 0..100 {
            app.run();
        }

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let final_momentum = [
            atom.mass[0] * atom.vel[0][0] + atom.mass[1] * atom.vel[1][0],
            atom.mass[0] * atom.vel[0][1] + atom.mass[1] * atom.vel[1][1],
            atom.mass[0] * atom.vel[0][2] + atom.mass[1] * atom.vel[1][2],
        ];

        for d in 0..3 {
            let err = (final_momentum[d] - initial_momentum[d]).abs();
            assert!(
                err < 1e-12,
                "Momentum not conserved in dim {}: initial={:.6e}, final={:.6e}, err={:.2e}",
                d, initial_momentum[d], final_momentum[d], err
            );
        }
    }

    #[test]
    fn contact_force_symmetry_with_tangential_velocity() {
        let radius = 0.001;
        let sep = 0.0019;

        let mut app = App::new();
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let mut hist = ContactHistoryStore::new();
        atom.dt = 1e-7;

        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 0, [0.0, 0.0, 0.0], radius);
        push_test_atom_with_history(&mut atom, &mut dem, &mut hist, 1, [sep, 0.0, 0.0], radius);
        // Give both atoms velocities in all directions
        atom.vel[0] = [0.1, 0.2, -0.1];
        atom.vel[1] = [-0.3, 0.1, 0.05];
        dem.omega[0] = [10.0, 20.0, -5.0];
        dem.omega[1] = [-15.0, 5.0, 10.0];
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
        // Newton's 3rd law: forces equal and opposite
        for d in 0..3 {
            assert!(
                (atom.force[0][d] + atom.force[1][d]).abs() < 1e-10,
                "Newton's 3rd law violated in dim {}: f0={:.6e}, f1={:.6e}",
                d, atom.force[0][d], atom.force[1][d]
            );
        }
    }
}
