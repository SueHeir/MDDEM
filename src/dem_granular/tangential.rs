use std::collections::{HashMap, HashSet};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use crate::{
    dem_atom::DemAtom,
    mddem_atom::{Atom, AtomDataRegistry},
    mddem_input::Input,
    mddem_neighbor::Neighbor,
};

// √(5/3) — damping coefficient constant shared with Hertz normal model
const SQRT_5_3: f64 = 0.9128709291752768;

// ─── Resource ─────────────────────────────────────────────────────────────────

pub struct MindlinTangential {
    pub friction_coefficient: f64, // μ (Coulomb friction cap), default 0.4
}

// ─── Plugin ───────────────────────────────────────────────────────────────────

pub struct MindlinTangentialForcePlugin;

impl Plugin for MindlinTangentialForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(MindlinTangential { friction_coefficient: 0.4 })
            .add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_update_system(
                mindlin_tangential_force
                    .label("mindlin_tangential")
                    .after("hertz_normal"),
                ScheduleSet::Force,
            );
    }
}

pub fn read_input(
    input: Res<Input>,
    scheduler_manager: Res<SchedulerManager>,
    mut tangential: ResMut<MindlinTangential>,
) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();
        if values.len() >= 2 && values[0] == "friction_coefficient" {
            tangential.friction_coefficient = values[1].parse::<f64>().unwrap();
        }
    }
}

// ─── Mindlin tangential force with spring history ─────────────────────────────

/// Mindlin spring-history tangential contact force with Coulomb friction cap.
///
/// Spring history is stored per contact pair in `Local<HashMap<(u32,u32), Vector3<f64>>>`.
/// The key is `(min_tag, max_tag)` so the same pair always maps to the same entry.
/// The stored vector `δ_t` points from min-tag particle toward max-tag particle.
///
/// Per contact step:
/// 1. Retrieve spring displacement δ_t (zero if new contact)
/// 2. Project out any normal drift accumulated since last step
/// 3. Increment by tangential relative displacement: δ_t += v_t · dt
/// 4. Coulomb slip: if k_t|δ_t| > μ·F_n, truncate δ_t to the slip limit
/// 5. Force: F_t = −k_t·δ_t + γ_t·v_t
/// 6. Torque: τ = r_contact × F_t (contact point at r_i·n̂ from center)
///
/// Dead contacts are pruned from history before the loop so rejoining particles
/// start with a fresh spring.
pub fn mindlin_tangential_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    tangential: Res<MindlinTangential>,
    mut history: Local<HashMap<(u32, u32), Vector3<f64>>>,
) {
    let dem = registry.get::<DemAtom>().unwrap();
    let mu = tangential.friction_coefficient;
    let dt = atoms.dt;

    // ── Build active contact set and prune stale history ──────────────────────
    let mut active: HashSet<(u32, u32)> = HashSet::new();
    for &(i, j) in &neighbor.neighbor_list {
        let d = (atoms.pos[j] - atoms.pos[i]).norm();
        if d < dem.radius[i] + dem.radius[j] {
            let ti = atoms.tag[i].min(atoms.tag[j]);
            let tj = atoms.tag[i].max(atoms.tag[j]);
            active.insert((ti, tj));
        }
    }
    history.retain(|k, _| active.contains(k));

    // ── Contact loop ──────────────────────────────────────────────────────────
    for &(i, j) in &neighbor.neighbor_list {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let diff = atoms.pos[j] - atoms.pos[i];
        let distance = diff.norm();

        if distance >= r1 + r2 || distance == 0.0 {
            continue;
        }

        let n = diff / distance;          // unit normal i→j
        let delta = (r1 + r2) - distance; // overlap

        // ── Material parameters ───────────────────────────────────────────────
        let r_eff = (r1 * r2) / (r1 + r2);
        let e_eff = 1.0
            / ((1.0 - dem.poisson_ratio[i].powi(2)) / dem.youngs_mod[i]
                + (1.0 - dem.poisson_ratio[j].powi(2)) / dem.youngs_mod[j]);
        let g_eff = 1.0
            / (2.0 * (2.0 - dem.poisson_ratio[i]) * (1.0 + dem.poisson_ratio[i]) / dem.youngs_mod[i]
                + 2.0 * (2.0 - dem.poisson_ratio[j]) * (1.0 + dem.poisson_ratio[j]) / dem.youngs_mod[j]);

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;
        let k_t = 8.0 * g_eff * sqrt_dr;

        let m_r = (atoms.mass[i] * atoms.mass[j]) / (atoms.mass[i] + atoms.mass[j]);

        // ── Normal force magnitude (for Coulomb cap) ──────────────────────────
        // Contact-point velocity includes rotational contribution: v + ω × r_contact
        let v_contact_i = atoms.velocity[i] + atoms.omega[i].cross(&(r1 * n));
        let v_contact_j = atoms.velocity[j] + atoms.omega[j].cross(&(-r2 * n));
        let v_rel = v_contact_j - v_contact_i;
        let v_n_scalar = v_rel.dot(&n);
        let v_t = v_rel - v_n_scalar * n; // tangential relative velocity

        let f_diss_n = 2.0 * dem.beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n_scalar;
        let f_n_mag = (k_n * delta - f_diss_n).max(0.0);

        // ── Spring history retrieval with tag-ordering convention ─────────────
        let tag_i = atoms.tag[i];
        let tag_j = atoms.tag[j];
        let canonical_key = (tag_i.min(tag_j), tag_i.max(tag_j));
        // If tag[i] > tag[j], the stored vector points from j toward i, so negate.
        let sign: f64 = if tag_i < tag_j { 1.0 } else { -1.0 };

        let stored = history.entry(canonical_key).or_insert_with(Vector3::zeros);
        let mut s = sign * *stored; // spring vector in i→j frame

        // ── Project out normal drift (contact normal rotates between steps) ───
        s -= s.dot(&n) * n;

        // ── Increment by tangential displacement this step ────────────────────
        s += v_t * dt;

        // ── Coulomb slip truncation ───────────────────────────────────────────
        let f_t_spring_mag = k_t * s.norm();
        let f_t_max = mu * f_n_mag;
        if f_t_spring_mag > f_t_max && f_t_spring_mag > 1e-30 {
            s = s * (f_t_max / f_t_spring_mag);
        }

        // ── Tangential force: spring + viscous damping ────────────────────────
        let gamma_t = -2.0 * SQRT_5_3 * dem.beta * (k_t * m_r).sqrt();
        let mut f_t = k_t * s - gamma_t * v_t;
        let f_t_mag = f_t.norm();
        if f_t_mag > f_t_max && f_t_mag > 1e-30 {
            f_t = f_t * (f_t_max / f_t_mag);
        }

        // ── Torques (contact point at r·n̂ from sphere center) ─────────────────
        let torque_i = (r1 * n).cross(&f_t);
        let torque_j = (-r2 * n).cross(&(-f_t));

        // ── Apply forces and torques ──────────────────────────────────────────
        let scale = if atoms.is_ghost[i] || atoms.is_ghost[j] { 0.5 } else { 1.0 };
        atoms.force[i] += f_t * scale;
        atoms.force[j] -= f_t * scale;
        atoms.torque[i] += torque_i * scale;
        atoms.torque[j] += torque_j * scale;

        // ── Write updated spring back in canonical frame ───────────────────────
        *stored = sign * s;
    }
}
