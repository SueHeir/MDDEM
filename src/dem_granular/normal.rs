use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use crate::{
    dem_atom::DemAtom,
    mddem_atom::{Atom, AtomDataRegistry},
    mddem_neighbor::Neighbor,
};

// √(5/3) — appears in the viscoelastic damping formula
const SQRT_5_3: f64 = 0.9128709291752768;

pub struct HertzNormalForcePlugin;

impl Plugin for HertzNormalForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(clear_forces_and_torques, ScheduleSet::PreForce)
            .add_update_system(
                hertz_normal_force.label("hertz_normal"),
                ScheduleSet::Force,
            );
    }
}

/// Zero force and torque on all local atoms at the start of each force evaluation step.
/// Ghost atom forces are implicitly reset because ghosts are destroyed and recreated
/// fresh every step by `borders()`.
pub fn clear_forces_and_torques(mut atoms: ResMut<Atom>) {
    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        atoms.force[i] = Vector3::zeros();
        atoms.torque[i] = Vector3::zeros();
    }
}

/// Hertz normal contact force with viscoelastic damping (no tangential component).
///
/// For each overlapping pair:
/// - k_n = 4/3 · E_eff · √(δ · r_eff)
/// - s_n = 2 · E_eff · √(δ · r_eff)
/// - F_n = k_n · δ  (Hertz spring)
/// - F_diss = 2β · √(5/3) · √(s_n · m_r) · v_n  (viscoelastic damping)
/// - F_net = max(F_n − F_diss, 0)  (no-tension clamp)
///
/// Cross-boundary pairs (where either atom is a ghost) are halved to avoid
/// double-counting: the pair is computed on both ranks that share the boundary.
pub fn hertz_normal_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
) {
    let dem = registry.get::<DemAtom>().unwrap();

    for &(i, j) in &neighbor.neighbor_list {
        let p1 = atoms.pos[i];
        let p2 = atoms.pos[j];

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let diff = p2 - p1;
        let distance = diff.norm();

        if distance >= r1 + r2 {
            continue;
        }

        if distance == 0.0 {
            eprintln!("warning: zero separation between tags {} {}", atoms.tag[i], atoms.tag[j]);
            continue;
        }

        if distance / (r1 + r2) < 0.90 {
            eprintln!("large overlap: tags {} {} ratio {:.3}", atoms.tag[i], atoms.tag[j], distance / (r1 + r2));
            panic!("excessive overlap detected — check timestep or initial configuration");
        }

        atoms.is_collision[i] = true;
        atoms.is_collision[j] = true;

        let n = diff / distance;          // unit normal from i toward j
        let delta = (r1 + r2) - distance; // overlap magnitude

        let r_eff = (r1 * r2) / (r1 + r2);
        let e_eff = 1.0
            / ((1.0 - dem.poisson_ratio[i].powi(2)) / dem.youngs_mod[i]
                + (1.0 - dem.poisson_ratio[j].powi(2)) / dem.youngs_mod[j]);

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;

        let m_r = (atoms.mass[i] * atoms.mass[j]) / (atoms.mass[i] + atoms.mass[j]);

        let v_rel = atoms.velocity[j] - atoms.velocity[i];
        let v_n = v_rel.dot(&n); // positive = approaching

        let f_spring = k_n * delta;
        let f_diss = 2.0 * dem.beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
        let f_net = (f_spring - f_diss).max(0.0);

        let force = f_net * n;
        let scale = if atoms.is_ghost[i] || atoms.is_ghost[j] { 0.5 } else { 1.0 };

        atoms.force[i] -= force * scale;
        atoms.force[j] += force * scale;
    }
}
