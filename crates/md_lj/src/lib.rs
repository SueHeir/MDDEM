use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

fn default_epsilon() -> f64 {
    1.0
}
fn default_sigma() -> f64 {
    1.0
}
fn default_cutoff() -> f64 {
    2.5
}

#[derive(Deserialize, Clone)]
pub struct LJConfig {
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default = "default_cutoff")]
    pub cutoff: f64,
}

impl Default for LJConfig {
    fn default() -> Self {
        LJConfig {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
        }
    }
}

// ── Resources ───────────────────────────────────────────────────────────────

pub struct VirialAccumulator {
    pub virial_sum: f64,
    pub active: bool,
}

impl Default for VirialAccumulator {
    fn default() -> Self {
        VirialAccumulator {
            virial_sum: 0.0,
            active: true,
        }
    }
}

pub struct LJTailCorrections {
    pub energy_tail: f64,
    pub pressure_tail: f64,
}

impl Default for LJTailCorrections {
    fn default() -> Self {
        LJTailCorrections {
            energy_tail: 0.0,
            pressure_tail: 0.0,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

pub struct LJForcePlugin;

impl Plugin for LJForcePlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[lj]
epsilon = 1.0    # well depth (reduced units)
sigma = 1.0      # length scale
cutoff = 2.5     # in sigma units"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<LJConfig>(app, "lj");

        app.add_resource(VirialAccumulator::default())
            .add_resource(LJTailCorrections::default())
            .add_setup_system(setup_lj_tails, ScheduleSetupSet::PostSetup)
            .add_update_system(zero_virial, ScheduleSet::PostInitialIntegration)
            .add_update_system(lj_force.label("lj"), ScheduleSet::Force);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

pub fn setup_lj_tails(
    lj: Res<LJConfig>,
    atoms: Res<Atom>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut tails: ResMut<LJTailCorrections>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

    let n = comm.all_reduce_sum_f64(atoms.nlocal as f64);
    let v = domain.volume;
    let rho = n / v;
    let rc = lj.cutoff * lj.sigma;
    let sigma3 = lj.sigma.powi(3);
    let rc3 = rc.powi(3);
    let rc9 = rc3.powi(3);
    let sigma6 = sigma3 * sigma3;
    let sigma9 = sigma6 * sigma3;
    // Standard LJ tail corrections
    // E_tail = (8/3) * pi * N * rho * eps * sigma^3 * [ (1/3)(sigma/rc)^9 - (sigma/rc)^3 ]
    tails.energy_tail =
        (8.0 / 3.0) * PI * n * rho * lj.epsilon * sigma3 * (sigma9 / (3.0 * rc9) - sigma3 / rc3);

    // P_tail = (16/3) * pi * rho^2 * eps * sigma^3 * [ (2/3)(sigma/rc)^9 - (sigma/rc)^3 ]
    tails.pressure_tail = (16.0 / 3.0) * PI * rho * rho * lj.epsilon * sigma3
        * (2.0 * sigma9 / (3.0 * rc9) - sigma3 / rc3);

    if comm.rank() == 0 {
        println!(
            "LJ: eps={}, sigma={}, rc={}, rho={:.4}",
            lj.epsilon, lj.sigma, lj.cutoff, rho
        );
        println!(
            "LJ tail corrections: E_tail={:.6}, P_tail={:.6}",
            tails.energy_tail, tails.pressure_tail
        );
    }
}

pub fn zero_virial(mut virial: ResMut<VirialAccumulator>) {
    virial.virial_sum = 0.0;
}

/// Kept for tests — production code uses the inlined version in `lj_force`.
#[cfg(test)]
#[inline(always)]
fn lj_pair(
    atoms: &mut Atom,
    i: usize,
    j: usize,
    sigma2: f64,
    cutoff2: f64,
    eps24: f64,
    virial: &mut VirialAccumulator,
) {
    let dx = atoms.pos_x[j] - atoms.pos_x[i];
    let dy = atoms.pos_y[j] - atoms.pos_y[i];
    let dz = atoms.pos_z[j] - atoms.pos_z[i];
    let r2 = dx * dx + dy * dy + dz * dz;

    if r2 >= cutoff2 {
        return;
    }

    let sr2 = sigma2 / r2;
    let sr6 = sr2 * sr2 * sr2;
    let f_over_r = eps24 / r2 * (2.0 * sr6 * sr6 - sr6);

    virial.virial_sum += f_over_r * r2;

    let fx = f_over_r * dx;
    let fy = f_over_r * dy;
    let fz = f_over_r * dz;

    atoms.force_x[i] -= fx;
    atoms.force_y[i] -= fy;
    atoms.force_z[i] -= fz;
    atoms.force_x[j] += fx;
    atoms.force_y[j] += fy;
    atoms.force_z[j] += fz;
}

pub fn lj_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    lj: Res<LJConfig>,
    _domain: Res<Domain>,
    mut virial: ResMut<VirialAccumulator>,
) {
    let sigma2 = lj.sigma * lj.sigma;
    let cutoff2 = (lj.cutoff * lj.sigma).powi(2);
    let eps24 = 24.0 * lj.epsilon;

    let nlocal = atoms.nlocal as usize;
    let compute_virial = virial.active;
    let mut virial_sum = 0.0f64;

    let pos_x = atoms.pos_x.as_ptr();
    let pos_y = atoms.pos_y.as_ptr();
    let pos_z = atoms.pos_z.as_ptr();
    let force_x = atoms.force_x.as_mut_ptr();
    let force_y = atoms.force_y.as_mut_ptr();
    let force_z = atoms.force_z.as_mut_ptr();
    let offsets = neighbor.neighbor_offsets.as_ptr();
    let indices = neighbor.neighbor_indices.as_ptr();

    // Safety invariants:
    // - i ranges over 0..nlocal, all valid atom indices
    // - neighbor_offsets has length nlocal+1, so offsets[i] and offsets[i+1] are valid
    // - k ranges over offsets[i]..offsets[i+1], bounded by neighbor_indices.len()
    // - j comes from neighbor_indices which contains valid atom indices (0..total)
    // - pos_x/y/z and force_x/y/z have length >= total (nlocal + nghost)
    for i in 0..nlocal {
        unsafe {
            let xi = *pos_x.add(i);
            let yi = *pos_y.add(i);
            let zi = *pos_z.add(i);
            let mut fix = 0.0f64;
            let mut fiy = 0.0f64;
            let mut fiz = 0.0f64;
            let start = *offsets.add(i) as usize;
            let end = *offsets.add(i + 1) as usize;
            for k in start..end {
                let j = *indices.add(k) as usize;
                let dx = *pos_x.add(j) - xi;
                let dy = *pos_y.add(j) - yi;
                let dz = *pos_z.add(j) - zi;
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 {
                    continue;
                }
                let sr2 = sigma2 / r2;
                let sr6 = sr2 * sr2 * sr2;
                let f_over_r = eps24 / r2 * (2.0 * sr6 * sr6 - sr6);
                if compute_virial {
                    virial_sum += f_over_r * r2;
                }
                let fx = f_over_r * dx;
                let fy = f_over_r * dy;
                let fz = f_over_r * dz;
                fix -= fx;
                fiy -= fy;
                fiz -= fz;
                *force_x.add(j) += fx;
                *force_y.add(j) += fy;
                *force_z.add(j) += fz;
            }
            *force_x.add(i) += fix;
            *force_y.add(i) += fiy;
            *force_z.add(i) += fiz;
        }
    }

    if compute_virial {
        virial.virial_sum = virial_sum;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    fn push_atom(atom: &mut Atom, tag: u32, x: f64, y: f64, z: f64, mass: f64) {
        use nalgebra::{Quaternion, UnitQuaternion};
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(x);
        atom.pos_y.push(y);
        atom.pos_z.push(z);
        atom.vel_x.push(0.0);
        atom.vel_y.push(0.0);
        atom.vel_z.push(0.0);
        atom.force_x.push(0.0);
        atom.force_y.push(0.0);
        atom.force_z.push(0.0);
        atom.torque_x.push(0.0);
        atom.torque_y.push(0.0);
        atom.torque_z.push(0.0);
        atom.omega_x.push(0.0);
        atom.omega_y.push(0.0);
        atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0);
        atom.ang_mom_y.push(0.0);
        atom.ang_mom_z.push(0.0);
        atom.mass.push(mass);
        atom.skin.push(0.5);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion
            .push(UnitQuaternion::from_quaternion(Quaternion::identity()));
    }

    fn make_two_atom_app(distance: f64) -> App {
        let mut app = App::new();

        let lj_config = LJConfig {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
        };
        app.add_resource(lj_config);
        app.add_resource(VirialAccumulator::default());
        app.add_resource(Domain::default());

        let mut atom = Atom::new();
        push_atom(&mut atom, 0, 0.0, 0.0, 0.0, 1.0);
        push_atom(&mut atom, 1, distance, 0.0, 0.0, 1.0);
        atom.nlocal = 2;
        atom.natoms = 2;
        app.add_resource(atom);

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_list.push((0, 1));
        // CSR format: offsets for 2 local atoms
        neighbor.neighbor_offsets = vec![0, 1, 1]; // atom 0 has 1 neighbor, atom 1 has 0
        neighbor.neighbor_indices = vec![1];        // atom 0's neighbor is atom 1
        app.add_resource(neighbor);

        app.add_update_system(zero_virial, ScheduleSet::PostInitialIntegration);
        app.add_update_system(lj_force, ScheduleSet::Force);
        app.organize_systems();
        app
    }

    #[test]
    fn lj_repulsive_at_close_range() {
        let mut app = make_two_atom_app(0.9);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force_x[0] < 0.0,
            "atom 0 should be pushed in -x: got {}",
            atom.force_x[0]
        );
        assert!(
            atom.force_x[1] > 0.0,
            "atom 1 should be pushed in +x: got {}",
            atom.force_x[1]
        );
    }

    #[test]
    fn lj_attractive_at_medium_range() {
        let mut app = make_two_atom_app(1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force_x[0] > 0.0,
            "atom 0 should be pulled in +x: got {}",
            atom.force_x[0]
        );
        assert!(
            atom.force_x[1] < 0.0,
            "atom 1 should be pulled in -x: got {}",
            atom.force_x[1]
        );
    }

    #[test]
    fn lj_zero_beyond_cutoff() {
        let mut app = make_two_atom_app(3.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force_x[0].abs() < 1e-15,
            "force should be zero beyond cutoff"
        );
        assert!(
            atom.force_x[1].abs() < 1e-15,
            "force should be zero beyond cutoff"
        );
    }

    #[test]
    fn lj_newtons_third_law() {
        let mut app = make_two_atom_app(1.2);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force_x[0] + atom.force_x[1]).abs() < 1e-10,
            "Newton's 3rd law violated in x"
        );
        assert!(
            (atom.force_y[0] + atom.force_y[1]).abs() < 1e-10,
            "Newton's 3rd law violated in y"
        );
        assert!(
            (atom.force_z[0] + atom.force_z[1]).abs() < 1e-10,
            "Newton's 3rd law violated in z"
        );
    }

    #[test]
    fn virial_positive_at_close_range() {
        let mut app = make_two_atom_app(0.9);
        app.run();
        let virial = app.get_resource_ref::<VirialAccumulator>().unwrap();
        assert!(
            virial.virial_sum > 0.0,
            "virial should be positive at close range"
        );
    }
}
