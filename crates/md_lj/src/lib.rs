//! Lennard-Jones 12-6 pair force with virial accumulator and tail corrections.

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, VirialStressPlugin};
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
#[serde(deny_unknown_fields)]
/// TOML `[lj]` — Lennard-Jones potential parameters.
pub struct LJConfig {
    /// Well depth (energy units).
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Particle diameter (length units).
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    /// Cutoff distance in units of sigma.
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

/// Long-range tail corrections for energy and pressure beyond the LJ cutoff.
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

/// Registers LJ 12-6 pair force, virial accumulator, and tail correction systems.
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

        app.add_plugins(VirialStressPlugin);
        app.add_resource(LJTailCorrections::default())
            .add_setup_system(setup_lj_tails, ScheduleSetupSet::PostSetup)
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

pub fn lj_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    lj: Res<LJConfig>,
    _domain: Res<Domain>,
    virial: Option<ResMut<mddem_core::VirialStress>>,
) {
    let cutoff2 = (lj.cutoff * lj.sigma).powi(2);
    // Precompute LAMMPS-style constants: lj1 = 48*eps*sigma^12, lj2 = 24*eps*sigma^6
    // Inner loop then needs only one division (r2inv = 1/r2) and multiplies.
    let sigma6 = (lj.sigma * lj.sigma).powi(3);
    let lj1 = 48.0 * lj.epsilon * sigma6 * sigma6;
    let lj2 = 24.0 * lj.epsilon * sigma6;

    let nlocal = atoms.nlocal as usize;

    // SAFETY: i < nlocal <= atoms.len(), neighbor_offsets has nlocal+1 entries,
    // neighbor_indices[k] < atoms.len() (from CSR construction), j < atoms.len().
    let pos_ptr = atoms.pos.as_ptr();
    let force_ptr = atoms.force.as_mut_ptr();
    // Cache CSR pointers to prevent alias-reload in inner loops.
    // Without this, the compiler reloads the Vec data pointer every iteration
    // because it can't prove the force writes don't alias the Neighbor struct.
    let offsets_ptr = neighbor.neighbor_offsets.as_ptr();
    let indices_ptr = neighbor.neighbor_indices.as_ptr();

    let virial_active = virial.as_ref().map_or(false, |v| v.active);

    if virial_active {
        // Virial path: accumulate virial stress tensor alongside forces.
        let mut vxx = 0.0f64;
        let mut vyy = 0.0f64;
        let mut vzz = 0.0f64;
        let mut vxy = 0.0f64;
        let mut vxz = 0.0f64;
        let mut vyz = 0.0f64;

        for i in 0..nlocal {
            let pi = unsafe { *pos_ptr.add(i) };
            let mut fi = unsafe { *force_ptr.add(i) };
            let start = unsafe { *offsets_ptr.add(i) } as usize;
            let end = unsafe { *offsets_ptr.add(i + 1) } as usize;

            for k in start..end {
                let j = unsafe { *indices_ptr.add(k) } as usize;
                let pj = unsafe { *pos_ptr.add(j) };
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                if r2 >= cutoff2 {
                    continue;
                }

                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * lj1.mul_add(r6inv, -lj2);

                let fx = -fpair * dx;
                let fy = -fpair * dy;
                let fz = -fpair * dz;
                vxx += dx * fx;
                vyy += dy * fy;
                vzz += dz * fz;
                vxy += dx * fy;
                vxz += dx * fz;
                vyz += dy * fz;

                fi[0] += fx;
                fi[1] += fy;
                fi[2] += fz;
                let fj = unsafe { &mut *force_ptr.add(j) };
                fj[0] -= fx;
                fj[1] -= fy;
                fj[2] -= fz;
            }
            unsafe { *force_ptr.add(i) = fi };
        }

        if let Some(mut virial) = virial {
            virial.xx += vxx;
            virial.yy += vyy;
            virial.zz += vzz;
            virial.xy += vxy;
            virial.xz += vxz;
            virial.yz += vyz;
        }
    } else {
        // Fast path: no virial accumulation, enables FMA for force updates.
        for i in 0..nlocal {
            let pi = unsafe { *pos_ptr.add(i) };
            let mut fi = unsafe { *force_ptr.add(i) };
            let start = unsafe { *offsets_ptr.add(i) } as usize;
            let end = unsafe { *offsets_ptr.add(i + 1) } as usize;

            for k in start..end {
                let j = unsafe { *indices_ptr.add(k) } as usize;
                let pj = unsafe { *pos_ptr.add(j) };
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                if r2 >= cutoff2 {
                    continue;
                }

                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * lj1.mul_add(r6inv, -lj2);

                fi[0] = (-fpair).mul_add(dx, fi[0]);
                fi[1] = (-fpair).mul_add(dy, fi[1]);
                fi[2] = (-fpair).mul_add(dz, fi[2]);
                let fj = unsafe { &mut *force_ptr.add(j) };
                fj[0] = fpair.mul_add(dx, fj[0]);
                fj[1] = fpair.mul_add(dy, fj[1]);
                fj[2] = fpair.mul_add(dz, fj[2]);
            }
            unsafe { *force_ptr.add(i) = fi };
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    fn push_atom(atom: &mut Atom, tag: u32, x: f64, y: f64, z: f64, mass: f64) {
        use nalgebra::Vector3;
        atom.push_test_atom(tag, Vector3::new(x, y, z), 0.5, mass);
    }

    fn make_two_atom_app(distance: f64) -> App {
        let mut app = App::new();

        let lj_config = LJConfig {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
        };
        app.add_resource(lj_config);
        app.add_resource(mddem_core::VirialStress::default());
        app.add_resource(mddem_core::RunState::default());
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

        app.add_update_system(
            mddem_core::virial::zero_virial_stress,
            ScheduleSet::PreForce,
        );
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
            atom.force[0][0] < 0.0,
            "atom 0 should be pushed in -x: got {}",
            atom.force[0][0]
        );
        assert!(
            atom.force[1][0] > 0.0,
            "atom 1 should be pushed in +x: got {}",
            atom.force[1][0]
        );
    }

    #[test]
    fn lj_attractive_at_medium_range() {
        let mut app = make_two_atom_app(1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][0] > 0.0,
            "atom 0 should be pulled in +x: got {}",
            atom.force[0][0]
        );
        assert!(
            atom.force[1][0] < 0.0,
            "atom 1 should be pulled in -x: got {}",
            atom.force[1][0]
        );
    }

    #[test]
    fn lj_zero_beyond_cutoff() {
        let mut app = make_two_atom_app(3.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][0].abs() < 1e-15,
            "force should be zero beyond cutoff"
        );
        assert!(
            atom.force[1][0].abs() < 1e-15,
            "force should be zero beyond cutoff"
        );
    }

    #[test]
    fn lj_newtons_third_law() {
        let mut app = make_two_atom_app(1.2);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "Newton's 3rd law violated in x"
        );
        assert!(
            (atom.force[0][1] + atom.force[1][1]).abs() < 1e-10,
            "Newton's 3rd law violated in y"
        );
        assert!(
            (atom.force[0][2] + atom.force[1][2]).abs() < 1e-10,
            "Newton's 3rd law violated in z"
        );
    }

    #[test]
    fn virial_negative_trace_at_close_range() {
        let mut app = make_two_atom_app(0.9);
        app.run();
        let virial = app
            .get_resource_ref::<mddem_core::VirialStress>()
            .unwrap();
        assert!(
            virial.trace() < 0.0,
            "virial trace should be negative at close range (repulsion)"
        );
    }
}
