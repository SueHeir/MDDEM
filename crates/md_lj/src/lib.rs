//! Lennard-Jones 12-6 pair force with virial accumulator and tail corrections.

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, MixingRule, PairCoeffTable, VirialStressPlugin};
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
    /// Well depth (energy units). Used as default for single-type mode.
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Particle diameter (length units). Used as default for single-type mode.
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    /// Cutoff distance in units of sigma.
    #[serde(default = "default_cutoff")]
    pub cutoff: f64,
    /// Mixing rule for multi-type: "geometric" or "arithmetic".
    #[serde(default)]
    pub mixing: Option<String>,
    /// Per-type LJ parameters. If present, enables multi-type mode.
    #[serde(default)]
    pub types: Option<Vec<LJTypeConfig>>,
    /// Explicit pair coefficient overrides.
    #[serde(default)]
    pub pair_coeffs: Option<Vec<LJPairOverride>>,
}

impl Default for LJConfig {
    fn default() -> Self {
        LJConfig {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
            mixing: None,
            types: None,
            pair_coeffs: None,
        }
    }
}

/// Per-type LJ parameters from `[[lj.types]]`.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LJTypeConfig {
    pub epsilon: f64,
    pub sigma: f64,
}

/// Explicit pair override from `[[lj.pair_coeffs]]`.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LJPairOverride {
    pub types: [usize; 2],
    pub epsilon: f64,
    pub sigma: f64,
    #[serde(default)]
    pub cutoff: Option<f64>,
}

// ── Precomputed pair coefficients ───────────────────────────────────────────

/// Precomputed LJ pair coefficients for fast inner-loop evaluation.
#[derive(Clone)]
pub struct LJPairCoeffs {
    /// 48 * eps * sigma^12
    pub lj1: f64,
    /// 24 * eps * sigma^6
    pub lj2: f64,
    /// cutoff^2 (in absolute length units)
    pub cutoff2: f64,
    pub epsilon: f64,
    pub sigma: f64,
}

impl Default for LJPairCoeffs {
    fn default() -> Self {
        LJPairCoeffs {
            lj1: 0.0,
            lj2: 0.0,
            cutoff2: 0.0,
            epsilon: 0.0,
            sigma: 0.0,
        }
    }
}

impl LJPairCoeffs {
    pub fn from_params(epsilon: f64, sigma: f64, cutoff_sigma: f64) -> Self {
        let sigma6 = (sigma * sigma).powi(3);
        LJPairCoeffs {
            lj1: 48.0 * epsilon * sigma6 * sigma6,
            lj2: 24.0 * epsilon * sigma6,
            cutoff2: (cutoff_sigma * sigma).powi(2),
            epsilon,
            sigma,
        }
    }
}

/// Wrapper resource for the LJ pair coefficient table.
pub struct LJPairTable(pub PairCoeffTable<LJPairCoeffs>);

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
    fn dependencies(&self) -> Vec<&str> {
        vec!["NeighborPlugin"]
    }

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
        // Add a default 1x1 pair table; build_lj_pair_table will replace it.
        let mut default_table = PairCoeffTable::new(1, LJPairCoeffs::default());
        default_table.set(0, 0, LJPairCoeffs::from_params(1.0, 1.0, 2.5));
        app.add_resource(LJPairTable(default_table));
        app.add_resource(LJTailCorrections::default())
            .add_setup_system(
                build_lj_pair_table,
                ScheduleSetupSet::Setup,
            )
            .add_setup_system(
                setup_lj_tails.run_if(first_stage_only()),
                ScheduleSetupSet::PostSetup,
            )
            .add_update_system(lj_force.label("lj"), ScheduleSet::Force);
    }
}

// ── Pair table construction ─────────────────────────────────────────────────

pub fn build_lj_pair_table(
    lj: Res<LJConfig>,
    mut atoms: ResMut<Atom>,
    comm: Res<CommResource>,
    mut pair_table_res: ResMut<LJPairTable>,
) {
    let mixing = match lj.mixing.as_deref() {
        Some("arithmetic") => MixingRule::Arithmetic,
        _ => MixingRule::Geometric,
    };

    if let Some(ref types) = lj.types {
        let n = types.len();
        atoms.ntypes = n;
        let mut table = PairCoeffTable::new(n, LJPairCoeffs::default());

        for i in 0..n {
            for j in i..n {
                let eps = match mixing {
                    MixingRule::Geometric => (types[i].epsilon * types[j].epsilon).sqrt(),
                    MixingRule::Arithmetic => (types[i].epsilon + types[j].epsilon) / 2.0,
                };
                let sig = match mixing {
                    MixingRule::Geometric => (types[i].sigma * types[j].sigma).sqrt(),
                    MixingRule::Arithmetic => (types[i].sigma + types[j].sigma) / 2.0,
                };
                table.set(i, j, LJPairCoeffs::from_params(eps, sig, lj.cutoff));
            }
        }

        // Apply explicit overrides
        if let Some(ref overrides) = lj.pair_coeffs {
            for ov in overrides {
                let cut = ov.cutoff.unwrap_or(lj.cutoff);
                table.set(ov.types[0], ov.types[1], LJPairCoeffs::from_params(ov.epsilon, ov.sigma, cut));
            }
        }

        if comm.rank() == 0 {
            println!("LJ: {} types, mixing={:?}", n, mixing);
        }
        pair_table_res.0 = table;
    } else {
        // Single-type legacy mode
        atoms.ntypes = 1;
        let mut table = PairCoeffTable::new(1, LJPairCoeffs::default());
        table.set(0, 0, LJPairCoeffs::from_params(lj.epsilon, lj.sigma, lj.cutoff));
        pair_table_res.0 = table;
    };
}

// ── Systems ─────────────────────────────────────────────────────────────────

pub fn setup_lj_tails(
    lj: Res<LJConfig>,
    atoms: Res<Atom>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut tails: ResMut<LJTailCorrections>,
    pair_table_res: Res<LJPairTable>,
) {
    let n = comm.all_reduce_sum_f64(atoms.nlocal as f64);
    let v = domain.volume;
    let rho = n / v;

    let pair_table = &pair_table_res.0;
    let ntypes = pair_table.ntypes();

    let mut e_tail = 0.0;
    let mut p_tail = 0.0;

    // For simplicity, assume uniform type distribution when multi-type.
    // Single-type case: frac = 1.0, so result is identical to original.
    let frac = 1.0 / ntypes as f64;

    for i in 0..ntypes {
        for j in 0..ntypes {
            let c = pair_table.get(i as u32, j as u32);
            let rc = c.cutoff2.sqrt();
            let sigma3 = c.sigma.powi(3);
            let rc3 = rc.powi(3);
            let rc9 = rc3.powi(3);
            let sigma6 = sigma3 * sigma3;
            let sigma9 = sigma6 * sigma3;

            let w = frac * frac;
            e_tail += w * (8.0 / 3.0) * PI * n * rho * c.epsilon * sigma3
                * (sigma9 / (3.0 * rc9) - sigma3 / rc3);
            p_tail += w * (16.0 / 3.0) * PI * rho * rho * c.epsilon * sigma3
                * (2.0 * sigma9 / (3.0 * rc9) - sigma3 / rc3);
        }
    }

    tails.energy_tail = e_tail;
    tails.pressure_tail = p_tail;

    if comm.rank() == 0 {
        if ntypes == 1 {
            let c = pair_table.get(0, 0);
            println!(
                "LJ: eps={}, sigma={}, rc={}, rho={:.4}",
                c.epsilon, c.sigma, lj.cutoff, rho
            );
        } else {
            println!("LJ: {} types, rho={:.4}", ntypes, rho);
            eprintln!(
                "WARNING: LJ tail corrections assume equimolar type distribution (1/{} each). \
                 For non-equimolar mixtures, tail corrections will be approximate.",
                ntypes
            );
        }
        println!(
            "LJ tail corrections: E_tail={:.6}, P_tail={:.6}",
            tails.energy_tail, tails.pressure_tail
        );
    }
}

pub fn lj_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    _lj: Res<LJConfig>,
    _domain: Res<Domain>,
    virial: Option<ResMut<mddem_core::VirialStress>>,
    pair_table_res: Res<LJPairTable>,
) {
    let pair_table = &pair_table_res.0;
    let nlocal = atoms.nlocal as usize;
    let multi_type = pair_table.ntypes() > 1;

    let pos_ptr = atoms.pos.as_ptr();
    let force_ptr = atoms.force.as_mut_ptr();
    let offsets_ptr = neighbor.neighbor_offsets.as_ptr();
    let indices_ptr = neighbor.neighbor_indices.as_ptr();
    let type_ptr = atoms.atom_type.as_ptr();

    let virial_active = virial.as_ref().map_or(false, |v| v.active);

    if virial_active {
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
            let ti = unsafe { *type_ptr.add(i) };

            for k in start..end {
                let j = unsafe { *indices_ptr.add(k) } as usize;
                let pj = unsafe { *pos_ptr.add(j) };
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                let c = if multi_type {
                    let tj = unsafe { *type_ptr.add(j) };
                    pair_table.get(ti, tj)
                } else {
                    pair_table.get(0, 0)
                };

                if r2 >= c.cutoff2 {
                    continue;
                }

                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * c.lj1.mul_add(r6inv, -c.lj2);

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
        for i in 0..nlocal {
            let pi = unsafe { *pos_ptr.add(i) };
            let mut fi = unsafe { *force_ptr.add(i) };
            let start = unsafe { *offsets_ptr.add(i) } as usize;
            let end = unsafe { *offsets_ptr.add(i + 1) } as usize;
            let ti = unsafe { *type_ptr.add(i) };

            for k in start..end {
                let j = unsafe { *indices_ptr.add(k) } as usize;
                let pj = unsafe { *pos_ptr.add(j) };
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                let c = if multi_type {
                    let tj = unsafe { *type_ptr.add(j) };
                    pair_table.get(ti, tj)
                } else {
                    pair_table.get(0, 0)
                };

                if r2 >= c.cutoff2 {
                    continue;
                }

                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * c.lj1.mul_add(r6inv, -c.lj2);

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
        atom.push_test_atom(tag, [x, y, z], 0.5, mass);
    }

    fn make_two_atom_app(distance: f64) -> App {
        let mut app = App::new();

        let lj_config = LJConfig {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
            mixing: None,
            types: None,
            pair_coeffs: None,
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

        // Build pair table as resource
        let mut table = PairCoeffTable::new(1, LJPairCoeffs::default());
        table.set(0, 0, LJPairCoeffs::from_params(1.0, 1.0, 2.5));
        app.add_resource(LJPairTable(table));

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_list.push((0, 1));
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];
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
