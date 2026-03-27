//! # Lennard-Jones 12-6 Pair Potential
//!
//! This crate implements the standard Lennard-Jones 12-6 pair force for molecular dynamics
//! simulations in MDDEM. The LJ potential models van der Waals interactions between
//! neutral atoms or molecules:
//!
//! ```text
//!   V(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]
//! ```
//!
//! where **ε** (epsilon) is the depth of the potential well and **σ** (sigma) is the
//! finite distance at which the inter-particle potential is zero. The potential is
//! truncated at a cutoff distance `r_c` (specified in units of σ).
//!
//! ## Features
//!
//! - Single-type and multi-type pair interactions with mixing rules
//! - Precomputed pair coefficient table for fast inner-loop evaluation
//! - Virial stress accumulation for pressure computation
//! - Long-range tail corrections for energy and pressure beyond the cutoff
//! - Newton's third law optimization (half neighbor list)
//!
//! ## TOML Configuration
//!
//! ```toml
//! [lj]
//! epsilon = 1.0          # well depth ε (energy units), default: 1.0
//! sigma = 1.0            # particle diameter σ (length units), default: 1.0
//! cutoff = 2.5           # cutoff distance in units of σ, default: 2.5
//!
//! # Optional: mixing rule for multi-type ("geometric" or "arithmetic")
//! # mixing = "geometric"
//!
//! # Optional: per-type parameters (enables multi-type mode)
//! # [[lj.types]]
//! # epsilon = 1.0
//! # sigma = 1.0
//! #
//! # [[lj.types]]
//! # epsilon = 0.5
//! # sigma = 1.2
//!
//! # Optional: explicit pair coefficient overrides
//! # [[lj.pair_coeffs]]
//! # types = [0, 1]
//! # epsilon = 0.8
//! # sigma = 1.1
//! # cutoff = 3.0          # optional per-pair cutoff (in σ units)
//! ```
//!
//! ## Usage
//!
//! Register the [`LJForcePlugin`] with your app:
//!
//! ```rust,ignore
//! app.add_plugins(LJForcePlugin);
//! ```
//!
//! The plugin depends on `NeighborPlugin` and automatically registers the
//! [`VirialStressPlugin`](mddem_core::VirialStressPlugin) for pressure computation.

use std::f64::consts::PI;

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, MixingRule, PairCoeffTable, ScheduleSet, ScheduleSetupSet, VirialStressPlugin};
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

/// TOML configuration for the Lennard-Jones 12-6 pair potential.
///
/// Parsed from the `[lj]` section of the simulation config file.
/// Supports both single-type mode (using top-level `epsilon`/`sigma`) and
/// multi-type mode (using the `types` array with optional `mixing` rule).
///
/// # Example
///
/// Single-type (default):
/// ```toml
/// [lj]
/// epsilon = 1.0
/// sigma = 1.0
/// cutoff = 2.5
/// ```
///
/// Multi-type with mixing:
/// ```toml
/// [lj]
/// cutoff = 2.5
/// mixing = "geometric"
///
/// [[lj.types]]
/// epsilon = 1.0
/// sigma = 1.0
///
/// [[lj.types]]
/// epsilon = 0.5
/// sigma = 1.2
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LJConfig {
    /// Well depth ε (energy units). Default: `1.0`.
    /// Used in single-type mode; ignored when `types` is present.
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Particle diameter σ (length units). Default: `1.0`.
    /// Used in single-type mode; ignored when `types` is present.
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    /// Cutoff distance in units of σ. Default: `2.5`.
    /// Applied to all pairs unless overridden in `pair_coeffs`.
    #[serde(default = "default_cutoff")]
    pub cutoff: f64,
    /// Mixing rule for cross-type interactions: `"geometric"` (default) or `"arithmetic"`.
    /// Only relevant when `types` is present.
    #[serde(default)]
    pub mixing: Option<String>,
    /// Per-type LJ parameters. When present, enables multi-type mode
    /// and the top-level `epsilon`/`sigma` are ignored.
    #[serde(default)]
    pub types: Option<Vec<LJTypeConfig>>,
    /// Explicit pair coefficient overrides that bypass mixing rules.
    /// Each entry specifies the two type indices and their LJ parameters.
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
///
/// Each entry defines ε and σ for one atom type. The index in the array
/// corresponds to the atom type index (0-based).
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LJTypeConfig {
    /// Well depth ε for this type (energy units).
    pub epsilon: f64,
    /// Particle diameter σ for this type (length units).
    pub sigma: f64,
}

/// Explicit pair coefficient override from `[[lj.pair_coeffs]]`.
///
/// Allows specifying exact ε, σ, and optionally a per-pair cutoff for a
/// specific type pair, bypassing the mixing rule.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LJPairOverride {
    /// The two atom type indices (0-based) this override applies to.
    pub types: [usize; 2],
    /// Well depth ε for this pair (energy units).
    pub epsilon: f64,
    /// Particle diameter σ for this pair (length units).
    pub sigma: f64,
    /// Optional per-pair cutoff in units of σ. Falls back to the global `cutoff` if absent.
    #[serde(default)]
    pub cutoff: Option<f64>,
}

// ── Precomputed pair coefficients ───────────────────────────────────────────

/// Precomputed LJ pair coefficients for fast inner-loop evaluation.
///
/// The force between two particles at distance `r` is computed as:
///
/// ```text
///   F(r) = r⁻² × r⁻⁶ × (lj1 × r⁻⁶ − lj2)
/// ```
///
/// where `lj1 = 48εσ¹²` and `lj2 = 24εσ⁶`. This avoids recomputing powers
/// of σ on every pair interaction.
#[derive(Clone)]
pub struct LJPairCoeffs {
    /// `48 × ε × σ¹²` — coefficient for the repulsive r⁻¹² term.
    pub lj1: f64,
    /// `24 × ε × σ⁶` — coefficient for the attractive r⁻⁶ term.
    pub lj2: f64,
    /// Squared cutoff distance `(cutoff × σ)²` in absolute length units.
    pub cutoff2: f64,
    /// Original ε value, retained for tail correction calculations.
    pub epsilon: f64,
    /// Original σ value, retained for tail correction calculations.
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
    /// Create precomputed coefficients from physical parameters.
    ///
    /// - `epsilon`: well depth ε (energy units)
    /// - `sigma`: particle diameter σ (length units)
    /// - `cutoff_sigma`: cutoff distance in units of σ
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

/// Wrapper resource holding the symmetric pair coefficient table for all LJ type pairs.
pub struct LJPairTable(pub PairCoeffTable<LJPairCoeffs>);

// ── Resources ───────────────────────────────────────────────────────────────

/// Long-range tail corrections for energy and pressure beyond the LJ cutoff.
///
/// Because the LJ potential is truncated at `r_c`, the contributions from
/// pairs beyond the cutoff are approximated analytically assuming a uniform
/// pair distribution function g(r) = 1 for r > r_c:
///
/// ```text
///   E_tail = (8/3) π N ρ ε σ³ [ σ⁹/(3 r_c⁹) − σ³/r_c³ ]
///   P_tail = (16/3) π ρ² ε σ³ [ 2σ⁹/(3 r_c⁹) − σ³/r_c³ ]
/// ```
///
/// These corrections are computed once during setup and added to thermodynamic
/// output quantities.
pub struct LJTailCorrections {
    /// Total tail correction to energy (energy units).
    pub energy_tail: f64,
    /// Tail correction to pressure (pressure units).
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

/// Plugin that registers the LJ 12-6 pair force, virial accumulator, and tail corrections.
///
/// # Dependencies
///
/// Requires `NeighborPlugin` for neighbor list construction.
/// Automatically registers [`VirialStressPlugin`](mddem_core::VirialStressPlugin).
///
/// # Systems
///
/// - **`build_lj_pair_table`** (Setup): builds the pair coefficient table from config
/// - **`setup_lj_tails`** (PostSetup, first stage only): computes long-range tail corrections
/// - **`lj_force`** (Force): evaluates pair forces and accumulates virial stress
pub struct LJForcePlugin;

impl Plugin for LJForcePlugin {
    fn dependencies(&self) -> Vec<std::any::TypeId> {
        sim_app::type_ids![mddem_neighbor::NeighborPlugin]
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
        // Add a default 1×1 pair table; build_lj_pair_table will replace it during setup.
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

/// Builds the LJ pair coefficient table from the `[lj]` config section.
///
/// In multi-type mode, applies the configured mixing rule to compute cross-type
/// parameters, then applies any explicit `pair_coeffs` overrides. In single-type
/// mode, uses the top-level `epsilon` and `sigma` directly.
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
        // Multi-type mode: build NxN pair table using mixing rules
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

        // Apply explicit pair coefficient overrides (bypass mixing rules)
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
        // Single-type mode: 1×1 table from top-level epsilon/sigma
        atoms.ntypes = 1;
        let mut table = PairCoeffTable::new(1, LJPairCoeffs::default());
        table.set(0, 0, LJPairCoeffs::from_params(lj.epsilon, lj.sigma, lj.cutoff));
        pair_table_res.0 = table;
    };
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Computes long-range tail corrections for energy and pressure.
///
/// These analytical corrections account for the truncation of the LJ potential
/// at the cutoff distance, assuming a uniform radial distribution function
/// g(r) = 1 for r > r_c.
///
/// For multi-type systems, assumes an equimolar (uniform) type distribution.
/// A warning is printed for non-single-type systems since the correction is
/// approximate for non-equimolar mixtures.
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
    let rho = n / v; // number density

    let pair_table = &pair_table_res.0;
    let ntypes = pair_table.ntypes();

    let mut e_tail = 0.0;
    let mut p_tail = 0.0;

    // Assume uniform type distribution: each type has fraction 1/ntypes.
    // For single-type, frac = 1.0, so the result is exact.
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

            // Weight by type fraction squared (x_i * x_j) for the pair contribution
            let w = frac * frac;

            // E_tail = (8/3) π N ρ ε σ³ [ σ⁹/(3 r_c⁹) − σ³/r_c³ ]
            e_tail += w * (8.0 / 3.0) * PI * n * rho * c.epsilon * sigma3
                * (sigma9 / (3.0 * rc9) - sigma3 / rc3);

            // P_tail = (16/3) π ρ² ε σ³ [ 2σ⁹/(3 r_c⁹) − σ³/r_c³ ]
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

/// Evaluates LJ 12-6 pair forces for all local atoms using the neighbor list.
///
/// Uses Newton's third law (half neighbor list): each pair (i, j) is visited once,
/// and forces are applied to both atoms. When virial stress tracking is active,
/// the virial tensor components are accumulated simultaneously.
///
/// # Force computation
///
/// For each pair within the cutoff:
/// ```text
///   r²_inv = 1 / r²
///   r⁶_inv = r²_inv³
///   F_pair = r²_inv × r⁶_inv × (lj1 × r⁶_inv − lj2)
///   F_ij   = −F_pair × Δr
/// ```
///
/// # Safety
///
/// Uses raw pointer arithmetic for performance in the inner loop. This is safe
/// because neighbor list indices are guaranteed to be within bounds by the
/// neighbor list builder.
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

    // Raw pointers for inner-loop performance (avoids bounds checks)
    let pos_ptr = atoms.pos.as_ptr();
    let force_ptr = atoms.force.as_mut_ptr();
    let offsets_ptr = neighbor.neighbor_offsets.as_ptr();
    let indices_ptr = neighbor.neighbor_indices.as_ptr();
    let type_ptr = atoms.atom_type.as_ptr();

    let virial_active = virial.as_ref().map_or(false, |v| v.active);

    if virial_active {
        // Force loop with virial stress accumulation
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

                // LJ force: F = r⁻² × r⁻⁶ × (48εσ¹² × r⁻⁶ − 24εσ⁶)
                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * c.lj1.mul_add(r6inv, -c.lj2);

                let fx = -fpair * dx;
                let fy = -fpair * dy;
                let fz = -fpair * dz;

                // Virial: W_αβ = Σ r_α × F_β (summed over all pairs)
                vxx += dx * fx;
                vyy += dy * fy;
                vzz += dz * fz;
                vxy += dx * fy;
                vxz += dx * fz;
                vyz += dy * fz;

                // Newton's third law: equal and opposite forces
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
        // Force-only loop (no virial accumulation)
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

                // LJ force: F = r⁻² × r⁻⁶ × (48εσ¹² × r⁻⁶ − 24εσ⁶)
                let r2inv = 1.0 / r2;
                let r6inv = r2inv * r2inv * r2inv;
                let fpair = r2inv * r6inv * c.lj1.mul_add(r6inv, -c.lj2);

                // Newton's third law: equal and opposite forces
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
