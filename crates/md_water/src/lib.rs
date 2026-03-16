//! Water model support for MDDEM.
//!
//! Provides a type-filtered radial distribution function (RDF) plugin that
//! measures g(r) between specific atom type pairs. While motivated by the
//! SPC/E water model (O-O correlations), the implementation is fully general
//! and works for any multi-type simulation.
//!
//! # SPC/E Reference Parameters
//!
//! The SPC/E water model uses:
//! - O-O Lennard-Jones: epsilon = 0.1553 kcal/mol, sigma = 3.166 A
//! - O-H bond length: 1.0 A, H-O-H angle: 109.47 deg
//! - Charges: q_O = -0.8476 e, q_H = +0.4238 e
//!
//! For the O-O only proof-of-concept, we use only the LJ part in reduced
//! units (epsilon* = 1, sigma* = 1) at liquid-state conditions.

use std::f64::consts::PI;
use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

fn default_type_rdf_bins() -> usize {
    200
}
fn default_type_rdf_cutoff() -> f64 {
    3.0
}
fn default_type_rdf_interval() -> usize {
    100
}
fn default_type_rdf_output_interval() -> usize {
    1000
}

/// Configuration for type-filtered RDF measurement.
///
/// TOML section `[water]`. Measures g(r) between a specific pair of atom types.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct WaterConfig {
    /// First atom type index for the RDF pair.
    #[serde(default)]
    pub rdf_type_i: u32,
    /// Second atom type index for the RDF pair.
    #[serde(default)]
    pub rdf_type_j: u32,
    /// Number of histogram bins.
    #[serde(default = "default_type_rdf_bins")]
    pub rdf_bins: usize,
    /// RDF cutoff distance.
    #[serde(default = "default_type_rdf_cutoff")]
    pub rdf_cutoff: f64,
    /// Accumulate RDF every N steps.
    #[serde(default = "default_type_rdf_interval")]
    pub rdf_interval: usize,
    /// Write output every N steps.
    #[serde(default = "default_type_rdf_output_interval")]
    pub output_interval: usize,
}

impl Default for WaterConfig {
    fn default() -> Self {
        WaterConfig {
            rdf_type_i: 0,
            rdf_type_j: 0,
            rdf_bins: 200,
            rdf_cutoff: 3.0,
            rdf_interval: 100,
            output_interval: 1000,
        }
    }
}

// ── Resources ───────────────────────────────────────────────────────────────

/// Accumulates type-filtered radial distribution function histogram.
pub struct TypeRdfAccumulator {
    /// Histogram bins (unnormalized, accumulated across samples).
    pub bins: Vec<f64>,
    /// Number of accumulated samples.
    pub n_samples: usize,
    /// Bin width.
    pub dr: f64,
    /// Cutoff distance.
    pub cutoff: f64,
    /// Atom type i.
    pub type_i: u32,
    /// Atom type j.
    pub type_j: u32,
}

impl TypeRdfAccumulator {
    fn new(n_bins: usize, cutoff: f64, type_i: u32, type_j: u32) -> Self {
        TypeRdfAccumulator {
            bins: vec![0.0; n_bins],
            n_samples: 0,
            dr: cutoff / n_bins as f64,
            cutoff,
            type_i,
            type_j,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Type-filtered RDF measurement plugin.
///
/// Measures g(r) between a specified pair of atom types. For an O-O only
/// simulation (all atoms type 0), this is equivalent to the standard RDF.
/// For multi-type systems (e.g., O + H), this allows measuring specific
/// pair correlations like g_OO(r), g_OH(r), g_HH(r).
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[water]
rdf_type_i = 0       # first atom type for RDF
rdf_type_j = 0       # second atom type for RDF
rdf_bins = 200       # number of RDF histogram bins
rdf_cutoff = 3.0     # RDF cutoff distance
rdf_interval = 100   # accumulate RDF every N steps
output_interval = 1000  # write output every N steps"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = Config::load::<WaterConfig>(app, "water");

        app.add_resource(TypeRdfAccumulator::new(
            config.rdf_bins,
            config.rdf_cutoff,
            config.rdf_type_i,
            config.rdf_type_j,
        ))
        .add_update_system(
            accumulate_type_rdf,
            ScheduleSet::PostFinalIntegration,
        )
        .add_update_system(
            write_type_rdf,
            ScheduleSet::PostFinalIntegration,
        );
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Accumulate type-filtered RDF using brute-force pair counting with
/// minimum-image convention.
///
/// For pairs where type_i == type_j, counts local-local pairs (i < j) to
/// avoid double counting. For type_i != type_j, counts all (i, j) pairs
/// where atom i has type_i and atom j has type_j.
pub fn accumulate_type_rdf(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    run_state: Res<RunState>,
    config: Res<WaterConfig>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut rdf: ResMut<TypeRdfAccumulator>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.rdf_interval) {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let n_bins = rdf.bins.len();
    let dr = rdf.dr;
    let type_i = rdf.type_i;
    let type_j = rdf.type_j;
    let same_type = type_i == type_j;

    // Cap RDF cutoff at ghost_cutoff on multi-proc
    let effective_cutoff = if comm.size() > 1 && neighbor.ghost_cutoff > 0.0 {
        rdf.cutoff.min(neighbor.ghost_cutoff)
    } else {
        rdf.cutoff
    };
    let cutoff2 = effective_cutoff * effective_cutoff;

    let mut local_hist = vec![0.0f64; n_bins];

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;
    let total = atoms.len();

    // Count type-i atoms locally (for normalization)
    let n_type_i_local: f64 = atoms.atom_type[..nlocal]
        .iter()
        .filter(|&&t| t == type_i)
        .count() as f64;
    let n_type_j_local: f64 = atoms.atom_type[..nlocal]
        .iter()
        .filter(|&&t| t == type_j)
        .count() as f64;

    // Brute-force pair counting with minimum-image convention
    for i in 0..nlocal {
        let ti = atoms.atom_type[i];

        // For same-type: atom i must be type_i, and we count j > i
        // For cross-type: atom i must be type_i, atom j must be type_j
        let i_is_type_i = ti == type_i;
        let i_is_type_j = ti == type_j;

        if !i_is_type_i && !((!same_type) && i_is_type_j) {
            continue;
        }

        // Local-local pairs
        let j_start = if same_type { i + 1 } else { 0 };
        for j in j_start..nlocal {
            if same_type && j == i {
                continue;
            }

            let tj = atoms.atom_type[j];

            // Check type matching
            let valid = if same_type {
                // Both must be type_i (== type_j)
                i_is_type_i && tj == type_j
            } else {
                // (i=type_i, j=type_j) or (i=type_j, j=type_i)
                (i_is_type_i && tj == type_j) || (i_is_type_j && tj == type_i)
            };
            if !valid {
                continue;
            }

            let mut dx = atoms.pos[j][0] - atoms.pos[i][0];
            let mut dy = atoms.pos[j][1] - atoms.pos[i][1];
            let mut dz = atoms.pos[j][2] - atoms.pos[i][2];

            if domain.is_periodic[0] {
                if dx > half_lx { dx -= lx; } else if dx < -half_lx { dx += lx; }
            }
            if domain.is_periodic[1] {
                if dy > half_ly { dy -= ly; } else if dy < -half_ly { dy += ly; }
            }
            if domain.is_periodic[2] {
                if dz > half_lz { dz -= lz; } else if dz < -half_lz { dz += lz; }
            }

            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= cutoff2 || r2 < 1e-20 {
                continue;
            }
            let bin = (r2.sqrt() / dr) as usize;
            if bin < n_bins {
                local_hist[bin] += 1.0;
            }
        }

        // Local-ghost pairs (weight 0.5 for same-type, 1.0 for cross-type when i is type_i only)
        if comm.size() > 1 {
            for j in nlocal..total {
                let tj = atoms.atom_type[j];

                let (valid, weight) = if same_type {
                    (i_is_type_i && tj == type_j, 0.5)
                } else {
                    // Only count i=type_i, j=type_j direction; the reverse is handled
                    // when the other rank owns the type_j atom as local
                    if i_is_type_i && tj == type_j {
                        (true, 0.5)
                    } else if i_is_type_j && tj == type_i {
                        (true, 0.5)
                    } else {
                        (false, 0.0)
                    }
                };
                if !valid {
                    continue;
                }

                let dx = atoms.pos[j][0] - atoms.pos[i][0];
                let dy = atoms.pos[j][1] - atoms.pos[i][1];
                let dz = atoms.pos[j][2] - atoms.pos[i][2];

                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 || r2 < 1e-20 {
                    continue;
                }
                let bin = (r2.sqrt() / dr) as usize;
                if bin < n_bins {
                    local_hist[bin] += weight;
                }
            }
        }
    }

    // Reduce histogram across ranks
    for val in &mut local_hist {
        *val = comm.all_reduce_sum_f64(*val);
    }

    let n_i = comm.all_reduce_sum_f64(n_type_i_local);
    let n_j = comm.all_reduce_sum_f64(n_type_j_local);
    let v = domain.volume;

    // Normalize: g(r) = hist[k] * V / (N_pairs * 4*pi*r^2*dr)
    // For same-type: N_pairs = N_i * (N_i - 1) / 2
    // For cross-type: N_pairs = N_i * N_j
    let n_pairs = if same_type {
        n_i * (n_i - 1.0) / 2.0
    } else {
        n_i * n_j
    };

    for (k, (hist_val, rdf_bin)) in local_hist.iter().zip(rdf.bins.iter_mut()).enumerate() {
        let r_low = k as f64 * dr;
        let r_high = (k + 1) as f64 * dr;
        let shell_vol = (4.0 / 3.0) * PI * (r_high.powi(3) - r_low.powi(3));
        if shell_vol > 1e-30 && n_pairs > 0.0 {
            *rdf_bin += hist_val * v / (n_pairs * shell_vol);
        }
    }
    rdf.n_samples += 1;
}

/// Write type-filtered RDF to `data/type_rdf.txt`.
pub fn write_type_rdf(
    run_state: Res<RunState>,
    config: Res<WaterConfig>,
    rdf: Res<TypeRdfAccumulator>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.output_interval) {
        return;
    }
    if comm.rank() != 0 {
        return;
    }

    if rdf.n_samples == 0 {
        return;
    }

    let base_dir = input.output_dir.as_deref().unwrap_or(".");
    let data_dir = format!("{}/data", base_dir);
    let _ = fs::create_dir_all(&data_dir);

    let path = format!("{}/type_rdf.txt", data_dir);
    if let Ok(mut f) = fs::File::create(&path) {
        writeln!(
            f,
            "# Type-filtered RDF: types ({}, {}), {} samples",
            rdf.type_i, rdf.type_j, rdf.n_samples
        )
        .ok();
        writeln!(f, "# r g(r)").ok();
        let n_bins = rdf.bins.len();
        for k in 0..n_bins {
            let r = (k as f64 + 0.5) * rdf.dr;
            let gr = rdf.bins[k] / rdf.n_samples as f64;
            writeln!(f, "{:.6} {:.6}", r, gr).ok();
        }
        println!(
            "Wrote type-filtered RDF ({}-{}) to {}",
            rdf.type_i, rdf.type_j, path
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_rdf_accumulator_init() {
        let rdf = TypeRdfAccumulator::new(100, 3.0, 0, 0);
        assert_eq!(rdf.bins.len(), 100);
        assert!((rdf.dr - 0.03).abs() < 1e-10);
        assert_eq!(rdf.n_samples, 0);
        assert_eq!(rdf.type_i, 0);
        assert_eq!(rdf.type_j, 0);
    }

    #[test]
    fn type_rdf_cross_type_init() {
        let rdf = TypeRdfAccumulator::new(200, 5.0, 0, 1);
        assert_eq!(rdf.bins.len(), 200);
        assert!((rdf.dr - 0.025).abs() < 1e-10);
        assert_eq!(rdf.type_i, 0);
        assert_eq!(rdf.type_j, 1);
    }

    #[test]
    fn water_config_defaults() {
        let config = WaterConfig::default();
        assert_eq!(config.rdf_type_i, 0);
        assert_eq!(config.rdf_type_j, 0);
        assert_eq!(config.rdf_bins, 200);
        assert!((config.rdf_cutoff - 3.0).abs() < 1e-10);
    }
}
