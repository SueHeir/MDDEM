//! Type-filtered radial distribution function plugin for MDDEM.
//!
//! Measures g(r) between specific atom type pairs. Useful for any multi-type
//! simulation: binary LJ mixtures, alloys, polymer blends, water models, etc.
//!
//! # Configuration
//!
//! Use `[[type_rdf]]` array sections to measure multiple type pairs simultaneously:
//!
//! ```toml
//! [[type_rdf]]
//! type_i = 0
//! type_j = 0
//! bins = 200
//! cutoff = 4.0
//! interval = 50
//! output_interval = 5000
//!
//! [[type_rdf]]
//! type_i = 0
//! type_j = 1
//! bins = 200
//! cutoff = 4.0
//! interval = 50
//! output_interval = 5000
//! ```
//!
//! Each entry produces a separate output file: `data/type_rdf_0_0.txt`,
//! `data/type_rdf_0_1.txt`, etc.
//!
//! # Config fields
//!
//! | Field             | Type  | Default | Description                          |
//! |-------------------|-------|---------|--------------------------------------|
//! | `type_i`          | u32   | 0       | First atom type index for the pair   |
//! | `type_j`          | u32   | 0       | Second atom type index for the pair  |
//! | `bins`            | usize | 200     | Number of histogram bins             |
//! | `cutoff`          | f64   | 3.0     | RDF cutoff distance (length units)   |
//! | `interval`        | usize | 100     | Accumulate histogram every N steps   |
//! | `output_interval` | usize | 1000    | Write output file every N steps      |
//!
//! # Output format
//!
//! Each output file is a two-column text file with a header:
//!
//! ```text
//! # Type-filtered RDF: types (0, 1), 10 samples
//! # r g(r)
//! 0.007500 0.000000
//! 0.022500 0.000000
//! ...
//! ```
//!
//! The `r` column is the bin center and `g(r)` is the time-averaged pair
//! correlation function. For an ideal gas, g(r) = 1.0 at all distances.
//!
//! # Relationship to `md_measure`
//!
//! The existing [`md_measure`] crate computes a single global RDF over all
//! atoms regardless of type. This crate adds *type-filtered* RDF — g(r)
//! between specific (type_i, type_j) pairs — which is essential for
//! multi-component systems (binary LJ mixtures, alloys, water models, etc.).
//! Merging into `md_measure` is possible in the future, but keeping it
//! separate avoids coupling the simpler single-type RDF to the multi-type
//! config and normalization logic.

use std::any::TypeId;
use std::f64::consts::PI;
use std::fs;
use std::io::Write;

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState, ParticleSimScheduleSet};
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

/// Configuration for a single type-filtered RDF measurement.
///
/// Each `[[type_rdf]]` entry in the TOML config produces one of these.
/// When `type_i == type_j`, the RDF measures same-type correlations (e.g.,
/// solvent–solvent). When `type_i != type_j`, it measures cross-type
/// correlations (e.g., solvent–solute). The order does not matter:
/// `(type_i=0, type_j=1)` and `(type_i=1, type_j=0)` produce identical results.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TypeRdfEntry {
    /// First atom type index for the RDF pair (default: `0`).
    #[serde(default)]
    pub type_i: u32,
    /// Second atom type index for the RDF pair (default: `0`).
    #[serde(default)]
    pub type_j: u32,
    /// Number of histogram bins (default: `200`). More bins give finer radial
    /// resolution but noisier curves for small systems.
    #[serde(default = "default_type_rdf_bins")]
    pub bins: usize,
    /// RDF cutoff distance in simulation length units (default: `3.0`).
    /// Should not exceed half the smallest periodic box dimension.
    #[serde(default = "default_type_rdf_cutoff")]
    pub cutoff: f64,
    /// Accumulate the RDF histogram every N timesteps (default: `100`).
    #[serde(default = "default_type_rdf_interval")]
    pub interval: usize,
    /// Write the averaged g(r) to disk every N timesteps (default: `1000`).
    #[serde(default = "default_type_rdf_output_interval")]
    pub output_interval: usize,
}

impl Default for TypeRdfEntry {
    fn default() -> Self {
        TypeRdfEntry {
            type_i: 0,
            type_j: 0,
            bins: 200,
            cutoff: 3.0,
            interval: 100,
            output_interval: 1000,
        }
    }
}

/// Top-level config holding all `[[type_rdf]]` entries.
#[derive(Clone)]
pub struct TypeRdfConfig {
    /// All configured type-pair RDF measurements.
    pub entries: Vec<TypeRdfEntry>,
}

impl Default for TypeRdfConfig {
    fn default() -> Self {
        TypeRdfConfig {
            entries: Vec::new(),
        }
    }
}

// ── Resources ───────────────────────────────────────────────────────────────

/// Accumulates the type-filtered radial distribution function histogram for one
/// (type_i, type_j) pair.
///
/// Each bin stores the *cumulative* normalized g(r) contribution across all
/// samples. To obtain the time-averaged g(r), divide each bin value by
/// [`n_samples`](Self::n_samples).
pub struct TypeRdfAccumulator {
    /// Histogram bins storing cumulative g(r) contributions (divide by
    /// `n_samples` for the time-averaged g(r)).
    pub bins: Vec<f64>,
    /// Number of accumulated RDF samples so far.
    pub n_samples: usize,
    /// Bin width in simulation length units: `cutoff / num_bins`.
    pub dr: f64,
    /// RDF cutoff distance in simulation length units.
    pub cutoff: f64,
    /// First atom type index for this pair.
    pub type_i: u32,
    /// Second atom type index for this pair.
    pub type_j: u32,
    /// Accumulate the histogram every N timesteps.
    pub interval: usize,
    /// Write output every N timesteps.
    pub output_interval: usize,
}

impl TypeRdfAccumulator {
    /// Create a new accumulator from a config entry, with all bins zeroed.
    fn new(entry: &TypeRdfEntry) -> Self {
        TypeRdfAccumulator {
            bins: vec![0.0; entry.bins],
            n_samples: 0,
            dr: entry.cutoff / entry.bins as f64,
            cutoff: entry.cutoff,
            type_i: entry.type_i,
            type_j: entry.type_j,
            interval: entry.interval,
            output_interval: entry.output_interval,
        }
    }
}

/// Collection of all type-filtered RDF accumulators.
pub struct TypeRdfAccumulators {
    /// One accumulator per `[[type_rdf]]` config entry.
    pub accumulators: Vec<TypeRdfAccumulator>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Type-filtered RDF measurement plugin.
///
/// Measures g(r) between specified pairs of atom types. Supports multiple
/// simultaneous type pairs via `[[type_rdf]]` TOML array config.
///
/// For a single-type simulation (all atoms type 0), this is equivalent to
/// the standard RDF. For multi-type systems, this allows measuring specific
/// pair correlations like g_00(r), g_01(r), g_11(r).
pub struct TypeRdfPlugin;

impl Plugin for TypeRdfPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[[type_rdf]]
type_i = 0       # first atom type for RDF
type_j = 0       # second atom type for RDF
bins = 200       # number of RDF histogram bins
cutoff = 3.0     # RDF cutoff distance
interval = 100   # accumulate RDF every N steps
output_interval = 1000  # write output every N steps"#,
        )
    }

    fn build(&self, app: &mut App) {
        // Parse [[type_rdf]] array from TOML config
        let entries: Vec<TypeRdfEntry> =
            if let Some(raw_cell) = app.get_mut_resource(TypeId::of::<Config>()) {
                let raw = raw_cell.borrow();
                let config = raw.downcast_ref::<Config>().expect(
                    "Config resource must be registered before TypeRdfPlugin",
                );
                let e = config.parse_array::<TypeRdfEntry>("type_rdf");
                drop(raw);
                e
            } else {
                Vec::new()
            };

        let config = TypeRdfConfig {
            entries: entries.clone(),
        };

        let accumulators = TypeRdfAccumulators {
            accumulators: entries.iter().map(TypeRdfAccumulator::new).collect(),
        };

        app.add_resource(config)
            .add_resource(accumulators)
            .add_update_system(
                accumulate_type_rdf,
                ParticleSimScheduleSet::PostFinalIntegration,
            )
            .add_update_system(write_type_rdf, ParticleSimScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Accumulate type-filtered RDF using brute-force O(N²) pair counting with
/// minimum-image convention.
///
/// **Why brute-force instead of neighbor lists?**
/// RDF measurement typically uses a cutoff larger than the force cutoff (and
/// therefore larger than the neighbor-list ghost cutoff). Using the existing
/// neighbor list would silently miss pairs beyond the force cutoff, producing
/// incorrect g(r). The brute-force loop guarantees correctness for arbitrary
/// RDF cutoffs at the cost of O(N²) scaling, which is acceptable because
/// RDF accumulation runs infrequently (every `interval` steps) rather than
/// every timestep.
///
/// Iterates over all configured type pairs and accumulates their histograms.
/// For pairs where type_i == type_j, counts local-local pairs (i < j) to
/// avoid double counting. For type_i != type_j, counts all (i, j) pairs
/// where atom i has type_i and atom j has type_j.
pub fn accumulate_type_rdf(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    run_state: Res<RunState>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut accumulators: ResMut<TypeRdfAccumulators>,
) {
    let step = run_state.total_cycle;
    if step == 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;

    for acc in accumulators.accumulators.iter_mut() {
        if !step.is_multiple_of(acc.interval) {
            continue;
        }

        let n_bins = acc.bins.len();
        let dr = acc.dr;
        let type_i = acc.type_i;
        let type_j = acc.type_j;
        let same_type = type_i == type_j;

        // Cap RDF cutoff at ghost_cutoff on multi-proc
        let effective_cutoff = if comm.size() > 1 && neighbor.ghost_cutoff > 0.0 {
            acc.cutoff.min(neighbor.ghost_cutoff)
        } else {
            acc.cutoff
        };
        let cutoff2 = effective_cutoff * effective_cutoff;

        let mut local_hist = vec![0.0f64; n_bins];

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
            // For cross-type: atom i must be type_i or type_j
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

                if domain.is_periodic(0) {
                    if dx > half_lx {
                        dx -= lx;
                    } else if dx < -half_lx {
                        dx += lx;
                    }
                }
                if domain.is_periodic(1) {
                    if dy > half_ly {
                        dy -= ly;
                    } else if dy < -half_ly {
                        dy += ly;
                    }
                }
                if domain.is_periodic(2) {
                    if dz > half_lz {
                        dz -= lz;
                    } else if dz < -half_lz {
                        dz += lz;
                    }
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

            // Local-ghost pairs (weight 0.5 to avoid double counting across ranks)
            if comm.size() > 1 {
                for j in nlocal..total {
                    let tj = atoms.atom_type[j];

                    let valid = if same_type {
                        i_is_type_i && tj == type_j
                    } else {
                        (i_is_type_i && tj == type_j) || (i_is_type_j && tj == type_i)
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
                        local_hist[bin] += 0.5;
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

        // ── Normalization ──
        //
        // The radial distribution function is defined as:
        //   g(r) = (V / N_pairs) * n(r) / (4π r² dr)
        //
        // where n(r) is the number of pairs in the shell [r, r+dr], V is
        // the box volume, and N_pairs is the total number of distinct
        // type-matched pairs:
        //   same-type:  N_pairs = N_i * (N_i - 1) / 2
        //   cross-type: N_pairs = N_i * N_j
        //
        // We use the exact shell volume 4π/3 * (r_high³ - r_low³) instead
        // of the thin-shell approximation 4π r² dr for better accuracy at
        // small r where dr/r is not negligible.
        let n_pairs = if same_type {
            n_i * (n_i - 1.0) / 2.0
        } else {
            n_i * n_j
        };

        for (k, (hist_val, rdf_bin)) in local_hist.iter().zip(acc.bins.iter_mut()).enumerate() {
            let r_low = k as f64 * dr;
            let r_high = (k + 1) as f64 * dr;
            let shell_vol = (4.0 / 3.0) * PI * (r_high.powi(3) - r_low.powi(3));
            if shell_vol > 1e-30 && n_pairs > 0.0 {
                *rdf_bin += hist_val * v / (n_pairs * shell_vol);
            }
        }
        acc.n_samples += 1;
    }
}

/// Write the time-averaged type-filtered g(r) to disk for each configured pair.
///
/// Output path: `<output_dir>/data/type_rdf_<type_i>_<type_j>.txt`.
/// Only rank 0 writes. Each write overwrites the previous file, so the file
/// always contains the latest cumulative average.
pub fn write_type_rdf(
    run_state: Res<RunState>,
    accumulators: Res<TypeRdfAccumulators>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step == 0 {
        return;
    }
    if comm.rank() != 0 {
        return;
    }

    let base_dir = input.output_dir.as_deref().unwrap_or(".");
    let data_dir = format!("{}/data", base_dir);

    for acc in &accumulators.accumulators {
        if !step.is_multiple_of(acc.output_interval) {
            continue;
        }
        if acc.n_samples == 0 {
            continue;
        }

        let _ = fs::create_dir_all(&data_dir);
        let path = format!("{}/type_rdf_{}_{}.txt", data_dir, acc.type_i, acc.type_j);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(
                f,
                "# Type-filtered RDF: types ({}, {}), {} samples",
                acc.type_i, acc.type_j, acc.n_samples
            )
            .ok();
            writeln!(f, "# r g(r)").ok();
            let n_bins = acc.bins.len();
            for k in 0..n_bins {
                // Bin center: midpoint of [k*dr, (k+1)*dr]
                let r = (k as f64 + 0.5) * acc.dr;
                // Time-averaged g(r) = cumulative g(r) / number of samples
                let gr = acc.bins[k] / acc.n_samples as f64;
                writeln!(f, "{:.6} {:.6}", r, gr).ok();
            }
            println!(
                "Wrote type-filtered RDF ({}-{}) to {}",
                acc.type_i, acc.type_j, path
            );
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_rdf_accumulator_init() {
        let entry = TypeRdfEntry {
            bins: 100,
            cutoff: 3.0,
            ..Default::default()
        };
        let rdf = TypeRdfAccumulator::new(&entry);
        assert_eq!(rdf.bins.len(), 100);
        assert!((rdf.dr - 0.03).abs() < 1e-10);
        assert_eq!(rdf.n_samples, 0);
        assert_eq!(rdf.type_i, 0);
        assert_eq!(rdf.type_j, 0);
    }

    #[test]
    fn type_rdf_cross_type_init() {
        let entry = TypeRdfEntry {
            type_i: 0,
            type_j: 1,
            bins: 200,
            cutoff: 5.0,
            ..Default::default()
        };
        let rdf = TypeRdfAccumulator::new(&entry);
        assert_eq!(rdf.bins.len(), 200);
        assert!((rdf.dr - 0.025).abs() < 1e-10);
        assert_eq!(rdf.type_i, 0);
        assert_eq!(rdf.type_j, 1);
    }

    #[test]
    fn type_rdf_entry_defaults() {
        let entry = TypeRdfEntry::default();
        assert_eq!(entry.type_i, 0);
        assert_eq!(entry.type_j, 0);
        assert_eq!(entry.bins, 200);
        assert!((entry.cutoff - 3.0).abs() < 1e-10);
        assert_eq!(entry.interval, 100);
        assert_eq!(entry.output_interval, 1000);
    }

    #[test]
    fn type_rdf_config_default_empty() {
        let config = TypeRdfConfig::default();
        assert!(config.entries.is_empty());
    }

    /// Verify that uniformly distributed random particles produce g(r) ≈ 1.0.
    ///
    /// For an ideal gas (no interactions), the pair correlation function is
    /// exactly 1.0 at all distances. We place N particles uniformly at random
    /// in a cubic periodic box and compute g(r) using the same normalization
    /// as the plugin. With enough particles, every bin beyond the exclusion
    /// zone should be close to 1.0.
    #[test]
    fn uniform_random_particles_give_gr_approx_one() {
        use std::f64::consts::PI;

        // Deterministic pseudo-random via simple LCG
        let mut seed: u64 = 123456789;
        let mut rand_f64 = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };

        let n = 2000usize;
        let box_l = 20.0f64;
        let volume = box_l * box_l * box_l;
        let half_l = box_l * 0.5;

        // Generate random positions
        let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
        for _ in 0..n {
            pos.push([
                rand_f64() * box_l,
                rand_f64() * box_l,
                rand_f64() * box_l,
            ]);
        }

        // Compute RDF histogram (brute-force, minimum-image)
        let n_bins = 100usize;
        let cutoff = 8.0f64;
        let dr = cutoff / n_bins as f64;
        let mut hist = vec![0.0f64; n_bins];

        for i in 0..n {
            for j in (i + 1)..n {
                let mut dx = pos[j][0] - pos[i][0];
                let mut dy = pos[j][1] - pos[i][1];
                let mut dz = pos[j][2] - pos[i][2];
                if dx > half_l { dx -= box_l; } else if dx < -half_l { dx += box_l; }
                if dy > half_l { dy -= box_l; } else if dy < -half_l { dy += box_l; }
                if dz > half_l { dz -= box_l; } else if dz < -half_l { dz += box_l; }

                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < cutoff {
                    let bin = (r / dr) as usize;
                    if bin < n_bins {
                        hist[bin] += 1.0;
                    }
                }
            }
        }

        // Normalize to g(r)
        let n_pairs = (n * (n - 1)) as f64 / 2.0;
        let mut gr = vec![0.0f64; n_bins];
        for k in 0..n_bins {
            let r_low = k as f64 * dr;
            let r_high = (k + 1) as f64 * dr;
            let shell_vol = (4.0 / 3.0) * PI * (r_high.powi(3) - r_low.powi(3));
            gr[k] = hist[k] * volume / (n_pairs * shell_vol);
        }

        // Check that g(r) ≈ 1.0 for bins beyond r = 1.0 (skip first few bins
        // where statistical noise is large due to small shell volume)
        let start_bin = (1.0 / dr).ceil() as usize;
        for k in start_bin..n_bins {
            let r = (k as f64 + 0.5) * dr;
            assert!(
                (gr[k] - 1.0).abs() < 0.15,
                "g(r={:.2}) = {:.4}, expected ≈ 1.0 for uniform random particles",
                r,
                gr[k]
            );
        }

        // Also check the mean is very close to 1.0
        let mean_gr: f64 = gr[start_bin..].iter().sum::<f64>() / (n_bins - start_bin) as f64;
        assert!(
            (mean_gr - 1.0).abs() < 0.02,
            "Mean g(r) = {:.4}, expected ≈ 1.0",
            mean_gr
        );
    }

    #[test]
    fn type_rdf_multiple_pairs() {
        let entries = vec![
            TypeRdfEntry {
                type_i: 0,
                type_j: 0,
                ..Default::default()
            },
            TypeRdfEntry {
                type_i: 0,
                type_j: 1,
                ..Default::default()
            },
            TypeRdfEntry {
                type_i: 1,
                type_j: 1,
                ..Default::default()
            },
        ];
        let accumulators = TypeRdfAccumulators {
            accumulators: entries.iter().map(TypeRdfAccumulator::new).collect(),
        };
        assert_eq!(accumulators.accumulators.len(), 3);
        assert_eq!(accumulators.accumulators[0].type_i, 0);
        assert_eq!(accumulators.accumulators[0].type_j, 0);
        assert_eq!(accumulators.accumulators[1].type_i, 0);
        assert_eq!(accumulators.accumulators[1].type_j, 1);
        assert_eq!(accumulators.accumulators[2].type_i, 1);
        assert_eq!(accumulators.accumulators[2].type_j, 1);
    }
}
