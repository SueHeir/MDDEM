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

use std::any::TypeId;
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

/// Configuration for a single type-filtered RDF measurement.
///
/// Each `[[type_rdf]]` entry in the TOML config produces one of these.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TypeRdfEntry {
    /// First atom type index for the RDF pair.
    #[serde(default)]
    pub type_i: u32,
    /// Second atom type index for the RDF pair.
    #[serde(default)]
    pub type_j: u32,
    /// Number of histogram bins.
    #[serde(default = "default_type_rdf_bins")]
    pub bins: usize,
    /// RDF cutoff distance.
    #[serde(default = "default_type_rdf_cutoff")]
    pub cutoff: f64,
    /// Accumulate RDF every N steps.
    #[serde(default = "default_type_rdf_interval")]
    pub interval: usize,
    /// Write output every N steps.
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

/// Accumulates type-filtered radial distribution function histogram for one type pair.
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
    /// Accumulate every N steps.
    pub interval: usize,
    /// Write output every N steps.
    pub output_interval: usize,
}

impl TypeRdfAccumulator {
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
                let config = raw.downcast_ref::<Config>().unwrap();
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
                ScheduleSet::PostFinalIntegration,
            )
            .add_update_system(write_type_rdf, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Accumulate type-filtered RDF using brute-force pair counting with
/// minimum-image convention.
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

                if domain.is_periodic[0] {
                    if dx > half_lx {
                        dx -= lx;
                    } else if dx < -half_lx {
                        dx += lx;
                    }
                }
                if domain.is_periodic[1] {
                    if dy > half_ly {
                        dy -= ly;
                    } else if dy < -half_ly {
                        dy += ly;
                    }
                }
                if domain.is_periodic[2] {
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

        // Normalize: g(r) = hist[k] * V / (N_pairs * 4*pi*r^2*dr)
        // For same-type: N_pairs = N_i * (N_i - 1) / 2
        // For cross-type: N_pairs = N_i * N_j
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

/// Write type-filtered RDF to `data/type_rdf_<i>_<j>.txt` for each configured pair.
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
                let r = (k as f64 + 0.5) * acc.dr;
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
