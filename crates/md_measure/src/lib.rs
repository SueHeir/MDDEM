//! Measurement tools for molecular dynamics simulations: radial distribution
//! function (RDF), mean square displacement (MSD), and virial pressure.
//!
//! # Overview
//!
//! This crate provides three complementary measurement systems that run during
//! the `PostFinalIntegration` schedule set:
//!
//! - **RDF** — Computes the radial distribution function g(r), which measures
//!   how particle density varies as a function of distance from a reference
//!   particle. Useful for characterizing liquid structure.
//!
//! - **MSD** — Tracks mean square displacement ⟨|r(t) − r(0)|²⟩ over time,
//!   using unwrapped coordinates to handle periodic boundary crossings.
//!   Useful for measuring diffusion coefficients (D = MSD / 6t in 3D).
//!
//! - **Virial pressure** — Computes the instantaneous scalar pressure from
//!   the kinetic (ideal gas) and virial (interaction) contributions, plus
//!   optional LJ tail corrections.
//!
//! # TOML Configuration
//!
//! All settings live under the `[measure]` section:
//!
//! ```toml
//! [measure]
//! rdf_bins = 200          # Number of histogram bins for g(r) (default: 200)
//! rdf_cutoff = 3.0        # Maximum pair distance for RDF, in LJ units (default: 3.0)
//! rdf_interval = 100      # Accumulate an RDF sample every N steps (default: 100)
//! msd_interval = 10       # Record MSD and pressure every N steps (default: 10)
//! output_interval = 1000  # Write output files every N steps (default: 1000)
//! ```
//!
//! # Output Files
//!
//! All measurement files are written to `<output_dir>/data/`:
//!
//! - **`rdf.txt`** — Two columns: `r  g(r)`, where `r` is the bin center and
//!   `g(r)` is the time-averaged radial distribution function.
//! - **`msd.txt`** — Two columns: `dt  MSD`, where `dt` is elapsed steps since
//!   reference and `MSD` is the ensemble-averaged mean square displacement.
//! - **`pressure.txt`** — Two columns: `step  pressure`, giving the
//!   instantaneous virial pressure at each sampled timestep.
//!
//! # Plugin Registration
//!
//! Add [`MeasurePlugin`] to the app to enable all measurements:
//!
//! ```ignore
//! app.add_plugin(MeasurePlugin);
//! ```

use std::f64::consts::PI;
use std::fs;
use std::io::Write;

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::Deserialize;

use md_lj::LJTailCorrections;
use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState, ScheduleSet, ScheduleSetupSet, VirialStress};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

fn default_rdf_bins() -> usize {
    200
}
fn default_rdf_cutoff() -> f64 {
    3.0
}
fn default_rdf_interval() -> usize {
    100
}
fn default_msd_interval() -> usize {
    10
}
fn default_output_interval() -> usize {
    1000
}

/// TOML `[measure]` — configuration for RDF, MSD, and pressure measurements.
///
/// All fields have sensible defaults and can be omitted from the config file.
///
/// # Fields
///
/// | Field             | Type    | Default | Description                                    |
/// |-------------------|---------|---------|------------------------------------------------|
/// | `rdf_bins`        | `usize` | 200     | Number of histogram bins for g(r)             |
/// | `rdf_cutoff`      | `f64`  | 3.0     | Maximum pair distance for RDF (LJ units)       |
/// | `rdf_interval`    | `usize` | 100     | Accumulate an RDF sample every N steps         |
/// | `msd_interval`    | `usize` | 10      | Record MSD and pressure every N steps          |
/// | `output_interval` | `usize` | 1000    | Write measurement files every N steps          |
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MeasureConfig {
    /// Number of histogram bins for the RDF. More bins give finer radial
    /// resolution but noisier curves; fewer bins give smoother but coarser
    /// results. Bin width is `rdf_cutoff / rdf_bins`.
    #[serde(default = "default_rdf_bins")]
    pub rdf_bins: usize,

    /// Maximum pair distance included in the RDF, in simulation length units.
    /// On multi-process runs this is automatically capped to the ghost cutoff
    /// to avoid missing pairs.
    #[serde(default = "default_rdf_cutoff")]
    pub rdf_cutoff: f64,

    /// Accumulate one RDF histogram sample every this many timesteps.
    /// Set to 0 to disable RDF sampling.
    #[serde(default = "default_rdf_interval")]
    pub rdf_interval: usize,

    /// Record MSD and pressure every this many timesteps. Also controls the
    /// virial stress accumulation interval. Set to 0 to disable MSD tracking.
    #[serde(default = "default_msd_interval")]
    pub msd_interval: usize,

    /// Write all measurement output files (`rdf.txt`, `msd.txt`,
    /// `pressure.txt`) every this many timesteps.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        MeasureConfig {
            rdf_bins: 200,
            rdf_cutoff: 3.0,
            rdf_interval: 100,
            msd_interval: 10,
            output_interval: 1000,
        }
    }
}

// ── Resources ───────────────────────────────────────────────────────────────

/// Accumulates radial distribution function g(r) histogram samples over time.
///
/// The RDF measures how particle density varies with distance from a reference
/// particle, relative to an ideal gas at the same overall density. For a
/// homogeneous liquid, g(r) → 1 at large r.
///
/// # Formula
///
/// For each histogram bin k covering the spherical shell [r_k, r_{k+1}):
///
/// ```text
/// g(r_k) = (count_k × V) / (N_pairs × V_shell_k)
/// ```
///
/// where:
/// - `count_k` = number of pairs with separation in [r_k, r_{k+1})
/// - `V` = simulation box volume
/// - `N_pairs` = N(N−1)/2 total possible pairs
/// - `V_shell_k` = (4π/3)(r_{k+1}³ − r_k³), the spherical shell volume
pub struct RdfAccumulator {
    /// Cumulative (unnormalized) g(r) histogram; divide by `n_samples` for the
    /// time-averaged result.
    pub bins: Vec<f64>,
    /// Number of RDF samples accumulated so far.
    pub n_samples: usize,
    /// Bin width: `cutoff / n_bins`.
    pub dr: f64,
    /// Maximum pair distance for RDF sampling.
    pub cutoff: f64,
}

impl RdfAccumulator {
    fn new(n_bins: usize, cutoff: f64) -> Self {
        RdfAccumulator {
            bins: vec![0.0; n_bins],
            n_samples: 0,
            dr: cutoff / n_bins as f64,
            cutoff,
        }
    }
}

/// Tracks particle positions for mean square displacement (MSD) calculation.
///
/// MSD measures the average squared distance particles have moved from their
/// initial (reference) positions, using unwrapped coordinates to correctly
/// handle periodic boundary crossings.
///
/// # Formula
///
/// ```text
/// MSD(t) = (1/N) Σᵢ |r_unwrap_i(t) − r_ref_i|²
/// ```
///
/// where `r_unwrap` is the unwrapped (continuous) position that accumulates
/// boundary-crossing corrections, and `r_ref` is the position when tracking
/// began.
///
/// # Unwrapping Algorithm
///
/// Each timestep, the displacement `Δr = r(t) − r(t−1)` is computed. If any
/// component exceeds half the box length, a periodic image correction is
/// applied (e.g., if `Δx > L_x/2`, subtract `L_x`). The corrected
/// displacement is added to the unwrapped position, giving a continuous
/// trajectory even for atoms that cross periodic boundaries.
pub struct MsdTracker {
    /// Reference x-positions at the start of tracking (indexed by atom tag).
    pub ref_x: Vec<f64>,
    /// Reference y-positions at the start of tracking (indexed by atom tag).
    pub ref_y: Vec<f64>,
    /// Reference z-positions at the start of tracking (indexed by atom tag).
    pub ref_z: Vec<f64>,
    /// Unwrapped (continuous) x-positions, corrected for PBC crossings.
    pub unwrapped_x: Vec<f64>,
    /// Unwrapped (continuous) y-positions, corrected for PBC crossings.
    pub unwrapped_y: Vec<f64>,
    /// Unwrapped (continuous) z-positions, corrected for PBC crossings.
    pub unwrapped_z: Vec<f64>,
    /// Previous-step x-positions for detecting PBC jumps.
    pub prev_x: Vec<f64>,
    /// Previous-step y-positions for detecting PBC jumps.
    pub prev_y: Vec<f64>,
    /// Previous-step z-positions for detecting PBC jumps.
    pub prev_z: Vec<f64>,
    /// Whether each tag slot has been initialized with position data.
    pub has_entry: Vec<bool>,
    /// The timestep at which reference positions were recorded.
    pub ref_step: usize,
    /// Accumulated MSD values: `(elapsed_steps, msd_average)`.
    pub msd_values: Vec<(usize, f64)>,
    /// Whether the tracker has been initialized with reference positions.
    pub initialized: bool,
    /// Total number of atoms being tracked (across all ranks).
    pub n_tracked: f64,
}

impl Default for MsdTracker {
    fn default() -> Self {
        MsdTracker {
            ref_x: Vec::new(),
            ref_y: Vec::new(),
            ref_z: Vec::new(),
            unwrapped_x: Vec::new(),
            unwrapped_y: Vec::new(),
            unwrapped_z: Vec::new(),
            prev_x: Vec::new(),
            prev_y: Vec::new(),
            prev_z: Vec::new(),
            has_entry: Vec::new(),
            ref_step: 0,
            msd_values: Vec::new(),
            initialized: false,
            n_tracked: 0.0,
        }
    }
}

/// Time series of instantaneous virial pressure values.
///
/// Each entry is `(timestep, pressure)` where pressure is computed from the
/// virial equation of state:
///
/// ```text
/// P = ρT − Tr(W) / (3V) + P_tail
/// ```
///
/// where:
/// - `ρ = N/V` is the number density
/// - `T = 2 KE / (3N − 3)` is the instantaneous temperature
/// - `Tr(W)` is the trace of the virial stress tensor
/// - `P_tail` is the optional LJ tail correction for truncated potentials
#[derive(Default)]
pub struct PressureHistory {
    /// Recorded `(timestep, pressure)` pairs.
    pub values: Vec<(usize, f64)>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that registers RDF, MSD, and virial pressure measurement systems.
///
/// Reads configuration from the `[measure]` TOML section and registers four
/// systems in `PostFinalIntegration`: RDF accumulation, MSD tracking, pressure
/// computation, and periodic file output.
pub struct MeasurePlugin;

impl Plugin for MeasurePlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[measure]
rdf_bins = 200
rdf_cutoff = 3.0
rdf_interval = 100
msd_interval = 10
output_interval = 1000"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = Config::load::<MeasureConfig>(app, "measure");

        app.add_resource(RdfAccumulator::new(config.rdf_bins, config.rdf_cutoff))
            .add_resource(MsdTracker::default())
            .add_resource(PressureHistory::default())
            .add_setup_system(setup_measure_virial, ScheduleSetupSet::PostSetup)
            .add_update_system(accumulate_rdf, ScheduleSet::PostFinalIntegration)
            .add_update_system(track_msd, ScheduleSet::PostFinalIntegration)
            .add_update_system(compute_pressure, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_measurements, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Configures the virial stress accumulation interval to match `msd_interval`.
pub fn setup_measure_virial(
    config: Res<MeasureConfig>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    if config.msd_interval > 0 {
        if let Some(ref mut v) = virial {
            v.set_interval(config.msd_interval);
        }
    }
}

/// Accumulates one RDF histogram sample using brute-force pair counting.
///
/// On each sampling step, iterates over all local–local and local–ghost pairs,
/// bins pair distances into a histogram, then normalizes by the ideal-gas
/// shell volume to produce g(r). Results are accumulated into [`RdfAccumulator`]
/// and time-averaged when written to disk.
pub fn accumulate_rdf(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    run_state: Res<RunState>,
    config: Res<MeasureConfig>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut rdf: ResMut<RdfAccumulator>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.rdf_interval) {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let n_bins = rdf.bins.len();
    let dr = rdf.dr;

    // Cap RDF cutoff at ghost_cutoff on multi-proc to avoid missing pairs
    // that lie beyond the ghost communication range.
    let effective_cutoff = if comm.size() > 1 && neighbor.ghost_cutoff > 0.0 {
        if rdf.cutoff > neighbor.ghost_cutoff && step == config.rdf_interval {
            if comm.rank() == 0 {
                println!(
                    "WARNING: RDF cutoff ({:.3}) > ghost_cutoff ({:.3}). Capping to ghost_cutoff.",
                    rdf.cutoff, neighbor.ghost_cutoff
                );
            }
        }
        rdf.cutoff.min(neighbor.ghost_cutoff)
    } else {
        rdf.cutoff
    };
    let cutoff2 = effective_cutoff * effective_cutoff;

    let mut local_hist = vec![0.0f64; n_bins];

    // Box dimensions for minimum-image convention
    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;
    let total = atoms.len();

    for i in 0..nlocal {
        // Local-local pairs: use i < j to count each pair exactly once.
        for j in (i + 1)..nlocal {
            let mut dx = atoms.pos[j][0] - atoms.pos[i][0];
            let mut dy = atoms.pos[j][1] - atoms.pos[i][1];
            let mut dz = atoms.pos[j][2] - atoms.pos[i][2];

            // Apply minimum-image convention for periodic axes
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
            if r2 >= cutoff2 || r2 < 1e-20 { continue; }
            let bin = (r2.sqrt() / dr) as usize;
            if bin < n_bins { local_hist[bin] += 1.0; }
        }
        // Local-ghost pairs: weight 0.5 because each cross-boundary pair is
        // seen by both owning ranks. On single-process runs the minimum-image
        // local-local loop already finds all pairs, so ghost pairs are skipped.
        if comm.size() > 1 {
            for j in nlocal..total {
                let dx = atoms.pos[j][0] - atoms.pos[i][0];
                let dy = atoms.pos[j][1] - atoms.pos[i][1];
                let dz = atoms.pos[j][2] - atoms.pos[i][2];

                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 || r2 < 1e-20 { continue; }
                let bin = (r2.sqrt() / dr) as usize;
                if bin < n_bins { local_hist[bin] += 0.5; }
            }
        }
    }

    // Reduce per-rank histograms into a global histogram
    for val in &mut local_hist {
        *val = comm.all_reduce_sum_f64(*val);
    }

    let n_total = comm.all_reduce_sum_f64(nlocal as f64);
    let v = domain.volume;

    // ── RDF histogram normalization ──
    //
    // The raw histogram count in bin k tells us how many pairs have
    // separation r in [k·dr, (k+1)·dr). To convert this into g(r) we
    // divide by the number of pairs that an *ideal gas* at the same
    // density would have in that shell:
    //
    //   g(r_k) = hist[k] × V / (N_pairs × V_shell_k)
    //
    // where:
    //   N_pairs = N(N-1)/2  — total distinct pairs
    //   V_shell_k = (4π/3)(r_{k+1}³ − r_k³) — volume of the spherical shell
    //
    // This normalization ensures g(r) → 1 for a uniform (ideal gas)
    // distribution.
    let n_pairs = n_total * (n_total - 1.0) / 2.0;
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

/// Tracks mean square displacement using unwrapped coordinates.
///
/// On the first call with atoms present, records reference positions. On every
/// subsequent step, updates unwrapped positions by detecting periodic boundary
/// crossings. At each `msd_interval` step, computes the ensemble-averaged MSD.
pub fn track_msd(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    config: Res<MeasureConfig>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut msd: ResMut<MsdTracker>,
) {
    if config.msd_interval == 0 {
        return;
    }

    let step = run_state.total_cycle;
    let nlocal = atoms.nlocal as usize;

    if !msd.initialized && nlocal > 0 {
        // Find max atom tag to size the tag-indexed dense arrays
        let max_tag = atoms.tag[..nlocal].iter().cloned().max().unwrap_or(0) as usize;
        let size = max_tag + 1;
        msd.ref_x.resize(size, 0.0);
        msd.ref_y.resize(size, 0.0);
        msd.ref_z.resize(size, 0.0);
        msd.unwrapped_x.resize(size, 0.0);
        msd.unwrapped_y.resize(size, 0.0);
        msd.unwrapped_z.resize(size, 0.0);
        msd.prev_x.resize(size, 0.0);
        msd.prev_y.resize(size, 0.0);
        msd.prev_z.resize(size, 0.0);
        msd.has_entry.resize(size, false);

        for i in 0..nlocal {
            let idx = atoms.tag[i] as usize;
            msd.ref_x[idx] = atoms.pos[i][0];
            msd.ref_y[idx] = atoms.pos[i][1];
            msd.ref_z[idx] = atoms.pos[i][2];
            msd.unwrapped_x[idx] = atoms.pos[i][0];
            msd.unwrapped_y[idx] = atoms.pos[i][1];
            msd.unwrapped_z[idx] = atoms.pos[i][2];
            msd.prev_x[idx] = atoms.pos[i][0];
            msd.prev_y[idx] = atoms.pos[i][1];
            msd.prev_z[idx] = atoms.pos[i][2];
            msd.has_entry[idx] = true;
        }
        msd.n_tracked = comm.all_reduce_sum_f64(nlocal as f64);
        msd.ref_step = step;
        msd.initialized = true;
        if comm.rank() == 0 {
            msd.msd_values.push((0, 0.0));
        }
        return;
    }

    if !msd.initialized {
        return;
    }

    // ── Unwrapped position update ──
    //
    // Atoms in periodic simulations are wrapped back into the box when they
    // cross a boundary, creating discontinuous jumps in their stored
    // positions. To compute correct displacements we maintain "unwrapped"
    // coordinates that remove these jumps:
    //
    //   1. Compute Δr = r(t) − r(t−1) using the raw (wrapped) positions.
    //   2. If |Δr_α| > L_α/2 for any axis α, the atom crossed a periodic
    //      boundary — correct by subtracting/adding L_α.
    //   3. Add the corrected Δr to the unwrapped position.
    //
    // This gives a continuous trajectory that faithfully tracks how far each
    // atom has actually traveled, even across multiple box-crossings.
    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;

    for i in 0..nlocal {
        let idx = atoms.tag[i] as usize;
        // Grow arrays if a new tag exceeds current size (e.g. atom migrated in)
        if idx >= msd.prev_x.len() {
            let new_size = idx + 1;
            msd.ref_x.resize(new_size, 0.0);
            msd.ref_y.resize(new_size, 0.0);
            msd.ref_z.resize(new_size, 0.0);
            msd.unwrapped_x.resize(new_size, 0.0);
            msd.unwrapped_y.resize(new_size, 0.0);
            msd.unwrapped_z.resize(new_size, 0.0);
            msd.prev_x.resize(new_size, 0.0);
            msd.prev_y.resize(new_size, 0.0);
            msd.prev_z.resize(new_size, 0.0);
            msd.has_entry.resize(new_size, false);
        }

        if msd.has_entry[idx] {
            // Step displacement in wrapped coordinates
            let mut dx = atoms.pos[i][0] - msd.prev_x[idx];
            let mut dy = atoms.pos[i][1] - msd.prev_y[idx];
            let mut dz = atoms.pos[i][2] - msd.prev_z[idx];

            // Correct for periodic boundary crossing: a jump of more than
            // half the box length means the atom wrapped around.
            if dx > half_lx { dx -= lx; } else if dx < -half_lx { dx += lx; }
            if dy > half_ly { dy -= ly; } else if dy < -half_ly { dy += ly; }
            if dz > half_lz { dz -= lz; } else if dz < -half_lz { dz += lz; }

            msd.unwrapped_x[idx] += dx;
            msd.unwrapped_y[idx] += dy;
            msd.unwrapped_z[idx] += dz;
        }
        msd.prev_x[idx] = atoms.pos[i][0];
        msd.prev_y[idx] = atoms.pos[i][1];
        msd.prev_z[idx] = atoms.pos[i][2];
        msd.has_entry[idx] = true;
    }

    if step == 0 || !step.is_multiple_of(config.msd_interval) {
        return;
    }

    // Compute MSD = (1/N) Σᵢ |r_unwrap(t) − r_ref|² for tracked atoms
    let mut local_msd_sum = 0.0;
    for i in 0..nlocal {
        let idx = atoms.tag[i] as usize;
        if idx < msd.has_entry.len() && msd.has_entry[idx] {
            let dx = msd.unwrapped_x[idx] - msd.ref_x[idx];
            let dy = msd.unwrapped_y[idx] - msd.ref_y[idx];
            let dz = msd.unwrapped_z[idx] - msd.ref_z[idx];
            local_msd_sum += dx * dx + dy * dy + dz * dz;
        }
    }
    let global_msd_sum = comm.all_reduce_sum_f64(local_msd_sum);
    let msd_avg = if msd.n_tracked > 0.0 { global_msd_sum / msd.n_tracked } else { 0.0 };

    if comm.rank() == 0 {
        let ref_step = msd.ref_step;
        msd.msd_values.push((step - ref_step, msd_avg));
    }
}

/// Computes the instantaneous virial pressure.
///
/// Uses the virial equation of state: `P = ρT − Tr(W)/(3V) + P_tail`,
/// where the temperature is computed from the kinetic energy and the virial
/// trace comes from the [`VirialStress`] resource (if available).
#[allow(clippy::too_many_arguments)]
pub fn compute_pressure(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    virial: Option<Res<VirialStress>>,
    tails: Option<Res<LJTailCorrections>>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    config: Res<MeasureConfig>,
    mut pressure_hist: ResMut<PressureHistory>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.msd_interval) {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let n = comm.all_reduce_sum_f64(nlocal as f64);
    let v = domain.volume;
    let rho = n / v;

    // Instantaneous temperature from equipartition: T = 2 KE / N_dof
    let local_ke: f64 = (0..nlocal)
        .map(|i| {
            atoms.mass[i]
                * (atoms.vel[i][0].powi(2) + atoms.vel[i][1].powi(2) + atoms.vel[i][2].powi(2))
        })
        .sum::<f64>()
        * 0.5;
    let global_ke = comm.all_reduce_sum_f64(local_ke);
    let ndof = 3.0 * n - 3.0;
    let temp = if ndof > 0.0 { 2.0 * global_ke / ndof } else { 0.0 };

    // Virial pressure: P = ρT − Tr(W)/(3V) + P_tail
    let global_trace = match virial {
        Some(ref v) => {
            let local_trace = v.trace();
            comm.all_reduce_sum_f64(local_trace)
        }
        None => 0.0,
    };
    let tail = tails.map_or(0.0, |t| t.pressure_tail);
    let pressure = rho * temp - global_trace / (3.0 * v) + tail;

    if comm.rank() == 0 {
        pressure_hist.values.push((step, pressure));
    }
}

/// Periodically writes RDF, MSD, and pressure data to text files.
///
/// Output is written to `<output_dir>/data/` every `output_interval` steps,
/// only on rank 0. See the [module-level docs](crate) for file format details.
pub fn write_measurements(
    run_state: Res<RunState>,
    config: Res<MeasureConfig>,
    rdf: Res<RdfAccumulator>,
    msd: Res<MsdTracker>,
    pressure_hist: Res<PressureHistory>,
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

    let base_dir = input
        .output_dir
        .as_deref()
        .unwrap_or(".");
    let data_dir = format!("{}/data", base_dir);
    let _ = fs::create_dir_all(&data_dir);

    // Write RDF: two-column format "r g(r)" with bin-center r values
    if rdf.n_samples > 0 {
        let path = format!("{}/rdf.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# r g(r) (averaged over {} samples)", rdf.n_samples).ok();
            let n_bins = rdf.bins.len();
            for k in 0..n_bins {
                let r = (k as f64 + 0.5) * rdf.dr;
                let gr = rdf.bins[k] / rdf.n_samples as f64;
                writeln!(f, "{:.6} {:.6}", r, gr).ok();
            }
            println!("Wrote RDF to {}", path);
        }
    }

    // Write MSD: two-column format "dt MSD"
    if !msd.msd_values.is_empty() {
        let path = format!("{}/msd.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# dt MSD").ok();
            for &(dt, msd_val) in &msd.msd_values {
                writeln!(f, "{} {:.6}", dt, msd_val).ok();
            }
            println!("Wrote MSD to {}", path);
        }
    }

    // Write pressure: two-column format "step pressure"
    if !pressure_hist.values.is_empty() {
        let path = format!("{}/pressure.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# step pressure").ok();
            for &(s, p) in &pressure_hist.values {
                writeln!(f, "{} {:.6}", s, p).ok();
            }
            println!("Wrote pressure to {}", path);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msd_zero_at_start() {
        let msd = MsdTracker::default();
        assert!(msd.msd_values.is_empty());
    }

    #[test]
    fn rdf_accumulator_init() {
        let rdf = RdfAccumulator::new(100, 3.0);
        assert_eq!(rdf.bins.len(), 100);
        assert!((rdf.dr - 0.03).abs() < 1e-10);
        assert_eq!(rdf.n_samples, 0);
    }
}
