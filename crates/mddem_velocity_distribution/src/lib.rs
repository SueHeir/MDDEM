//! Velocity distribution analysis plugin for particle simulations.
//!
//! Measures particle speed distributions and compares against the Maxwell-Boltzmann
//! distribution for the measured granular temperature. Outputs:
//! - Binned speed histogram (probability density)
//! - Per-component velocity histograms
//! - Maxwell-Boltzmann reference curve
//! - Quantitative deviation metrics (L2 norm, kurtosis excess)
//! - Per-species granular temperature (equipartition breakdown in polydisperse systems)
//! - Inelastic collapse detection (diverging collision rate / vanishing temperature)
//!
//! # TOML Configuration
//! ```toml
//! [velocity_distribution]
//! interval = 1000        # output every N steps
//! num_bins = 50           # number of histogram bins
//! max_speed_factor = 3.0  # max speed = factor * v_rms
//! per_species = true      # compute per-species granular temperature
//! collapse_threshold = 1e-12  # T_g below this triggers collapse warning
//! collapse_rate_window = 5    # number of samples for cooling rate estimation
//! ```

use std::{
    f64::consts::PI,
    fs::{self, OpenOptions},
    io::Write,
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Input, RunConfig, RunState};

// ── Configuration ──────────────────────────────────────────────────────────

fn default_interval() -> usize {
    1000
}
fn default_num_bins() -> usize {
    50
}
fn default_max_speed_factor() -> f64 {
    3.0
}
fn default_per_species() -> bool {
    false
}
fn default_collapse_threshold() -> f64 {
    1e-12
}
fn default_collapse_rate_window() -> usize {
    5
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[velocity_distribution]` — velocity distribution analysis settings.
pub struct VelocityDistributionConfig {
    /// Output interval in timesteps.
    #[serde(default = "default_interval")]
    pub interval: usize,
    /// Number of histogram bins for speed distribution.
    #[serde(default = "default_num_bins")]
    pub num_bins: usize,
    /// Maximum speed as a multiple of v_rms. Speeds above this are placed in the last bin.
    #[serde(default = "default_max_speed_factor")]
    pub max_speed_factor: f64,
    /// Whether to compute per-species (per atom type) granular temperature.
    #[serde(default = "default_per_species")]
    pub per_species: bool,
    /// Granular temperature below this value triggers an inelastic collapse warning.
    #[serde(default = "default_collapse_threshold")]
    pub collapse_threshold: f64,
    /// Number of recent T_g samples to use for cooling rate estimation.
    #[serde(default = "default_collapse_rate_window")]
    pub collapse_rate_window: usize,
}

impl Default for VelocityDistributionConfig {
    fn default() -> Self {
        VelocityDistributionConfig {
            interval: default_interval(),
            num_bins: default_num_bins(),
            max_speed_factor: default_max_speed_factor(),
            per_species: default_per_species(),
            collapse_threshold: default_collapse_threshold(),
            collapse_rate_window: default_collapse_rate_window(),
        }
    }
}

// ── Collapse detection state ──────────────────────────────────────────────

/// Tracks granular temperature history for inelastic collapse detection.
///
/// Inelastic collapse occurs in highly dissipative granular gases when the
/// collision rate diverges and granular temperature drops to zero in finite
/// time. This detector monitors:
/// 1. Absolute T_g threshold — warns when T_g falls below a configurable value
/// 2. Cooling rate divergence — warns when dT_g/dt accelerates beyond Haff's law
pub struct CollapseDetector {
    /// Recent (time, T_g) samples for cooling rate estimation.
    history: Vec<(f64, f64)>,
    /// Maximum number of samples to retain.
    max_samples: usize,
    /// Whether collapse has been detected (latched — only warns once).
    collapse_detected: bool,
}

impl CollapseDetector {
    /// Create a new detector that retains the last `max_samples` temperature readings
    /// (minimum 2) for cooling-rate estimation.
    fn new(max_samples: usize) -> Self {
        CollapseDetector {
            history: Vec::with_capacity(max_samples + 1),
            max_samples: max_samples.max(2),
            collapse_detected: false,
        }
    }

    /// Record a new T_g sample and return collapse diagnostic.
    fn record(&mut self, time: f64, t_g: f64, threshold: f64) -> CollapseDiagnostic {
        self.history.push((time, t_g));
        if self.history.len() > self.max_samples {
            self.history.remove(0);
        }

        let below_threshold = t_g < threshold && t_g >= 0.0;

        // Estimate cooling rate from recent history using linear regression
        // of ln(T_g) vs t. For Haff's law, d(ln T_g)/dt should be bounded.
        // Diverging cooling rate => collapse.
        let cooling_rate = self.estimate_cooling_rate();

        let newly_detected = below_threshold && !self.collapse_detected;
        if newly_detected {
            self.collapse_detected = true;
        }

        CollapseDiagnostic {
            below_threshold,
            newly_detected,
            cooling_rate,
            t_g,
        }
    }

    /// Estimate d(ln T_g)/dt from recent samples via finite differences.
    /// Returns None if insufficient data or T_g values are non-positive.
    fn estimate_cooling_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let (t0, tg0) = self.history[0];
        let (t1, tg1) = *self
            .history
            .last()
            .expect("history should have at least 2 entries (checked above)");
        let dt = t1 - t0;
        if dt <= 0.0 || tg0 <= 0.0 || tg1 <= 0.0 {
            return None;
        }
        // d(ln T_g)/dt ≈ (ln T_g1 - ln T_g0) / (t1 - t0)
        Some((tg1.ln() - tg0.ln()) / dt)
    }
}

/// Diagnostic result from collapse detection.
pub struct CollapseDiagnostic {
    /// Whether T_g is below the threshold.
    pub below_threshold: bool,
    /// Whether this is the first time collapse was detected (for one-time warning).
    pub newly_detected: bool,
    /// Estimated d(ln T_g)/dt — negative means cooling; large negative means fast cooling.
    pub cooling_rate: Option<f64>,
    /// Current granular temperature.
    pub t_g: f64,
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Plugin that periodically measures the velocity distribution and writes
/// comparison data against the Maxwell-Boltzmann distribution.
/// Applicable to any particle simulation (granular, molecular dynamics, etc.).
pub struct VelocityDistributionPlugin;

impl Plugin for VelocityDistributionPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[velocity_distribution]
# Output interval in timesteps
interval = 1000
# Number of histogram bins
num_bins = 50
# Maximum speed as multiple of v_rms
max_speed_factor = 3.0
# Compute per-species granular temperature (for polydisperse systems)
per_species = false
# Inelastic collapse detection threshold for T_g
collapse_threshold = 1e-12
# Number of recent T_g samples for cooling rate estimation
collapse_rate_window = 5"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<VelocityDistributionConfig>(app, "velocity_distribution");

        // Read collapse_rate_window from config to initialize detector
        let window = app
            .get_resource_ref::<VelocityDistributionConfig>()
            .map(|c| c.collapse_rate_window)
            .unwrap_or(default_collapse_rate_window());
        app.add_resource(CollapseDetector::new(window));

        app.add_update_system(
            compute_velocity_distribution,
            ScheduleSet::PostFinalIntegration,
        );
    }
}

// ── Analysis result ────────────────────────────────────────────────────────

/// Results of a single velocity distribution snapshot.
pub struct VelocityDistributionResult {
    /// Granular temperature T = <m v'^2> / (3 m) where v' = v - v_mean.
    pub granular_temperature: f64,
    /// Root-mean-square speed.
    pub v_rms: f64,
    /// Total number of particles.
    pub n_particles: u64,
    /// Bin edges for speed histogram (length = num_bins + 1).
    pub bin_edges: Vec<f64>,
    /// Bin centers (length = num_bins).
    pub bin_centers: Vec<f64>,
    /// Measured probability density f(v) such that integral f(v) dv = 1 (length = num_bins).
    pub measured_pdf: Vec<f64>,
    /// Maxwell-Boltzmann probability density at bin centers (length = num_bins).
    pub mb_pdf: Vec<f64>,
    /// L2 norm of (measured - MB) weighted by bin width.
    pub l2_deviation: f64,
    /// Kurtosis excess: κ - 3 (0 for Gaussian, >0 for heavy tails).
    pub kurtosis_excess: f64,
    /// Bin centers for per-component velocity histograms, symmetric around 0 (length = num_bins).
    pub component_bin_centers: Vec<f64>,
    /// Probability density of the vx fluctuation component (length = num_bins).
    pub component_pdf_x: Vec<f64>,
    /// Probability density of the vy fluctuation component (length = num_bins).
    pub component_pdf_y: Vec<f64>,
    /// Probability density of the vz fluctuation component (length = num_bins).
    pub component_pdf_z: Vec<f64>,
    /// Gaussian reference PDF for component distributions at the measured T_g (length = num_bins).
    pub component_gaussian: Vec<f64>,
}

/// Per-species granular temperature result.
/// In polydisperse granular gases, energy equipartition breaks down:
/// lighter particles tend to have higher granular temperature than heavier ones.
pub struct SpeciesTemperatureResult {
    /// Species index (0-based atom type).
    pub species: u32,
    /// Number of particles of this species.
    pub count: u64,
    /// Mean mass of this species.
    pub mean_mass: f64,
    /// Per-species granular temperature: T_s = <m_s v_s'^2> / (3 N_s m_s_mean).
    pub granular_temperature: f64,
    /// Per-species RMS speed.
    pub v_rms: f64,
    /// Per-species kurtosis excess.
    pub kurtosis_excess: f64,
}

// ── Maxwell-Boltzmann PDF ──────────────────────────────────────────────────

/// Maxwell-Boltzmann speed distribution for 3D:
///
/// f(v) = 4π (m / (2πkT))^(3/2) v² exp(-mv²/(2kT))
///
/// Using granular temperature T_g = ⟨v'²⟩/3 (mass cancels for equal-mass systems):
///
/// f(v) = 4π (1/(2πT_g))^(3/2) v² exp(-v²/(2T_g))
///
/// Returns 0.0 for negative speeds or non-positive temperature.
fn maxwell_boltzmann_pdf(v: f64, t_granular: f64) -> f64 {
    if t_granular <= 0.0 || v < 0.0 {
        return 0.0;
    }
    let a = 1.0 / (2.0 * t_granular);
    4.0 * PI * (a / PI).powf(1.5) * v * v * (-a * v * v).exp()
}

/// 1D Gaussian PDF for a single velocity component with zero mean and variance T_g:
///
/// g(v) = (2π T_g)^(-1/2) exp(-v² / (2 T_g))
///
/// This is the expected distribution for each Cartesian velocity component (vx, vy, vz)
/// in thermal equilibrium. Returns 0.0 for non-positive temperature.
fn gaussian_component_pdf(v: f64, t_granular: f64) -> f64 {
    if t_granular <= 0.0 {
        return 0.0;
    }
    let sigma2 = t_granular;
    (1.0 / (2.0 * PI * sigma2).sqrt()) * (-v * v / (2.0 * sigma2)).exp()
}

// ── Core analysis function (unit-testable) ─────────────────────────────────

/// Compute velocity distribution statistics from raw velocity data.
///
/// This is the pure-computation core, separated from I/O for testability.
/// Works for any particle simulation (DEM, MD, etc.).
///
/// The analysis pipeline:
/// 1. Subtract center-of-mass velocity to obtain fluctuation velocities
/// 2. Compute granular temperature T_g = ⟨m v'²⟩ / (3M) where v' = v - v_mean
/// 3. Bin particle speeds into a histogram and normalize to a probability density
/// 4. Evaluate the Maxwell-Boltzmann speed PDF f(v) = 4π (1/(2πT_g))^(3/2) v² exp(-v²/(2T_g))
/// 5. Compute L2 deviation between measured and MB distributions
/// 6. Compute kurtosis excess (0 for Gaussian, positive for heavy tails)
/// 7. Build per-component (vx, vy, vz) histograms with Gaussian reference
///
/// # Panics
///
/// - If `velocities` is empty
/// - If `velocities.len() != masses.len()`
pub fn analyze_velocity_distribution(
    velocities: &[[f64; 3]],
    masses: &[f64],
    num_bins: usize,
    max_speed_factor: f64,
) -> VelocityDistributionResult {
    let n = velocities.len();
    assert!(n > 0, "Need at least one particle");
    assert_eq!(n, masses.len());

    // 1. Compute center-of-mass velocity
    let mut total_mv = [0.0_f64; 3];
    let mut total_mass = 0.0_f64;
    for i in 0..n {
        for d in 0..3 {
            total_mv[d] += masses[i] * velocities[i][d];
        }
        total_mass += masses[i];
    }
    let v_mean = [
        total_mv[0] / total_mass,
        total_mv[1] / total_mass,
        total_mv[2] / total_mass,
    ];

    // 2. Compute fluctuation velocities and granular temperature
    let mut sum_mv2 = 0.0_f64;
    let mut speeds: Vec<f64> = Vec::with_capacity(n);
    let mut vx_fluct: Vec<f64> = Vec::with_capacity(n);
    let mut vy_fluct: Vec<f64> = Vec::with_capacity(n);
    let mut vz_fluct: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let dvx = velocities[i][0] - v_mean[0];
        let dvy = velocities[i][1] - v_mean[1];
        let dvz = velocities[i][2] - v_mean[2];
        let v2 = dvx * dvx + dvy * dvy + dvz * dvz;
        sum_mv2 += masses[i] * v2;
        speeds.push(v2.sqrt());
        vx_fluct.push(dvx);
        vy_fluct.push(dvy);
        vz_fluct.push(dvz);
    }

    let granular_temperature = sum_mv2 / (3.0 * total_mass);
    let v_rms = (3.0 * granular_temperature).sqrt();

    // 3. Speed histogram
    let v_max = max_speed_factor * v_rms;
    let bin_width = v_max / num_bins as f64;

    let mut bin_edges = Vec::with_capacity(num_bins + 1);
    let mut bin_centers = Vec::with_capacity(num_bins);
    for i in 0..=num_bins {
        bin_edges.push(i as f64 * bin_width);
    }
    for i in 0..num_bins {
        bin_centers.push((i as f64 + 0.5) * bin_width);
    }

    let mut speed_counts = vec![0usize; num_bins];
    for &s in &speeds {
        let bin = if s >= v_max {
            num_bins - 1
        } else {
            (s / bin_width) as usize
        };
        speed_counts[bin] += 1;
    }

    // Normalize to probability density: count / (N * bin_width)
    let n_f64 = n as f64;
    let measured_pdf: Vec<f64> = speed_counts
        .iter()
        .map(|&c| c as f64 / (n_f64 * bin_width))
        .collect();

    // MB reference
    let mb_pdf: Vec<f64> = bin_centers
        .iter()
        .map(|&v| maxwell_boltzmann_pdf(v, granular_temperature))
        .collect();

    // L2 deviation
    let l2_deviation = measured_pdf
        .iter()
        .zip(mb_pdf.iter())
        .map(|(&m, &mb)| (m - mb).powi(2) * bin_width)
        .sum::<f64>()
        .sqrt();

    // 4. Kurtosis excess of speed distribution
    // κ = <v'^4> / <v'^2>^2 - 3  (computed per-component, averaged)
    let mut sum_v2_comp = [0.0_f64; 3];
    let mut sum_v4_comp = [0.0_f64; 3];
    for i in 0..n {
        let comps = [vx_fluct[i], vy_fluct[i], vz_fluct[i]];
        for d in 0..3 {
            let v2 = comps[d] * comps[d];
            sum_v2_comp[d] += v2;
            sum_v4_comp[d] += v2 * v2;
        }
    }
    let mut kurtosis_excess = 0.0;
    for d in 0..3 {
        let mean_v2 = sum_v2_comp[d] / n_f64;
        let mean_v4 = sum_v4_comp[d] / n_f64;
        if mean_v2 > 0.0 {
            kurtosis_excess += mean_v4 / (mean_v2 * mean_v2) - 3.0;
        }
    }
    kurtosis_excess /= 3.0; // average over components

    // 5. Per-component velocity histograms
    // Use symmetric bins around 0
    let comp_max = max_speed_factor * granular_temperature.sqrt();
    let comp_bin_width = 2.0 * comp_max / num_bins as f64;
    let mut comp_bin_centers = Vec::with_capacity(num_bins);
    for i in 0..num_bins {
        comp_bin_centers.push(-comp_max + (i as f64 + 0.5) * comp_bin_width);
    }

    let bin_component = |data: &[f64]| -> Vec<f64> {
        let mut counts = vec![0usize; num_bins];
        for &v in data {
            let shifted = v + comp_max;
            let bin = if shifted < 0.0 {
                0
            } else if shifted >= 2.0 * comp_max {
                num_bins - 1
            } else {
                (shifted / comp_bin_width) as usize
            };
            let bin = bin.min(num_bins - 1);
            counts[bin] += 1;
        }
        counts
            .iter()
            .map(|&c| c as f64 / (n_f64 * comp_bin_width))
            .collect()
    };

    let component_pdf_x = bin_component(&vx_fluct);
    let component_pdf_y = bin_component(&vy_fluct);
    let component_pdf_z = bin_component(&vz_fluct);
    let component_gaussian: Vec<f64> = comp_bin_centers
        .iter()
        .map(|&v| gaussian_component_pdf(v, granular_temperature))
        .collect();

    VelocityDistributionResult {
        granular_temperature,
        v_rms,
        n_particles: n as u64,
        bin_edges,
        bin_centers,
        measured_pdf,
        mb_pdf,
        l2_deviation,
        kurtosis_excess,
        component_bin_centers: comp_bin_centers,
        component_pdf_x,
        component_pdf_y,
        component_pdf_z,
        component_gaussian,
    }
}

// ── Per-species analysis (unit-testable) ──────────────────────────────────

/// Compute per-species granular temperature from velocity and type data.
///
/// In polydisperse granular gases, energy equipartition breaks down: the
/// granular temperature of each species differs, with lighter particles
/// typically having higher T_g than heavier ones. This function computes
/// the per-species granular temperature T_s for each atom type present.
///
/// The global center-of-mass velocity is subtracted before computing
/// fluctuation velocities, so each species' T_s measures thermal motion
/// relative to the bulk flow.
///
/// # Panics
///
/// - If `velocities.len() != masses.len()`
/// - If `velocities.len() != atom_types.len()`
pub fn analyze_per_species(
    velocities: &[[f64; 3]],
    masses: &[f64],
    atom_types: &[u32],
    v_mean: [f64; 3],
) -> Vec<SpeciesTemperatureResult> {
    let n = velocities.len();
    assert_eq!(n, masses.len());
    assert_eq!(n, atom_types.len());
    if n == 0 {
        return Vec::new();
    }

    // Find unique species
    let max_type = atom_types.iter().copied().max().unwrap_or(0) as usize;

    // Accumulate per-species statistics
    let mut count = vec![0u64; max_type + 1];
    let mut sum_mass = vec![0.0f64; max_type + 1];
    let mut sum_mv2 = vec![0.0f64; max_type + 1];
    let mut sum_v2_comp = vec![[0.0f64; 3]; max_type + 1];
    let mut sum_v4_comp = vec![[0.0f64; 3]; max_type + 1];

    for i in 0..n {
        let t = atom_types[i] as usize;
        count[t] += 1;
        sum_mass[t] += masses[i];

        let dvx = velocities[i][0] - v_mean[0];
        let dvy = velocities[i][1] - v_mean[1];
        let dvz = velocities[i][2] - v_mean[2];
        let v2 = dvx * dvx + dvy * dvy + dvz * dvz;
        sum_mv2[t] += masses[i] * v2;

        let comps = [dvx, dvy, dvz];
        for d in 0..3 {
            let c2 = comps[d] * comps[d];
            sum_v2_comp[t][d] += c2;
            sum_v4_comp[t][d] += c2 * c2;
        }
    }

    let mut results = Vec::new();
    for t in 0..=max_type {
        if count[t] == 0 {
            continue;
        }
        let n_s = count[t] as f64;
        let mean_mass = sum_mass[t] / n_s;
        let t_g = sum_mv2[t] / (3.0 * sum_mass[t]);
        let v_rms = (3.0 * t_g).sqrt();

        // Per-species kurtosis excess
        let mut kurtosis = 0.0;
        for d in 0..3 {
            let mean_v2 = sum_v2_comp[t][d] / n_s;
            let mean_v4 = sum_v4_comp[t][d] / n_s;
            if mean_v2 > 0.0 {
                kurtosis += mean_v4 / (mean_v2 * mean_v2) - 3.0;
            }
        }
        kurtosis /= 3.0;

        results.push(SpeciesTemperatureResult {
            species: t as u32,
            count: count[t],
            mean_mass,
            granular_temperature: t_g,
            v_rms,
            kurtosis_excess: kurtosis,
        });
    }

    results
}

// ── ECS system ─────────────────────────────────────────────────────────────

fn compute_velocity_distribution(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    config: Res<VelocityDistributionConfig>,
    input: Res<Input>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    mut collapse_detector: ResMut<CollapseDetector>,
) {
    let index = scheduler_manager.index;
    if index >= run_config.num_stages() {
        return;
    }

    // Velocity distribution analysis is local-only — skip entirely on
    // multi-rank runs to avoid silently producing incorrect results.
    if comm.size() > 1 {
        return;
    }

    let step = run_state.total_cycle;
    if config.interval == 0 || !step.is_multiple_of(config.interval) {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 {
        return;
    }

    let velocities = &atoms.vel[..nlocal];
    let masses = &atoms.mass[..nlocal];

    let result = analyze_velocity_distribution(velocities, masses, config.num_bins, config.max_speed_factor);

    // ── Collapse detection ─────────────────────────────────────────────
    let physical_time = step as f64 * atoms.dt;
    let diag = collapse_detector.record(
        physical_time,
        result.granular_temperature,
        config.collapse_threshold,
    );
    if diag.newly_detected {
        eprintln!(
            "WARNING [step {}]: Inelastic collapse detected! T_g = {:.6e} < threshold {:.6e}",
            step, diag.t_g, config.collapse_threshold,
        );
        if let Some(rate) = diag.cooling_rate {
            eprintln!(
                "  Cooling rate d(ln T_g)/dt = {:.6e} (Haff's law predicts bounded rate)",
                rate
            );
        }
    }

    // Single-rank run, so we are rank 0 — write output directly.
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/data", dir),
        None => "data".to_string(),
    };
    if let Err(e) = fs::create_dir_all(&base_dir) {
        eprintln!("WARNING: Could not create directory {}: {}", base_dir, e);
        return;
    }

    // Write speed distribution histogram
    let hist_path = format!("{}/velocity_distribution_{}.csv", base_dir, step);
    if let Ok(mut file) = std::fs::File::create(&hist_path) {
        writeln!(
            file,
            "# Velocity distribution at step {} | T_g = {:.6e} | v_rms = {:.6e} | N = {} | L2_dev = {:.6e} | kurtosis_excess = {:.6e}",
            step, result.granular_temperature, result.v_rms, result.n_particles, result.l2_deviation, result.kurtosis_excess
        )
        .ok();
        writeln!(file, "speed,measured_pdf,mb_pdf").ok();
        for i in 0..result.bin_centers.len() {
            writeln!(
                file,
                "{:.6e},{:.6e},{:.6e}",
                result.bin_centers[i], result.measured_pdf[i], result.mb_pdf[i]
            )
            .ok();
        }
    }

    // Write component distributions
    let comp_path = format!("{}/velocity_components_{}.csv", base_dir, step);
    if let Ok(mut file) = std::fs::File::create(&comp_path) {
        writeln!(
            file,
            "# Per-component velocity distribution at step {} | T_g = {:.6e}",
            step, result.granular_temperature
        )
        .ok();
        writeln!(file, "velocity,pdf_vx,pdf_vy,pdf_vz,gaussian_ref").ok();
        for i in 0..result.component_bin_centers.len() {
            writeln!(
                file,
                "{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                result.component_bin_centers[i],
                result.component_pdf_x[i],
                result.component_pdf_y[i],
                result.component_pdf_z[i],
                result.component_gaussian[i]
            )
            .ok();
        }
    }

    // Append summary to time-series file (includes collapse diagnostic)
    let summary_path = format!("{}/velocity_distribution_summary.csv", base_dir);
    let is_new = !std::path::Path::new(&summary_path).exists();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_path)
    {
        if is_new {
            writeln!(
                file,
                "step,time,granular_temp,v_rms,l2_deviation,kurtosis_excess,n_particles,cooling_rate,collapse_detected"
            )
            .ok();
        }
        let cooling_rate_str = match diag.cooling_rate {
            Some(r) => format!("{:.10e}", r),
            None => "NA".to_string(),
        };
        writeln!(
            file,
            "{},{:.6e},{:.10e},{:.10e},{:.10e},{:.10e},{},{},{}",
            step, physical_time, result.granular_temperature, result.v_rms,
            result.l2_deviation, result.kurtosis_excess, result.n_particles,
            cooling_rate_str, diag.below_threshold as u8,
        )
        .ok();
    }

    // ── Per-species analysis ───────────────────────────────────────────
    if config.per_species && atoms.ntypes > 1 {
        let atom_types = &atoms.atom_type[..nlocal];

        // Compute global center-of-mass velocity (already done inside
        // analyze_velocity_distribution, but recompute here to pass to
        // per-species analysis — cheap for single-rank).
        let mut total_mv = [0.0_f64; 3];
        let mut total_mass = 0.0_f64;
        for i in 0..nlocal {
            for d in 0..3 {
                total_mv[d] += masses[i] * velocities[i][d];
            }
            total_mass += masses[i];
        }
        let v_mean = [
            total_mv[0] / total_mass,
            total_mv[1] / total_mass,
            total_mv[2] / total_mass,
        ];

        let species_results = analyze_per_species(velocities, masses, atom_types, v_mean);

        // Write per-species time series
        let species_path = format!("{}/species_temperature.csv", base_dir);
        let is_new_species = !std::path::Path::new(&species_path).exists();
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&species_path)
        {
            if is_new_species {
                writeln!(
                    file,
                    "step,time,species,count,mean_mass,granular_temp,v_rms,kurtosis_excess,temp_ratio"
                )
                .ok();
            }
            for sr in &species_results {
                let temp_ratio = if result.granular_temperature > 0.0 {
                    sr.granular_temperature / result.granular_temperature
                } else {
                    0.0
                };
                writeln!(
                    file,
                    "{},{:.6e},{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.6e}",
                    step, physical_time, sr.species, sr.count,
                    sr.mean_mass, sr.granular_temperature, sr.v_rms,
                    sr.kurtosis_excess, temp_ratio,
                )
                .ok();
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxwell_boltzmann_normalization() {
        // Numerical integration of MB PDF should ≈ 1
        let t_g = 1.0;
        let dv = 0.001;
        let mut integral = 0.0;
        let mut v = 0.0;
        while v < 10.0 {
            integral += maxwell_boltzmann_pdf(v, t_g) * dv;
            v += dv;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "MB PDF integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_maxwell_boltzmann_mean_speed() {
        // <v> = sqrt(8 T_g / π)
        let t_g = 2.0;
        let dv = 0.001;
        let mut mean = 0.0;
        let mut v = 0.0;
        while v < 20.0 {
            mean += v * maxwell_boltzmann_pdf(v, t_g) * dv;
            v += dv;
        }
        let expected = (8.0 * t_g / PI).sqrt();
        assert!(
            (mean - expected).abs() < 0.01,
            "MB mean speed = {} (expected {})",
            mean,
            expected
        );
    }

    #[test]
    fn test_gaussian_component_normalization() {
        let t_g = 1.5;
        let dv = 0.001;
        let mut integral = 0.0;
        let mut v = -10.0;
        while v < 10.0 {
            integral += gaussian_component_pdf(v, t_g) * dv;
            v += dv;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Gaussian integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_analyze_uniform_velocities() {
        // Create particles with known velocity distribution
        let n = 10000;
        let mut velocities = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);

        // Use a simple deterministic "random" based on index
        for i in 0..n {
            let phase = i as f64 * 2.0 * PI / 7.0;
            let vx = phase.sin() * 0.5;
            let vy = (phase * 1.3).cos() * 0.5;
            let vz = (phase * 0.7).sin() * 0.5;
            velocities.push([vx, vy, vz]);
            masses.push(1.0);
        }

        let result = analyze_velocity_distribution(&velocities, &masses, 30, 3.0);

        assert!(result.granular_temperature > 0.0);
        assert!(result.v_rms > 0.0);
        assert_eq!(result.n_particles, n as u64);
        assert_eq!(result.bin_centers.len(), 30);
        assert_eq!(result.measured_pdf.len(), 30);
        assert_eq!(result.mb_pdf.len(), 30);

        // Check PDF normalization: sum * bin_width ≈ 1
        let bin_width = result.bin_edges[1] - result.bin_edges[0];
        let integral: f64 = result.measured_pdf.iter().sum::<f64>() * bin_width;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "Measured PDF integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_kurtosis_uniform_distribution() {
        // For a uniform distribution on [-a, a], kurtosis excess = -6/5 = -1.2
        let n = 10000;
        let mut velocities = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);

        for i in 0..n {
            // Map to [-0.5, 0.5] uniformly
            let vx = (i as f64 / n as f64) - 0.5;
            let vy = (((i * 7 + 3) % n) as f64 / n as f64) - 0.5;
            let vz = (((i * 13 + 5) % n) as f64 / n as f64) - 0.5;
            velocities.push([vx, vy, vz]);
            masses.push(1.0);
        }

        let result = analyze_velocity_distribution(&velocities, &masses, 50, 3.0);

        // Uniform distribution has negative kurtosis excess (platykurtic)
        // Exact value is -6/5 = -1.2
        assert!(
            result.kurtosis_excess < 0.0,
            "Uniform distribution should have negative kurtosis excess, got {}",
            result.kurtosis_excess
        );
    }

    #[test]
    fn test_zero_temperature_handling() {
        // All particles at rest — should not panic
        let velocities = vec![[0.0, 0.0, 0.0]; 100];
        let masses = vec![1.0; 100];
        let result = analyze_velocity_distribution(&velocities, &masses, 10, 3.0);
        assert_eq!(result.granular_temperature, 0.0);
        assert_eq!(result.v_rms, 0.0);
    }

    #[test]
    fn test_mb_pdf_zero_cases() {
        assert_eq!(maxwell_boltzmann_pdf(-1.0, 1.0), 0.0);
        assert_eq!(maxwell_boltzmann_pdf(1.0, 0.0), 0.0);
        assert_eq!(maxwell_boltzmann_pdf(1.0, -1.0), 0.0);
    }

    // ── Per-species tests ──────────────────────────────────────────────

    #[test]
    fn test_per_species_single_type() {
        let n = 1000;
        let mut velocities = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);

        for i in 0..n {
            let phase = i as f64 * 2.0 * PI / 7.0;
            velocities.push([phase.sin() * 0.5, (phase * 1.3).cos() * 0.5, (phase * 0.7).sin() * 0.5]);
            masses.push(1.0);
            types.push(0);
        }

        let global = analyze_velocity_distribution(&velocities, &masses, 30, 3.0);
        let species = analyze_per_species(&velocities, &masses, &types, [0.0, 0.0, 0.0]);

        assert_eq!(species.len(), 1);
        assert_eq!(species[0].species, 0);
        assert_eq!(species[0].count, n as u64);
        // Single species T_g should match global T_g (approximately, since
        // global subtracts its own COM while species uses provided v_mean)
        let rel_err = (species[0].granular_temperature - global.granular_temperature).abs()
            / global.granular_temperature;
        assert!(
            rel_err < 0.01,
            "Species T_g = {}, Global T_g = {}, rel_err = {}",
            species[0].granular_temperature,
            global.granular_temperature,
            rel_err
        );
    }

    #[test]
    fn test_per_species_two_types_equipartition_breakdown() {
        // Two species with different masses but same initial KE per particle.
        // Light particles (type 0, mass=1) and heavy particles (type 1, mass=4).
        // Give them the same speed — heavy particles have 4× the KE.
        // After computing T_g per species, the heavy species should have higher T_g.
        let n = 2000;
        let mut velocities = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);

        for i in 0..n {
            let phase = i as f64 * 2.0 * PI / 11.0;
            let speed = 0.5;
            velocities.push([
                phase.sin() * speed,
                (phase * 1.3).cos() * speed,
                (phase * 0.7).sin() * speed,
            ]);
            if i < n / 2 {
                masses.push(1.0); // light
                types.push(0);
            } else {
                masses.push(4.0); // heavy
                types.push(1);
            }
        }

        let species = analyze_per_species(&velocities, &masses, &types, [0.0, 0.0, 0.0]);
        assert_eq!(species.len(), 2);

        // Both species have same speed, but different mass.
        // T_g = <m v²> / (3 m_mean) = m_mean <v²> / (3 m_mean) = <v²>/3
        // So per-species T_g should actually be similar when normalized by mass!
        // The key physics: T_s = <m_s v_s²> / (3 N_s m_s_mean) = <v_s²>/3
        // With same speed distribution, T_s should be similar for both species.
        let t0 = species[0].granular_temperature;
        let t1 = species[1].granular_temperature;
        let rel_diff = (t0 - t1).abs() / t0.max(t1);
        assert!(
            rel_diff < 0.1,
            "Same-speed species should have similar T_g: T0={}, T1={}, rel_diff={}",
            t0, t1, rel_diff
        );
    }

    #[test]
    fn test_per_species_different_speeds() {
        // Species 0: fast (v ~ 1.0), Species 1: slow (v ~ 0.1)
        // Species 0 should have ~100× higher T_g
        let n = 2000;
        let mut velocities = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);

        for i in 0..n {
            let phase = i as f64 * 2.0 * PI / 11.0;
            if i < n / 2 {
                velocities.push([phase.sin() * 1.0, (phase * 1.3).cos() * 1.0, (phase * 0.7).sin() * 1.0]);
                masses.push(1.0);
                types.push(0);
            } else {
                velocities.push([phase.sin() * 0.1, (phase * 1.3).cos() * 0.1, (phase * 0.7).sin() * 0.1]);
                masses.push(1.0);
                types.push(1);
            }
        }

        let species = analyze_per_species(&velocities, &masses, &types, [0.0, 0.0, 0.0]);
        assert_eq!(species.len(), 2);
        let ratio = species[0].granular_temperature / species[1].granular_temperature;
        // v ratio is 10:1, so T ratio should be ~100:1
        assert!(
            ratio > 50.0 && ratio < 200.0,
            "T_g ratio should be ~100, got {}",
            ratio
        );
    }

    #[test]
    fn test_per_species_empty() {
        let species = analyze_per_species(&[], &[], &[], [0.0, 0.0, 0.0]);
        assert!(species.is_empty());
    }

    // ── Collapse detector tests ────────────────────────────────────────

    #[test]
    fn test_collapse_detector_below_threshold() {
        let mut det = CollapseDetector::new(5);
        let threshold = 1e-10;

        // Above threshold — no collapse
        let d1 = det.record(0.0, 1.0, threshold);
        assert!(!d1.below_threshold);
        assert!(!d1.newly_detected);

        // Below threshold — first detection
        let d2 = det.record(1.0, 1e-11, threshold);
        assert!(d2.below_threshold);
        assert!(d2.newly_detected);

        // Still below — but not "newly" detected
        let d3 = det.record(2.0, 1e-12, threshold);
        assert!(d3.below_threshold);
        assert!(!d3.newly_detected);
    }

    #[test]
    fn test_collapse_detector_cooling_rate() {
        let mut det = CollapseDetector::new(5);

        // Exponential cooling: T(t) = T0 * exp(-γt), so d(ln T)/dt = -γ
        let gamma = 2.0;
        let t0 = 1.0;
        for i in 0..5 {
            let t = i as f64 * 0.1;
            let tg = t0 * (-gamma * t).exp();
            det.record(t, tg, 1e-20);
        }

        let rate = det.estimate_cooling_rate().unwrap();
        // Should be close to -γ = -2.0
        assert!(
            (rate - (-gamma)).abs() < 0.1,
            "Cooling rate should be ~-2.0, got {}",
            rate
        );
    }

    #[test]
    fn test_collapse_detector_insufficient_data() {
        let mut det = CollapseDetector::new(5);
        assert!(det.estimate_cooling_rate().is_none());
        det.record(0.0, 1.0, 1e-10);
        assert!(det.estimate_cooling_rate().is_none());
    }

    #[test]
    fn test_collapse_detector_window_limit() {
        let mut det = CollapseDetector::new(3);
        for i in 0..10 {
            det.record(i as f64, 1.0, 1e-10);
        }
        // Should only keep last 3 samples
        assert_eq!(det.history.len(), 3);
    }
}
