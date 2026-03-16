//! Velocity distribution analysis plugin for particle simulations.
//!
//! Measures particle speed distributions and compares against the Maxwell-Boltzmann
//! distribution for the measured granular temperature. Outputs:
//! - Binned speed histogram (probability density)
//! - Per-component velocity histograms
//! - Maxwell-Boltzmann reference curve
//! - Quantitative deviation metrics (L2 norm, kurtosis excess)
//!
//! # TOML Configuration
//! ```toml
//! [velocity_distribution]
//! interval = 1000        # output every N steps
//! num_bins = 50           # number of histogram bins
//! max_speed_factor = 3.0  # max speed = factor * v_rms
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
}

impl Default for VelocityDistributionConfig {
    fn default() -> Self {
        VelocityDistributionConfig {
            interval: default_interval(),
            num_bins: default_num_bins(),
            max_speed_factor: default_max_speed_factor(),
        }
    }
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
max_speed_factor = 3.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<VelocityDistributionConfig>(app, "velocity_distribution");
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
    /// Per-component velocity histograms (vx, vy, vz) — probability density.
    pub component_bin_centers: Vec<f64>,
    pub component_pdf_x: Vec<f64>,
    pub component_pdf_y: Vec<f64>,
    pub component_pdf_z: Vec<f64>,
    /// Gaussian reference for component distributions.
    pub component_gaussian: Vec<f64>,
}

// ── Maxwell-Boltzmann PDF ──────────────────────────────────────────────────

/// Maxwell-Boltzmann speed distribution for 3D:
/// f(v) = 4π (m / (2πkT))^(3/2) v² exp(-mv²/(2kT))
///
/// Using granular temperature T_g = <v'^2>/3 (mass cancels if all equal mass):
/// f(v) = 4π (1/(2πT_g))^(3/2) v² exp(-v²/(2T_g))
fn maxwell_boltzmann_pdf(v: f64, t_granular: f64) -> f64 {
    if t_granular <= 0.0 || v < 0.0 {
        return 0.0;
    }
    let a = 1.0 / (2.0 * t_granular);
    4.0 * PI * (a / PI).powf(1.5) * v * v * (-a * v * v).exp()
}

/// 1D Gaussian PDF for a single velocity component with zero mean and variance T_g.
fn gaussian_component_pdf(v: f64, t_granular: f64) -> f64 {
    if t_granular <= 0.0 {
        return 0.0;
    }
    let sigma2 = t_granular;
    (1.0 / (2.0 * PI * sigma2).sqrt()) * (-v * v / (2.0 * sigma2)).exp()
}

// ── Core analysis function (unit-testable) ─────────────────────────────────

/// Compute velocity distribution statistics from raw velocity data.
/// This is the pure-computation core, separated from I/O for testability.
/// Works for any particle simulation (DEM, MD, etc.).
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

// ── ECS system ─────────────────────────────────────────────────────────────

fn compute_velocity_distribution(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    config: Res<VelocityDistributionConfig>,
    input: Res<Input>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
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

    // Append summary to time-series file
    let summary_path = format!("{}/velocity_distribution_summary.csv", base_dir);
    let is_new = !std::path::Path::new(&summary_path).exists();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_path)
    {
        if is_new {
            writeln!(file, "step,time,granular_temp,v_rms,l2_deviation,kurtosis_excess,n_particles").ok();
        }
        let physical_time = step as f64 * atoms.dt;
        writeln!(
            file,
            "{},{:.6e},{:.10e},{:.10e},{:.10e},{:.10e},{}",
            step, physical_time, result.granular_temperature, result.v_rms,
            result.l2_deviation, result.kurtosis_excess, result.n_particles
        )
        .ok();
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
}
