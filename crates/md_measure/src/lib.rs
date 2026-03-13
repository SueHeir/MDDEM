//! Measurement tools: radial distribution function (RDF), mean square displacement (MSD),
//! and virial pressure.

use std::f64::consts::PI;
use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use md_lj::LJTailCorrections;
use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState, VirialStress};
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

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[measure]` — RDF, MSD, and pressure measurement settings.
pub struct MeasureConfig {
    /// Number of histogram bins for RDF.
    #[serde(default = "default_rdf_bins")]
    pub rdf_bins: usize,
    /// RDF cutoff distance.
    #[serde(default = "default_rdf_cutoff")]
    pub rdf_cutoff: f64,
    /// Sample RDF every N steps.
    #[serde(default = "default_rdf_interval")]
    pub rdf_interval: usize,
    /// Sample MSD every N steps.
    #[serde(default = "default_msd_interval")]
    pub msd_interval: usize,
    /// Write measurement output files every N steps.
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

/// Accumulates radial distribution function histogram samples.
pub struct RdfAccumulator {
    pub bins: Vec<f64>,
    pub n_samples: usize,
    pub dr: f64,
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

/// Tracks reference positions for mean square displacement calculation.
pub struct MsdTracker {
    pub ref_x: Vec<f64>,
    pub ref_y: Vec<f64>,
    pub ref_z: Vec<f64>,
    pub unwrapped_x: Vec<f64>,
    pub unwrapped_y: Vec<f64>,
    pub unwrapped_z: Vec<f64>,
    pub prev_x: Vec<f64>,
    pub prev_y: Vec<f64>,
    pub prev_z: Vec<f64>,
    pub has_entry: Vec<bool>,
    pub ref_step: usize,
    pub msd_values: Vec<(usize, f64)>,
    pub initialized: bool,
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


#[derive(Default)]
pub struct PressureHistory {
    pub values: Vec<(usize, f64)>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers RDF, MSD, and virial pressure measurement systems.
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
            .add_update_system(accumulate_rdf, ScheduleSet::PostFinalIntegration)
            .add_update_system(track_msd, ScheduleSet::PostFinalIntegration)
            .add_update_system(compute_pressure, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_measurements, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

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

    // Brute-force pair counting with minimum-image convention
    let lx = domain.size.x;
    let ly = domain.size.y;
    let lz = domain.size.z;
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;
    let total = atoms.len();

    for i in 0..nlocal {
        // Local-local pairs (i < j to avoid double counting)
        for j in (i + 1)..nlocal {
            let mut dx = atoms.pos[j][0] - atoms.pos[i][0];
            let mut dy = atoms.pos[j][1] - atoms.pos[i][1];
            let mut dz = atoms.pos[j][2] - atoms.pos[i][2];

            if domain.is_periodic.x {
                if dx > half_lx { dx -= lx; } else if dx < -half_lx { dx += lx; }
            }
            if domain.is_periodic.y {
                if dy > half_ly { dy -= ly; } else if dy < -half_ly { dy += ly; }
            }
            if domain.is_periodic.z {
                if dz > half_lz { dz -= lz; } else if dz < -half_lz { dz += lz; }
            }

            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= cutoff2 || r2 < 1e-20 { continue; }
            let bin = (r2.sqrt() / dr) as usize;
            if bin < n_bins { local_hist[bin] += 1.0; }
        }
        // Local-ghost pairs (weight 0.5 — each cross-boundary pair seen by both ranks)
        // Skip on single-process: minimum-image local-local loop already finds all pairs
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

    // Reduce histogram across ranks
    for val in &mut local_hist {
        *val = comm.all_reduce_sum_f64(*val);
    }

    let n_total = comm.all_reduce_sum_f64(nlocal as f64);
    let v = domain.volume;

    // Normalize: g(r) = hist[k] * V / (N*(N-1)/2 * 4*pi*r^2*dr)
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
        // Find max tag to size the dense arrays
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

    // Update unwrapped positions by detecting PBC jumps
    let lx = domain.size.x;
    let ly = domain.size.y;
    let lz = domain.size.z;
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
            let mut dx = atoms.pos[i][0] - msd.prev_x[idx];
            let mut dy = atoms.pos[i][1] - msd.prev_y[idx];
            let mut dz = atoms.pos[i][2] - msd.prev_z[idx];

            // Detect PBC boundary crossing
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

    // Compute MSD = <|r_unwrap(t) - r_ref|^2> for atoms we can track
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

    // Compute instantaneous temperature
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

    // Virial pressure: P = rho*T - trace/(3*V) + P_tail
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

    // Write RDF
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

    // Write MSD
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

    // Write pressure
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
