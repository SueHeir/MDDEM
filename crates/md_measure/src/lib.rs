use std::f64::consts::PI;
use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use md_lj::{LJTailCorrections, VirialAccumulator};
use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState};
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
pub struct MeasureConfig {
    #[serde(default = "default_rdf_bins")]
    pub rdf_bins: usize,
    #[serde(default = "default_rdf_cutoff")]
    pub rdf_cutoff: f64,
    #[serde(default = "default_rdf_interval")]
    pub rdf_interval: usize,
    #[serde(default = "default_msd_interval")]
    pub msd_interval: usize,
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

#[derive(Default)]
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
    pub ref_step: usize,
    pub msd_values: Vec<(usize, f64)>,
    pub initialized: bool,
}


#[derive(Default)]
pub struct PressureHistory {
    pub values: Vec<(usize, f64)>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

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
    let cutoff2 = rdf.cutoff * rdf.cutoff;

    let mut local_hist = vec![0.0f64; n_bins];

    // Use neighbor list for efficiency
    for &(i, j) in &neighbor.neighbor_list {
        let dx = atoms.pos_x[j] - atoms.pos_x[i];
        let dy = atoms.pos_y[j] - atoms.pos_y[i];
        let dz = atoms.pos_z[j] - atoms.pos_z[i];
        let r2 = dx * dx + dy * dy + dz * dz;

        if r2 >= cutoff2 || r2 < 1e-20 {
            continue;
        }

        let r = r2.sqrt();
        let bin = (r / dr) as usize;
        if bin < n_bins {
            // Count pair once for local-local, half for ghost pairs
            let weight = if atoms.is_ghost[i] || atoms.is_ghost[j] {
                0.5
            } else {
                1.0
            };
            local_hist[bin] += weight;
        }
    }

    // Reduce histogram across ranks
    for val in &mut local_hist {
        *val = comm.all_reduce_sum_f64(*val);
    }

    let n_total = comm.all_reduce_sum_f64(nlocal as f64);
    let v = domain.volume;

    // Normalize: g(r) = hist[k] * V / (N*(N-1)/2 * 4*pi*r^2*dr)
    // The neighbor list gives each pair once, so denominator uses N*(N-1)/2
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
    if comm.rank() != 0 {
        return;
    }

    let step = run_state.total_cycle;
    let nlocal = atoms.nlocal as usize;

    if !msd.initialized && nlocal > 0 {
        // Initialize reference and unwrapped positions
        msd.ref_x = atoms.pos_x[..nlocal].to_vec();
        msd.ref_y = atoms.pos_y[..nlocal].to_vec();
        msd.ref_z = atoms.pos_z[..nlocal].to_vec();
        msd.unwrapped_x = msd.ref_x.clone();
        msd.unwrapped_y = msd.ref_y.clone();
        msd.unwrapped_z = msd.ref_z.clone();
        msd.prev_x = msd.ref_x.clone();
        msd.prev_y = msd.ref_y.clone();
        msd.prev_z = msd.ref_z.clone();
        msd.ref_step = step;
        msd.initialized = true;
        msd.msd_values.push((0, 0.0));
        return;
    }

    if !msd.initialized || nlocal != msd.prev_x.len() {
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
        let dx = atoms.pos_x[i] - msd.prev_x[i];
        let dy = atoms.pos_y[i] - msd.prev_y[i];
        let dz = atoms.pos_z[i] - msd.prev_z[i];

        // Detect boundary crossing
        let mut ux = dx;
        let mut uy = dy;
        let mut uz = dz;
        if ux > half_lx {
            ux -= lx;
        } else if ux < -half_lx {
            ux += lx;
        }
        if uy > half_ly {
            uy -= ly;
        } else if uy < -half_ly {
            uy += ly;
        }
        if uz > half_lz {
            uz -= lz;
        } else if uz < -half_lz {
            uz += lz;
        }

        msd.unwrapped_x[i] += ux;
        msd.unwrapped_y[i] += uy;
        msd.unwrapped_z[i] += uz;
    }

    msd.prev_x = atoms.pos_x[..nlocal].to_vec();
    msd.prev_y = atoms.pos_y[..nlocal].to_vec();
    msd.prev_z = atoms.pos_z[..nlocal].to_vec();

    if step == 0 || !step.is_multiple_of(config.msd_interval) {
        return;
    }

    // Compute MSD = <|r_unwrap(t) - r_ref|^2>
    let mut msd_sum = 0.0;
    for i in 0..nlocal {
        let dx = msd.unwrapped_x[i] - msd.ref_x[i];
        let dy = msd.unwrapped_y[i] - msd.ref_y[i];
        let dz = msd.unwrapped_z[i] - msd.ref_z[i];
        msd_sum += dx * dx + dy * dy + dz * dz;
    }
    let msd_avg = msd_sum / nlocal as f64;
    let ref_step = msd.ref_step;
    msd.msd_values.push((step - ref_step, msd_avg));
}

#[allow(clippy::too_many_arguments)]
pub fn compute_pressure(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    virial: Res<VirialAccumulator>,
    tails: Res<LJTailCorrections>,
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
                * (atoms.vel_x[i].powi(2) + atoms.vel_y[i].powi(2) + atoms.vel_z[i].powi(2))
        })
        .sum::<f64>()
        * 0.5;
    let global_ke = comm.all_reduce_sum_f64(local_ke);
    let ndof = 3.0 * n - 3.0;
    let temp = if ndof > 0.0 { 2.0 * global_ke / ndof } else { 0.0 };

    // Virial pressure: P = rho*T + virial_sum/(3*V) + P_tail
    let global_virial = comm.all_reduce_sum_f64(virial.virial_sum);
    let pressure = rho * temp + global_virial / (3.0 * v) + tails.pressure_tail;

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
