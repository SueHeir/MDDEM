//! BPM MPI bond-migration test.
//!
//! Drifts a 3-atom bonded chain through a periodic domain that is split
//! across MPI ranks. Verifies, every `record_every` steps, that the expected
//! number of bonds are being evaluated globally (`bond_count`) and that none
//! are silently skipped for a missing partner (`bond_missing`).
//!
//! **Pass criterion** (watch stdout on rank 0): over the full run,
//! `bond_count` must stay ≥ the total number of bonds and `bond_missing`
//! must stay at 0. The thermo table and `data/bond_drift.csv` record these
//! per-step.
//!
//! Single-process run:
//! ```bash
//! cargo run --release --example bond_mpi_drift --no-default-features -- \
//!     examples/bond_mpi_drift/config.toml
//! ```
//!
//! MPI run (2 ranks along x):
//! ```bash
//! cargo build --release --example bond_mpi_drift
//! mpiexec -n 2 target/release/examples/bond_mpi_drift \
//!     examples/bond_mpi_drift/config.toml
//! ```

use mddem::prelude::*;
use mddem::dem_bond::BondMetrics;
use std::fs::{self, File};
use std::io::{BufWriter, Write as IoWrite};

/// Expected total number of bonds in the chain (3 atoms, 2 bonds).
const EXPECTED_BONDS: usize = 2;

/// Holds the CSV writer (rank 0 only) plus running min/max trackers.
struct DriftRecorder {
    writer: Option<BufWriter<File>>,
    min_bond_count: usize,
    max_missing: usize,
    last_printed_step: usize,
    initialized: bool,
    record_every: usize,
}

impl DriftRecorder {
    fn new() -> Self {
        Self {
            writer: None,
            min_bond_count: usize::MAX,
            max_missing: 0,
            last_printed_step: 0,
            initialized: false,
            record_every: 1000,
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(DemBondPlugin);

    app.add_resource(DriftRecorder::new());
    app.add_update_system(record_drift, ParticleSimScheduleSet::PostFinalIntegration);

    app.start();
}

/// Sums `bond_count` and `bond_missing` from the local [`BondMetrics`] across
/// all ranks every `record_every` steps. Tracks the global min/max and writes
/// a CSV row on rank 0. Prints a running summary every 10 sample intervals
/// so stdout stays readable over long runs.
fn record_drift(
    atoms: Res<Atom>,
    bond_metrics: Res<BondMetrics>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    mut rec: ResMut<DriftRecorder>,
) {
    let step = run_state.total_cycle;

    // `all_reduce_*` must run collectively on every rank for the same step,
    // so the sampling cadence is step-based and identical across ranks.
    if step % rec.record_every != 0 {
        return;
    }

    let global_nlocal = comm.all_reduce_sum_f64(atoms.nlocal as f64) as usize;
    let global_bond_count = comm.all_reduce_sum_f64(bond_metrics.bond_count as f64) as usize;
    let global_missing = comm.all_reduce_sum_f64(bond_metrics.missing_partner_skips as f64) as usize;

    if global_bond_count < rec.min_bond_count {
        rec.min_bond_count = global_bond_count;
    }
    if global_missing > rec.max_missing {
        rec.max_missing = global_missing;
    }

    if comm.rank() != 0 {
        return;
    }

    if !rec.initialized {
        let out_dir = input
            .output_dir
            .clone()
            .unwrap_or_else(|| "examples/bond_mpi_drift".to_string());
        fs::create_dir_all(format!("{}/data", out_dir)).ok();
        let path = format!("{}/data/bond_drift.csv", out_dir);
        let mut w = BufWriter::new(
            File::create(&path).unwrap_or_else(|e| panic!("Cannot create {}: {}", path, e)),
        );
        writeln!(w, "step,t,nlocal_global,bond_count,bond_missing,ranks").unwrap();
        rec.writer = Some(w);
        rec.initialized = true;

        println!("=== BPM MPI Drift Test ===");
        println!("  expected bonds : {}", EXPECTED_BONDS);
        println!("  MPI ranks      : {}", comm.size());
        println!("  output CSV     : {}/data/bond_drift.csv", out_dir);
    }

    let dt = atoms.dt;
    let t = step as f64 * dt;

    if let Some(ref mut w) = rec.writer {
        writeln!(
            w,
            "{},{:.8e},{},{},{},{}",
            step, t, global_nlocal, global_bond_count, global_missing, comm.size()
        )
        .ok();
    }

    // Print a running verdict every ~10 sample intervals.
    if step.saturating_sub(rec.last_printed_step) >= rec.record_every * 10 || step == 0 {
        let status = if rec.max_missing == 0 && rec.min_bond_count >= EXPECTED_BONDS {
            "OK"
        } else {
            "BROKEN"
        };
        println!(
            "  step {:>8}  atoms={:<4}  bond_count={:<3}  bond_missing={:<3}  [min={}, max_miss={}]  {}",
            step,
            global_nlocal,
            global_bond_count,
            global_missing,
            rec.min_bond_count,
            rec.max_missing,
            status
        );
        rec.last_printed_step = step;
    }
}
