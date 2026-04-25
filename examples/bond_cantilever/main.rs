//! BPM cantilever test — demonstrates `[[pin]]` as a hard constraint.
//!
//! A 10-sphere bonded chain is anchored at one end by a `[[pin]]` group.
//! Gravity bends the free end down; critical bond damping settles the chain
//! to its static deflection. The recorder tracks tip z-position, max bond
//! strain, and `bond_missing` every 100 steps — a PASS run shows
//! `bond_missing = 0` throughout and the tip converging to a stable
//! deflection (no blowup in the bond adjacent to the pin).
//!
//! Run:
//! ```bash
//! cargo run --release --example bond_cantilever --no-default-features -- \
//!     examples/bond_cantilever/config.toml
//! ```

use mddem::prelude::*;
use mddem::dem_bond::BondMetrics;
use mddem::mddem_core::BondStore;
use std::fs::{self, File};
use std::io::{BufWriter, Write as IoWrite};

const EXPECTED_BONDS: usize = 9; // 10 atoms → 9 bonds
const TIP_TAG: u32 = 9;          // free-end tag (CSV assigns 0..9)

struct Recorder {
    writer: Option<BufWriter<File>>,
    initialized: bool,
    record_every: usize,
    max_strain_seen: f64,
    min_tip_z: f64,
    max_missing: usize,
}

impl Recorder {
    fn new() -> Self {
        Self {
            writer: None,
            initialized: false,
            record_every: 1000,
            max_strain_seen: 0.0,
            min_tip_z: 0.0,
            max_missing: 0,
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(FixesPlugin)
        .add_plugins(GravityPlugin)
        .add_plugins(DemBondPlugin);

    app.add_resource(Recorder::new());
    app.add_update_system(record_cantilever, ParticleSimScheduleSet::PostFinalIntegration);

    app.start();
}

fn record_cantilever(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_metrics: Res<BondMetrics>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    mut rec: ResMut<Recorder>,
) {
    let step = run_state.total_cycle;
    if step % rec.record_every != 0 {
        return;
    }

    // Collective: bond counters must be reduced on every rank at the same step.
    let global_bond_count =
        comm.all_reduce_sum_f64(bond_metrics.bond_count as f64) as usize;
    let global_missing =
        comm.all_reduce_sum_f64(bond_metrics.missing_partner_skips as f64) as usize;

    // Find the tip locally; if not owned on this rank, contribute z = +inf so
    // the global min reduction yields the owning rank's value.
    let mut local_tip_z = f64::MAX;
    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        if atoms.tag[i] == TIP_TAG {
            local_tip_z = atoms.pos[i][2];
            break;
        }
    }
    let global_tip_z = comm.all_reduce_min_f64(local_tip_z);

    // Largest |δ / r₀| across all bonds this step (diagnostic for adjacent-
    // to-pin blowup). Iterate local bonds, pick partner from local+ghost.
    let max_local_strain = {
        let bonds = match registry.get::<BondStore>() {
            Some(b) => b,
            None => return,
        };
        let mut max_s = 0.0f64;
        // Build tag -> index lookup over local+ghost
        let mut tag_to_index = std::collections::HashMap::with_capacity(atoms.len());
        for idx in 0..atoms.len() {
            tag_to_index.insert(atoms.tag[idx], idx);
        }
        for i in 0..nlocal {
            if i >= bonds.bonds.len() {
                break;
            }
            for b in &bonds.bonds[i] {
                if atoms.tag[i] >= b.partner_tag {
                    continue; // process each bond once
                }
                let j = match tag_to_index.get(&b.partner_tag) {
                    Some(&jx) => jx,
                    None => continue,
                };
                let dx = atoms.pos[j][0] - atoms.pos[i][0];
                let dy = atoms.pos[j][1] - atoms.pos[i][1];
                let dz = atoms.pos[j][2] - atoms.pos[i][2];
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                let strain = ((dist - b.r0) / b.r0).abs();
                if strain > max_s {
                    max_s = strain;
                }
            }
        }
        max_s
    };
    // max via negated min
    let max_strain = -comm.all_reduce_min_f64(-max_local_strain);

    if max_strain > rec.max_strain_seen {
        rec.max_strain_seen = max_strain;
    }
    if global_tip_z < rec.min_tip_z {
        rec.min_tip_z = global_tip_z;
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
            .unwrap_or_else(|| "examples/bond_cantilever".to_string());
        fs::create_dir_all(format!("{}/data", out_dir)).ok();
        let path = format!("{}/data/cantilever.csv", out_dir);
        let mut w = BufWriter::new(
            File::create(&path).unwrap_or_else(|e| panic!("Cannot create {}: {}", path, e)),
        );
        writeln!(w, "step,t,tip_z,max_strain,bond_count,bond_missing").unwrap();
        rec.writer = Some(w);
        rec.initialized = true;

        println!("=== BPM Cantilever Test ===");
        println!("  chain length   : 10 atoms, 9 bonds");
        println!("  anchor         : tag 0 (pinned via [[pin]])");
        println!("  tip tag        : {}", TIP_TAG);
        println!("  MPI ranks      : {}", comm.size());
    }

    let dt = atoms.dt;
    let t = step as f64 * dt;

    if let Some(ref mut w) = rec.writer {
        writeln!(
            w,
            "{},{:.8e},{:.8e},{:.8e},{},{}",
            step, t, global_tip_z, max_strain, global_bond_count, global_missing
        )
        .ok();
    }

    // Print a running summary every 10 sample intervals.
    if step % (rec.record_every * 10) == 0 {
        let status = if rec.max_missing == 0 && global_bond_count >= EXPECTED_BONDS {
            "OK"
        } else {
            "BROKEN"
        };
        println!(
            "  step {:>7}  tip_z={:+.3e}  max_strain={:.3e}  bonds={}  missing={}  {}",
            step,
            global_tip_z,
            max_strain,
            global_bond_count,
            global_missing,
            status
        );
    }
}
