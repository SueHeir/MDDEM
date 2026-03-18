//! Hertz contact rebound benchmark — validates coefficient of restitution,
//! contact duration, and peak overlap against Hertz contact theory.
//!
//! Drops a single sphere onto a rigid flat wall and records the impact
//! velocity, rebound velocity, contact duration, and peak overlap.
//!
//! ```bash
//! cargo run --release --example bench_hertz_rebound --no-default-features -- examples/bench_hertz_rebound/config.toml
//! ```

use mddem::prelude::*;
use mddem::dem_atom::DemAtom;
use std::fs;
use std::io::Write as IoWrite;

/// Tracks contact state for the rebound measurement.
struct ReboundTracker {
    /// True if the particle has been in contact with the wall.
    was_in_contact: bool,
    /// True once the particle has separated after contact.
    finished: bool,
    /// z-velocity just before first contact (impact velocity, negative = downward).
    v_impact: f64,
    /// z-velocity just after separation (rebound velocity, positive = upward).
    v_rebound: f64,
    /// Timestep when contact first occurs.
    step_contact_start: usize,
    /// Timestep when contact ends (separation).
    step_contact_end: usize,
    /// Maximum overlap during contact.
    max_overlap: f64,
    /// z-velocity at the previous step (to capture pre-contact velocity).
    prev_vz: f64,
    /// Output directory for results.
    output_dir: String,
}

impl ReboundTracker {
    fn new() -> Self {
        Self {
            was_in_contact: false,
            finished: false,
            v_impact: 0.0,
            v_rebound: 0.0,
            step_contact_start: 0,
            step_contact_end: 0,
            max_overlap: 0.0,
            prev_vz: 0.0,
            output_dir: String::new(),
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(WallPlugin);

    app.add_resource(ReboundTracker::new());

    app.add_update_system(track_rebound, ScheduleSet::PostFinalIntegration);

    app.start();
}

/// System that monitors the single particle's contact with the floor wall
/// and records impact/rebound velocity, contact duration, and peak overlap.
fn track_rebound(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    input: Res<Input>,
    mut tracker: ResMut<ReboundTracker>,
) {
    if tracker.finished || atoms.nlocal == 0 {
        return;
    }

    let dem = registry.expect::<DemAtom>("track_rebound");
    let step = run_state.total_cycle;

    // Single particle at index 0
    let z = atoms.pos[0][2];
    let vz = atoms.vel[0][2];
    let r = dem.radius[0];

    // Check overlap with the floor wall (z = 0, normal +z)
    // The floor is the first wall defined in config
    // Overlap = radius - z (positive when overlapping)
    let overlap = r - z;
    let in_contact = overlap > 0.0;

    if !tracker.was_in_contact && !in_contact {
        // Pre-contact: record velocity for next step
        tracker.prev_vz = vz;
    } else if !tracker.was_in_contact && in_contact {
        // First contact! Record impact velocity from previous step
        tracker.was_in_contact = true;
        tracker.v_impact = tracker.prev_vz;
        tracker.step_contact_start = step;
        tracker.max_overlap = overlap;
    } else if tracker.was_in_contact && in_contact {
        // During contact: track max overlap
        if overlap > tracker.max_overlap {
            tracker.max_overlap = overlap;
        }
    } else if tracker.was_in_contact && !in_contact {
        // Separation: contact has ended
        tracker.finished = true;
        tracker.v_rebound = vz;
        tracker.step_contact_end = step;

        let dt = atoms.dt;
        let contact_steps = tracker.step_contact_end - tracker.step_contact_start;
        let contact_time = contact_steps as f64 * dt;
        let cor_measured = (tracker.v_rebound / tracker.v_impact).abs();

        // Determine output directory
        let out_dir = if let Some(ref dir) = input.output_dir {
            dir.clone()
        } else {
            "examples/bench_hertz_rebound/data".to_string()
        };
        tracker.output_dir = out_dir.clone();

        // Ensure data directory exists
        let data_dir = format!("{}/data", out_dir);
        fs::create_dir_all(&data_dir).ok();

        // Write results to file
        let results_file = format!("{}/data/rebound_results.csv", out_dir);
        let mut f = fs::File::create(&results_file)
            .unwrap_or_else(|e| panic!("Cannot create {}: {}", results_file, e));
        writeln!(f, "v_impact,v_rebound,cor_measured,contact_time,max_overlap,dt,radius,density")
            .unwrap();
        writeln!(
            f,
            "{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}",
            tracker.v_impact.abs(),
            tracker.v_rebound.abs(),
            cor_measured,
            contact_time,
            tracker.max_overlap,
            dt,
            r,
            dem.density[0],
        )
        .unwrap();

        println!("=== Hertz Rebound Results ===");
        println!("  Impact velocity:  {:.6e} m/s", tracker.v_impact.abs());
        println!("  Rebound velocity: {:.6e} m/s", tracker.v_rebound.abs());
        println!("  COR (measured):   {:.6}", cor_measured);
        println!("  Contact duration: {:.6e} s ({} steps)", contact_time, contact_steps);
        println!("  Peak overlap:     {:.6e} m", tracker.max_overlap);
        println!("  Timestep dt:      {:.6e} s", dt);
        println!("  Results saved to: {}", results_file);
    }

    if !tracker.finished && !in_contact {
        tracker.prev_vz = vz;
    }
}
