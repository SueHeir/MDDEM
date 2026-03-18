//! General-purpose measurement plane plugin for MDDEM.
//!
//! A **measurement plane** is an infinite plane defined by a point and a normal
//! vector. Each timestep, the plugin computes the signed distance of every local
//! particle from the plane. When a particle's signed distance changes from
//! non-positive to positive between consecutive steps, the particle is counted
//! as having **crossed** the plane in the positive-normal direction.
//!
//! This is useful for measuring throughput in hoppers, chutes, conveyors, and
//! other granular flows where you need to know how many particles (and how much
//! mass) pass through a specific cross-section per unit time.
//!
//! # Crossing detection algorithm
//!
//! For each particle tracked by tag:
//! 1. Compute the signed distance `d = (pos - point) · normal`.
//! 2. If the previous signed distance `d_prev ≤ 0` and the current `d > 0`,
//!    the particle crossed the plane in the positive-normal direction.
//! 3. Only positive-direction crossings are counted; reverse crossings are
//!    ignored.
//!
//! # Configuration
//!
//! ```toml
//! [[measure_plane]]
//! name = "outlet"           # Unique name; used in thermo output keys
//! point = [0.1, 0.0, 0.0]  # Any point on the plane [length units]
//! normal = [1.0, 0.0, 0.0] # Outward normal (automatically normalized)
//! report_interval = 1000   # Averaging window in timesteps (default: 1000)
//! ```
//!
//! Multiple `[[measure_plane]]` blocks can be defined. Each plane tracks
//! crossings independently. Results are pushed to thermo as:
//! - `crossings_<name>` — total cumulative crossing count (positive direction)
//! - `flow_rate_<name>` — mass flow rate (mass/time) averaged over `report_interval`
//! - `cross_rate_<name>` — particle crossing rate (1/time) averaged over `report_interval`

use std::collections::HashMap;

use sim_app::prelude::*;
use mddem_core::{Atom, CommResource, Config, RunState, ScheduleSet};
use mddem_print::Thermo;
use sim_scheduler::prelude::*;
use serde::Deserialize;

// ── Configuration ───────────────────────────────────────────────────────────

/// Default report interval in timesteps when not specified in config.
fn default_report_interval() -> usize {
    1000
}

/// TOML configuration for a single measurement plane (`[[measure_plane]]`).
///
/// Defines an infinite plane in 3D space used for particle crossing detection.
/// The plane is specified by a point and a normal vector; the normal is
/// automatically normalized at construction time.
///
/// # TOML fields
///
/// | Field             | Type       | Default | Description                                      |
/// |-------------------|------------|---------|--------------------------------------------------|
/// | `name`            | `String`   | —       | Unique name used in thermo output keys           |
/// | `point`           | `[f64; 3]` | —       | Any point lying on the plane (length units)      |
/// | `normal`          | `[f64; 3]` | —       | Outward normal direction (auto-normalized)       |
/// | `report_interval` | `usize`    | `1000`  | Averaging window in timesteps                    |
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MeasurePlaneDef {
    /// Unique human-readable name for this measurement plane.
    /// Used as a suffix in thermo output keys (e.g., `crossings_<name>`).
    pub name: String,
    /// A point on the plane, in simulation length units `[x, y, z]`.
    pub point: [f64; 3],
    /// Outward normal direction of the plane `[nx, ny, nz]`.
    /// Does not need to be unit length — it is normalized automatically.
    pub normal: [f64; 3],
    /// Reporting interval in timesteps. Crossing and flow rates are averaged
    /// over this window. Default: `1000`.
    #[serde(default = "default_report_interval")]
    pub report_interval: usize,
}

// ── Runtime state ───────────────────────────────────────────────────────────

/// Per-plane runtime state for crossing detection.
///
/// Tracks the signed distance of each particle (by tag) from the plane at
/// the previous timestep, enabling sign-change detection for crossings.
/// Accumulates crossing count and mass within a reporting window, then
/// resets the window counters when results are reported to thermo.
struct MeasurePlaneState {
    /// Human-readable name (from config), used in thermo output keys.
    name: String,
    /// A point on the plane (copied from config).
    point: [f64; 3],
    /// Unit normal vector (normalized from config at construction time).
    normal: [f64; 3],
    /// Reporting interval in timesteps.
    report_interval: usize,

    /// Signed distance of each tracked particle at the previous timestep.
    /// Key: atom tag, Value: signed distance `d = (pos - point) · normal`.
    prev_signed_dist: HashMap<u32, f64>,

    /// Number of positive-direction crossings accumulated since the last report.
    crossings_window: u64,
    /// Total mass of particles that crossed in the positive direction since the last report.
    mass_window: f64,
    /// Total cumulative crossings since simulation start (never reset).
    total_crossings: u64,
    /// Timestep at which the current reporting window started.
    window_start_step: usize,
}

impl MeasurePlaneState {
    /// Create a new `MeasurePlaneState` from a config definition.
    ///
    /// Normalizes the normal vector. If the provided normal has near-zero
    /// magnitude (< 1e-30), falls back to the +x direction `[1, 0, 0]`.
    fn new(def: &MeasurePlaneDef) -> Self {
        // Normalize the normal vector to unit length for signed-distance calculations.
        let mag = (def.normal[0].powi(2) + def.normal[1].powi(2) + def.normal[2].powi(2)).sqrt();
        let normal = if mag > 1e-30 {
            [def.normal[0] / mag, def.normal[1] / mag, def.normal[2] / mag]
        } else {
            [1.0, 0.0, 0.0] // fallback to +x if degenerate
        };
        MeasurePlaneState {
            name: def.name.clone(),
            point: def.point,
            normal,
            report_interval: def.report_interval,
            prev_signed_dist: HashMap::new(),
            crossings_window: 0,
            mass_window: 0.0,
            total_crossings: 0,
            window_start_step: 0,
        }
    }

    /// Compute the signed distance from the plane for a given position.
    ///
    /// Returns `(pos - point) · normal`. Positive values mean the particle is
    /// on the side the normal points toward; negative values mean the opposite side.
    #[inline]
    fn signed_distance(&self, pos: &[f64; 3]) -> f64 {
        let dx = pos[0] - self.point[0];
        let dy = pos[1] - self.point[1];
        let dz = pos[2] - self.point[2];
        dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2]
    }
}

/// Resource holding runtime state for all configured measurement planes.
///
/// Inserted into the ECS by [`MeasurePlanePlugin::build`]. Contains an empty
/// `Vec` if no `[[measure_plane]]` blocks are present in config.
pub struct MeasurePlanes {
    planes: Vec<MeasurePlaneState>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that registers measurement plane systems for particle crossing
/// detection and throughput tracking.
///
/// Reads `[[measure_plane]]` blocks from the TOML config and sets up:
/// - A [`MeasurePlanes`] resource with per-plane runtime state
/// - A crossing-detection system running at [`ScheduleSet::PostFinalIntegration`]
/// - A reporting system that writes averaged rates to [`Thermo`](mddem_print::Thermo)
///
/// If no `[[measure_plane]]` blocks are configured, only an empty resource is
/// inserted and no systems are registered.
pub struct MeasurePlanePlugin;

impl Plugin for MeasurePlanePlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Measurement planes for particle crossing detection and throughput tracking.
# [[measure_plane]]
# name = "outlet"
# point = [0.1, 0.0, 0.0]
# normal = [1.0, 0.0, 0.0]
# report_interval = 1000"#,
        )
    }

    fn build(&self, app: &mut App) {
        let defs = {
            let config = app
                .get_resource_ref::<Config>()
                .expect("Config resource must exist");
            config.parse_array::<MeasurePlaneDef>("measure_plane")
        };

        if defs.is_empty() {
            app.add_resource(MeasurePlanes { planes: Vec::new() });
            return;
        }

        let planes: Vec<MeasurePlaneState> = defs.iter().map(MeasurePlaneState::new).collect();

        app.add_resource(MeasurePlanes { planes });
        app.add_update_system(
            measure_plane_detect_crossings,
            ScheduleSet::PostFinalIntegration,
        );
        app.add_update_system(
            measure_plane_report,
            ScheduleSet::PostFinalIntegration,
        );
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Detect particles crossing each measurement plane.
///
/// Runs every timestep at [`ScheduleSet::PostFinalIntegration`]. For each
/// local particle, computes the signed distance from each plane and compares
/// it to the previous step's distance (stored by atom tag). A crossing is
/// recorded when the signed distance transitions from `≤ 0` to `> 0`,
/// meaning the particle moved through the plane in the positive-normal direction.
fn measure_plane_detect_crossings(atoms: Res<Atom>, mut planes: ResMut<MeasurePlanes>) {
    let nlocal = atoms.nlocal as usize;

    for plane in planes.planes.iter_mut() {
        for i in 0..nlocal {
            let tag = atoms.tag[i];

            // Compute signed distance: positive means on the normal side of the plane.
            let dist = plane.signed_distance(&atoms.pos[i]);

            if let Some(&prev_dist) = plane.prev_signed_dist.get(&tag) {
                // Crossing detection: a sign change from non-positive to positive
                // indicates the particle crossed the plane in the normal direction.
                // Reverse crossings (positive → negative) are intentionally ignored.
                if prev_dist <= 0.0 && dist > 0.0 {
                    plane.crossings_window += 1;
                    plane.total_crossings += 1;
                    plane.mass_window += atoms.mass[i];
                }
            }
            // If no previous distance exists (first timestep for this particle),
            // we just record the current distance without counting a crossing.

            plane.prev_signed_dist.insert(tag, dist);
        }
    }
}

/// Report measurement plane statistics to thermo output at each plane's
/// configured `report_interval`.
///
/// Performs an MPI all-reduce to sum crossings and mass across all ranks,
/// computes time-averaged flow and crossing rates over the reporting window,
/// then pushes results to [`Thermo`](mddem_print::Thermo) and resets the
/// window counters.
fn measure_plane_report(
    run_state: Res<RunState>,
    atoms: Res<Atom>,
    comm: Res<CommResource>,
    mut planes: ResMut<MeasurePlanes>,
    mut thermo: ResMut<Thermo>,
) {
    let step = run_state.total_cycle;

    for plane in planes.planes.iter_mut() {
        if plane.report_interval == 0 {
            continue;
        }
        if !step.is_multiple_of(plane.report_interval) {
            continue;
        }

        // Sum crossing counts and mass across all MPI ranks so every rank
        // sees the global totals (needed for consistent thermo output).
        let local_crossings = plane.crossings_window as f64;
        let local_mass = plane.mass_window;
        let global_crossings = comm.all_reduce_sum_f64(local_crossings);
        let global_mass = comm.all_reduce_sum_f64(local_mass);
        let global_total = comm.all_reduce_sum_f64(plane.total_crossings as f64);

        // Compute time-averaged rates over the reporting window.
        let dt = atoms.dt;
        let window_steps = step - plane.window_start_step;
        let window_time = window_steps as f64 * dt;

        let mass_flow_rate = if window_time > 0.0 {
            global_mass / window_time
        } else {
            0.0
        };
        let crossing_rate = if window_time > 0.0 {
            global_crossings / window_time
        } else {
            0.0
        };

        // Push to thermo for output.
        thermo.set(&format!("crossings_{}", plane.name), global_total);
        thermo.set(&format!("flow_rate_{}", plane.name), mass_flow_rate);
        thermo.set(&format!("cross_rate_{}", plane.name), crossing_rate);

        if comm.rank() == 0 {
            println!(
                "  [{}] crossings={}, mass_flow_rate={:.6e} kg/s, crossing_rate={:.1} /s",
                plane.name, global_total as u64, mass_flow_rate, crossing_rate,
            );
        }

        // Reset window counters.
        plane.crossings_window = 0;
        plane.mass_window = 0.0;
        plane.window_start_step = step;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_distance() {
        let def = MeasurePlaneDef {
            name: "test".to_string(),
            point: [0.5, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            report_interval: 100,
        };
        let state = MeasurePlaneState::new(&def);

        // Point on positive side of plane
        assert!(state.signed_distance(&[0.6, 0.0, 0.0]) > 0.0);
        // Point on negative side of plane
        assert!(state.signed_distance(&[0.4, 0.0, 0.0]) < 0.0);
        // Point on the plane
        assert!((state.signed_distance(&[0.5, 0.0, 0.0])).abs() < 1e-15);
    }

    #[test]
    fn test_normal_normalization() {
        let def = MeasurePlaneDef {
            name: "test".to_string(),
            point: [0.0, 0.0, 0.0],
            normal: [3.0, 4.0, 0.0],
            report_interval: 100,
        };
        let state = MeasurePlaneState::new(&def);
        let mag = (state.normal[0].powi(2) + state.normal[1].powi(2) + state.normal[2].powi(2)).sqrt();
        assert!((mag - 1.0).abs() < 1e-12);
        assert!((state.normal[0] - 0.6).abs() < 1e-12);
        assert!((state.normal[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_crossing_detection_logic() {
        // Verify sign-change logic
        let prev_dist = -0.1_f64;
        let curr_dist = 0.1_f64;
        // positive crossing
        assert!(prev_dist <= 0.0 && curr_dist > 0.0);

        // no crossing (both positive)
        let prev_dist2 = 0.1_f64;
        let curr_dist2 = 0.2_f64;
        assert!(!(prev_dist2 <= 0.0 && curr_dist2 > 0.0));

        // negative crossing (positive to negative) — not counted
        let prev_dist3 = 0.1_f64;
        let curr_dist3 = -0.1_f64;
        assert!(!(prev_dist3 <= 0.0 && curr_dist3 > 0.0));
    }
}
