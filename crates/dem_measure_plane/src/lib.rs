//! General-purpose measurement plane plugin for MDDEM.
//!
//! Counts particles crossing a defined plane per unit time and reports
//! mass flow rate, particle count, and crossing rate to thermo output.
//!
//! # Configuration
//!
//! ```toml
//! [[measure_plane]]
//! name = "outlet"
//! point = [0.1, 0.0, 0.0]
//! normal = [1.0, 0.0, 0.0]
//! report_interval = 1000
//! ```
//!
//! Multiple `[[measure_plane]]` blocks can be defined. Each plane tracks
//! crossings independently. Results are pushed to thermo as:
//! - `crossings_<name>` — total crossing count (positive direction)
//! - `flow_rate_<name>` — mass flow rate (kg/s) averaged over `report_interval`
//! - `cross_rate_<name>` — particle crossing rate (1/s) averaged over `report_interval`

use std::collections::HashMap;

use mddem_app::prelude::*;
use mddem_core::{Atom, CommResource, Config, RunState};
use mddem_print::Thermo;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

// ── Configuration ───────────────────────────────────────────────────────────

fn default_report_interval() -> usize {
    1000
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[[measure_plane]]` — defines a measurement plane for particle crossing detection.
pub struct MeasurePlaneDef {
    /// Human-readable name for this measurement plane.
    pub name: String,
    /// A point on the plane (3D coordinates).
    pub point: [f64; 3],
    /// Outward normal of the plane (will be normalized).
    pub normal: [f64; 3],
    /// Report interval in timesteps. Crossing rates are averaged over this window.
    #[serde(default = "default_report_interval")]
    pub report_interval: usize,
}

// ── Runtime state ───────────────────────────────────────────────────────────

/// Per-plane runtime state for crossing detection.
struct MeasurePlaneState {
    /// Plane definition.
    name: String,
    point: [f64; 3],
    normal: [f64; 3],
    report_interval: usize,

    /// Signed distance of each tracked particle at the previous step.
    /// Key: atom tag, Value: signed distance from plane.
    prev_signed_dist: HashMap<u32, f64>,

    /// Cumulative crossings in the positive normal direction since last report.
    crossings_window: u64,
    /// Cumulative mass crossed in the positive normal direction since last report.
    mass_window: f64,
    /// Total cumulative crossings (never reset).
    total_crossings: u64,
    /// Step at which the last report window started.
    window_start_step: usize,
}

impl MeasurePlaneState {
    fn new(def: &MeasurePlaneDef) -> Self {
        // Normalize the normal vector.
        let mag = (def.normal[0].powi(2) + def.normal[1].powi(2) + def.normal[2].powi(2)).sqrt();
        let normal = if mag > 1e-30 {
            [def.normal[0] / mag, def.normal[1] / mag, def.normal[2] / mag]
        } else {
            [1.0, 0.0, 0.0] // fallback
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

    /// Compute signed distance from the plane for a given position.
    #[inline]
    fn signed_distance(&self, pos: &[f64; 3]) -> f64 {
        let dx = pos[0] - self.point[0];
        let dy = pos[1] - self.point[1];
        let dz = pos[2] - self.point[2];
        dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2]
    }
}

/// Resource holding all measurement plane states.
pub struct MeasurePlanes {
    planes: Vec<MeasurePlaneState>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers measurement plane systems for particle crossing detection and throughput tracking.
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
fn measure_plane_detect_crossings(atoms: Res<Atom>, mut planes: ResMut<MeasurePlanes>) {
    let nlocal = atoms.nlocal as usize;

    for plane in planes.planes.iter_mut() {
        for i in 0..nlocal {
            let tag = atoms.tag[i];
            let dist = plane.signed_distance(&atoms.pos[i]);

            if let Some(&prev_dist) = plane.prev_signed_dist.get(&tag) {
                // Detect crossing: previous was negative (or zero), now positive.
                if prev_dist <= 0.0 && dist > 0.0 {
                    plane.crossings_window += 1;
                    plane.total_crossings += 1;
                    plane.mass_window += atoms.mass[i];
                }
            }

            plane.prev_signed_dist.insert(tag, dist);
        }
    }
}

/// Report measurement plane statistics to thermo output.
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

        // MPI reduce crossings and mass across all ranks.
        let local_crossings = plane.crossings_window as f64;
        let local_mass = plane.mass_window;
        let global_crossings = comm.all_reduce_sum_f64(local_crossings);
        let global_mass = comm.all_reduce_sum_f64(local_mass);
        let global_total = comm.all_reduce_sum_f64(plane.total_crossings as f64);

        // Compute rates over the window.
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
