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
//! shape = "circular"        # optional: "infinite" (default) or "circular"
//! radius = 0.01             # required when shape = "circular"
//! rolling_windows = 5       # optional: number of windows for rolling average
//! per_type = true           # optional: report per-atom-type statistics
//! ```
//!
//! Multiple `[[measure_plane]]` blocks can be defined. Each plane tracks
//! crossings independently. Results are pushed to thermo as:
//! - `crossings_<name>` — total crossing count (positive direction)
//! - `flow_rate_<name>` — mass flow rate (kg/s) averaged over `report_interval`
//! - `cross_rate_<name>` — particle crossing rate (1/s) averaged over `report_interval`
//! - `avg_flow_rate_<name>` — rolling average mass flow rate (when `rolling_windows > 1`)
//! - `crossings_<name>_type<T>` — per-type crossing count (when `per_type = true`)
//! - `flow_rate_<name>_type<T>` — per-type mass flow rate (when `per_type = true`)

use std::collections::{HashMap, VecDeque};

use mddem_app::prelude::*;
use mddem_core::{Atom, CommResource, Config, RunState};
use mddem_print::Thermo;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

// ── Configuration ───────────────────────────────────────────────────────────

fn default_report_interval() -> usize {
    1000
}

fn default_shape() -> String {
    "infinite".to_string()
}

fn default_rolling_windows() -> usize {
    1
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
    /// Shape of the measurement plane: "infinite" (default) or "circular".
    #[serde(default = "default_shape")]
    pub shape: String,
    /// Radius of the circular measurement plane [m]. Required when shape = "circular".
    #[serde(default)]
    pub radius: Option<f64>,
    /// Number of reporting windows to average over for rolling statistics.
    /// 1 = no averaging (default). >1 = rolling average over the last N windows.
    #[serde(default = "default_rolling_windows")]
    pub rolling_windows: usize,
    /// If true, report crossings broken down by atom_type.
    #[serde(default)]
    pub per_type: bool,
}

// ── Runtime state ───────────────────────────────────────────────────────────

/// Per-plane runtime state for crossing detection.
struct MeasurePlaneState {
    /// Plane definition.
    name: String,
    point: [f64; 3],
    normal: [f64; 3],
    report_interval: usize,

    /// Whether this plane is circular (true) or infinite (false).
    circular: bool,
    /// Squared radius for circular planes (avoids sqrt in hot path).
    radius_sq: f64,
    /// Number of windows for rolling average.
    rolling_windows: usize,
    /// Whether to track per-type statistics.
    per_type: bool,

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

    /// Ring buffer of past window values for rolling average: (crossings, mass, window_time).
    history: VecDeque<(f64, f64, f64)>,

    /// Per-type crossing count within current window. Key: atom_type.
    crossings_by_type: HashMap<u32, u64>,
    /// Per-type mass within current window. Key: atom_type.
    mass_by_type: HashMap<u32, f64>,
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

        let circular = def.shape == "circular";
        let radius_sq = if circular {
            let r = def.radius.unwrap_or_else(|| {
                eprintln!(
                    "ERROR: measure_plane '{}' has shape='circular' but no 'radius' specified",
                    def.name
                );
                std::process::exit(1);
            });
            r * r
        } else {
            0.0
        };

        MeasurePlaneState {
            name: def.name.clone(),
            point: def.point,
            normal,
            report_interval: def.report_interval,
            circular,
            radius_sq,
            rolling_windows: def.rolling_windows.max(1),
            per_type: def.per_type,
            prev_signed_dist: HashMap::new(),
            crossings_window: 0,
            mass_window: 0.0,
            total_crossings: 0,
            window_start_step: 0,
            history: VecDeque::new(),
            crossings_by_type: HashMap::new(),
            mass_by_type: HashMap::new(),
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

    /// Check if a position is within the circular region of this plane.
    /// Always returns true for infinite planes.
    #[inline]
    fn in_bounds(&self, pos: &[f64; 3]) -> bool {
        if !self.circular {
            return true;
        }
        let dx = pos[0] - self.point[0];
        let dy = pos[1] - self.point[1];
        let dz = pos[2] - self.point[2];
        // Project out the normal component to get in-plane displacement
        let proj_n = dx * self.normal[0] + dy * self.normal[1] + dz * self.normal[2];
        let in_plane_sq = dx * dx + dy * dy + dz * dz - proj_n * proj_n;
        in_plane_sq <= self.radius_sq
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
# report_interval = 1000
# shape = "infinite"        # or "circular" (requires 'radius')
# radius = 0.01
# rolling_windows = 1       # >1 for rolling average statistics
# per_type = false           # true to report per-atom-type statistics"#,
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
            let pos = &atoms.pos[i];

            // Check circular bounds before computing distance
            if !plane.in_bounds(pos) {
                continue;
            }

            let dist = plane.signed_distance(pos);

            if let Some(&prev_dist) = plane.prev_signed_dist.get(&tag) {
                // Detect crossing: previous was negative (or zero), now positive.
                if prev_dist <= 0.0 && dist > 0.0 {
                    plane.crossings_window += 1;
                    plane.total_crossings += 1;
                    plane.mass_window += atoms.mass[i];

                    // Per-type tracking
                    if plane.per_type {
                        let atype = atoms.atom_type[i];
                        *plane.crossings_by_type.entry(atype).or_insert(0) += 1;
                        *plane.mass_by_type.entry(atype).or_insert(0.0) += atoms.mass[i];
                    }
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

        // Rolling average statistics
        if plane.rolling_windows > 1 {
            plane
                .history
                .push_back((global_crossings, global_mass, window_time));
            while plane.history.len() > plane.rolling_windows {
                plane.history.pop_front();
            }

            let (total_cross, total_mass, total_time) =
                plane
                    .history
                    .iter()
                    .fold((0.0, 0.0, 0.0), |(c, m, t), &(wc, wm, wt)| {
                        (c + wc, m + wm, t + wt)
                    });

            let avg_flow_rate = if total_time > 0.0 {
                total_mass / total_time
            } else {
                0.0
            };
            let avg_cross_rate = if total_time > 0.0 {
                total_cross / total_time
            } else {
                0.0
            };

            thermo.set(
                &format!("avg_flow_rate_{}", plane.name),
                avg_flow_rate,
            );
            thermo.set(
                &format!("avg_cross_rate_{}", plane.name),
                avg_cross_rate,
            );
        }

        // Per-type statistics
        if plane.per_type {
            // Collect all type keys, reduce each
            let type_keys: Vec<u32> = plane.crossings_by_type.keys().copied().collect();
            for atype in &type_keys {
                let local_tc = *plane.crossings_by_type.get(atype).unwrap_or(&0) as f64;
                let local_tm = *plane.mass_by_type.get(atype).unwrap_or(&0.0);
                let global_tc = comm.all_reduce_sum_f64(local_tc);
                let global_tm = comm.all_reduce_sum_f64(local_tm);

                let type_flow_rate = if window_time > 0.0 {
                    global_tm / window_time
                } else {
                    0.0
                };

                thermo.set(
                    &format!("crossings_{}_type{}", plane.name, atype),
                    global_tc,
                );
                thermo.set(
                    &format!("flow_rate_{}_type{}", plane.name, atype),
                    type_flow_rate,
                );
            }

            plane.crossings_by_type.clear();
            plane.mass_by_type.clear();
        }

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
            shape: "infinite".to_string(),
            radius: None,
            rolling_windows: 1,
            per_type: false,
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
            shape: "infinite".to_string(),
            radius: None,
            rolling_windows: 1,
            per_type: false,
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

    #[test]
    fn test_circular_plane_in_bounds() {
        let def = MeasurePlaneDef {
            name: "circ".to_string(),
            point: [0.0, 0.0, 0.5],
            normal: [0.0, 0.0, 1.0],
            report_interval: 100,
            shape: "circular".to_string(),
            radius: Some(0.01),
            rolling_windows: 1,
            per_type: false,
        };
        let state = MeasurePlaneState::new(&def);

        // Point within circular region (at center)
        assert!(state.in_bounds(&[0.0, 0.0, 0.5]));
        // Point within circular region (near edge)
        assert!(state.in_bounds(&[0.005, 0.005, 0.5]));
        // Point outside circular region
        assert!(!state.in_bounds(&[0.02, 0.0, 0.5]));
        // Point at different z but within radial bounds
        assert!(state.in_bounds(&[0.005, 0.0, 0.6]));
    }

    #[test]
    fn test_infinite_plane_always_in_bounds() {
        let def = MeasurePlaneDef {
            name: "inf".to_string(),
            point: [0.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            report_interval: 100,
            shape: "infinite".to_string(),
            radius: None,
            rolling_windows: 1,
            per_type: false,
        };
        let state = MeasurePlaneState::new(&def);

        assert!(state.in_bounds(&[100.0, 200.0, 300.0]));
        assert!(state.in_bounds(&[-50.0, -50.0, -50.0]));
    }

    #[test]
    fn test_rolling_history() {
        let def = MeasurePlaneDef {
            name: "test".to_string(),
            point: [0.0; 3],
            normal: [1.0, 0.0, 0.0],
            report_interval: 100,
            shape: "infinite".to_string(),
            radius: None,
            rolling_windows: 3,
            per_type: false,
        };
        let mut state = MeasurePlaneState::new(&def);
        assert_eq!(state.rolling_windows, 3);

        // Simulate pushing window data
        state.history.push_back((10.0, 0.5, 1.0));
        state.history.push_back((20.0, 1.0, 1.0));
        state.history.push_back((30.0, 1.5, 1.0));
        assert_eq!(state.history.len(), 3);

        // Push one more — oldest should be evicted when we check
        state.history.push_back((40.0, 2.0, 1.0));
        while state.history.len() > state.rolling_windows {
            state.history.pop_front();
        }
        assert_eq!(state.history.len(), 3);

        // Rolling average should be over windows 2,3,4
        let (total_cross, total_mass, total_time) =
            state
                .history
                .iter()
                .fold((0.0, 0.0, 0.0), |(c, m, t), &(wc, wm, wt)| {
                    (c + wc, m + wm, t + wt)
                });
        assert!((total_cross - 90.0).abs() < 1e-10);
        assert!((total_mass - 4.5).abs() < 1e-10);
        assert!((total_time - 3.0).abs() < 1e-10);
    }
}
