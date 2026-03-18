//! Wall contact forces for DEM simulations using Hertz normal contact with
//! viscous damping and optional adhesion (JKR, DMT, SJKR cohesion).
//!
//! # Wall Types
//!
//! | Type | Description | Config key |
//! |------|-------------|------------|
//! | **Plane** | Infinite flat plane defined by a point and unit normal | `type = "plane"` |
//! | **Cylinder** | Infinite cylinder along X/Y/Z axis with finite axial bounds | `type = "cylinder"` |
//! | **Sphere** | Sphere defined by center and radius | `type = "sphere"` |
//! | **Region** | Any [`Region`] shape used as a wall surface | `type = "region"` |
//!
//! All wall types treat the wall as having infinite mass and infinite radius
//! for contact mechanics, so the effective radius equals the particle radius
//! and the reduced mass equals the particle mass.
//!
//! # Motion Types
//!
//! | Motion | Description |
//! |--------|-------------|
//! | **Static** | Wall does not move (default) |
//! | **Constant velocity** | Wall translates at a fixed velocity each timestep |
//! | **Oscillating** | Sinusoidal displacement along the wall normal |
//! | **Servo** | Proportional controller adjusting velocity to reach a target force |
//!
//! Motion is currently supported only for plane walls.
//!
//! # TOML Configuration
//!
//! Walls are defined as `[[wall]]` array-of-tables entries. Each entry requires
//! a `material` field matching a name in `[[dem.materials]]`.
//!
//! ```toml
//! # Plane wall (floor at z=0, normal pointing up)
//! [[wall]]
//! type = "plane"
//! point_z = 0.0
//! normal_z = 1.0
//! material = "glass"
//! name = "floor"                  # optional, for runtime enable/disable
//!
//! # Cylinder wall (particles confined inside a z-aligned cylinder)
//! [[wall]]
//! type = "cylinder"
//! axis = "z"
//! center = [0.005, 0.005]         # center in the XY plane
//! radius = 0.004
//! lo = 0.0                        # axial lo bound (default: -inf)
//! hi = 0.01                       # axial hi bound (default: +inf)
//! inside = true                   # particles live inside the cylinder
//! material = "glass"
//!
//! # Sphere wall (particles confined inside a sphere)
//! [[wall]]
//! type = "sphere"
//! center = [0.005, 0.005, 0.005]
//! radius = 0.004
//! inside = true
//! material = "glass"
//!
//! # Region wall (any Region shape as a wall surface)
//! [[wall]]
//! type = "region"
//! inside = true
//! material = "glass"
//! region = { type = "cone", center = [0.005, 0.005], axis = "z",
//!            rad_lo = 0.004, rad_hi = 0.002, lo = 0.0, hi = 0.01 }
//!
//! # Moving wall with constant velocity
//! [[wall]]
//! type = "plane"
//! normal_z = 1.0
//! material = "glass"
//! velocity = [0.0, 0.0, -0.01]    # [vx, vy, vz]
//!
//! # Oscillating wall (sinusoidal along normal)
//! [[wall]]
//! type = "plane"
//! point_z = 0.1
//! normal_z = 1.0
//! material = "glass"
//! oscillate = { amplitude = 0.001, frequency = 50.0 }
//!
//! # Servo-controlled wall (adjusts velocity to reach target force)
//! [[wall]]
//! type = "plane"
//! point_z = 0.1
//! normal_z = -1.0
//! material = "glass"
//! servo = { target_force = 100.0, max_velocity = 0.1, gain = 0.001 }
//! ```
//!
//! # Plugin Registration
//!
//! Add [`WallPlugin`] to your app. It depends on `DemAtomPlugin` (for
//! [`MaterialTable`] and [`DemAtom`] data).
//!
//! # Systems
//!
//! | System | Schedule | Purpose |
//! |--------|----------|---------|
//! | [`wall_move`] | `PreInitialIntegration` | Updates wall positions from motion modes |
//! | [`wall_zero_force_accumulators`] | `PreForce` | Zeros per-wall force accumulators |
//! | [`wall_contact_force`] | `Force` | Computes Hertz contact + damping + adhesion |

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::{DemAtom, MaterialTable, SQRT_5_3};
use mddem_core::region::Region;
use mddem_core::{Atom, AtomDataRegistry, Config, ScheduleSet};

fn default_neg_inf() -> f64 {
    f64::NEG_INFINITY
}
fn default_pos_inf() -> f64 {
    f64::INFINITY
}
fn default_wall_type() -> String {
    "plane".to_string()
}

// ── Config structs ──────────────────────────────────────────────────────────

/// Sinusoidal oscillation parameters for a wall.
///
/// The wall displaces along its normal as `amplitude * sin(2π * frequency * t)`.
/// Velocity is computed analytically as the time derivative.
///
/// # TOML
/// ```toml
/// oscillate = { amplitude = 0.001, frequency = 50.0 }
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OscillateDef {
    /// Peak displacement from the origin position (meters).
    pub amplitude: f64,
    /// Oscillation frequency (Hz).
    pub frequency: f64,
}

/// Proportional servo controller parameters for a wall.
///
/// Each timestep the servo computes `error = target_force - measured_force`,
/// then sets `velocity = clamp(gain * error, -max_velocity, max_velocity)`
/// along the wall normal. This drives the wall toward a steady-state contact
/// force equal to `target_force`.
///
/// # TOML
/// ```toml
/// servo = { target_force = 100.0, max_velocity = 0.1, gain = 0.001 }
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ServoDef {
    /// Desired total contact force on this wall (N).
    pub target_force: f64,
    /// Maximum wall velocity magnitude (m/s), prevents overshooting.
    pub max_velocity: f64,
    /// Proportional gain (m/s per N of force error).
    pub gain: f64,
}

/// TOML definition of a single wall entry (`[[wall]]`).
///
/// This is a union struct: different fields are relevant depending on `type`.
/// See the [module-level docs](self) for TOML examples of each wall type.
#[derive(Deserialize, Clone)]
pub struct WallDef {
    /// Wall type: `"plane"` (default), `"cylinder"`, `"sphere"`, or `"region"`.
    #[serde(default = "default_wall_type", rename = "type")]
    pub wall_type: String,

    // ── Plane fields ────────────────────────────────────────────────────
    /// X-coordinate of a point on the plane (default: 0.0).
    #[serde(default)]
    pub point_x: f64,
    /// Y-coordinate of a point on the plane (default: 0.0).
    #[serde(default)]
    pub point_y: f64,
    /// Z-coordinate of a point on the plane (default: 0.0).
    #[serde(default)]
    pub point_z: f64,
    /// X-component of the outward normal vector (will be normalized).
    #[serde(default)]
    pub normal_x: f64,
    /// Y-component of the outward normal vector (will be normalized).
    #[serde(default)]
    pub normal_y: f64,
    /// Z-component of the outward normal vector (will be normalized).
    #[serde(default)]
    pub normal_z: f64,

    // ── Cylinder/sphere fields ──────────────────────────────────────────
    /// Cylinder axis: `"x"`, `"y"`, or `"z"` (default: `"z"`).
    #[serde(default)]
    pub axis: Option<String>,
    /// Center coordinates: `[c0, c1]` for cylinder (in the plane ⊥ to axis),
    /// or `[x, y, z]` for sphere.
    #[serde(default)]
    pub center: Option<Vec<f64>>,
    /// Radius of the cylinder or sphere wall surface (meters).
    #[serde(default)]
    pub radius: Option<f64>,
    /// Lower axial bound for cylinder (default: −∞).
    #[serde(default)]
    pub lo: Option<f64>,
    /// Upper axial bound for cylinder (default: +∞).
    #[serde(default)]
    pub hi: Option<f64>,
    /// If `true`, particles live inside the wall surface and the contact
    /// normal points inward. If `false` (default), particles are outside.
    #[serde(default)]
    pub inside: Option<bool>,

    // ── Common fields ───────────────────────────────────────────────────
    /// Material name — must match a `[[dem.materials]]` entry.
    pub material: String,
    /// Optional name for runtime enable/disable via [`Walls::deactivate_by_name`].
    #[serde(default)]
    pub name: Option<String>,
    /// Lower X bound for the plane wall's active region (default: −∞).
    #[serde(default = "default_neg_inf")]
    pub bound_x_low: f64,
    /// Upper X bound for the plane wall's active region (default: +∞).
    #[serde(default = "default_pos_inf")]
    pub bound_x_high: f64,
    /// Lower Y bound for the plane wall's active region (default: −∞).
    #[serde(default = "default_neg_inf")]
    pub bound_y_low: f64,
    /// Upper Y bound for the plane wall's active region (default: +∞).
    #[serde(default = "default_pos_inf")]
    pub bound_y_high: f64,
    /// Lower Z bound for the plane wall's active region (default: −∞).
    #[serde(default = "default_neg_inf")]
    pub bound_z_low: f64,
    /// Upper Z bound for the plane wall's active region (default: +∞).
    #[serde(default = "default_pos_inf")]
    pub bound_z_high: f64,
    /// Constant wall velocity `[vx, vy, vz]` (m/s). Plane walls only.
    #[serde(default)]
    pub velocity: Option<[f64; 3]>,
    /// Sinusoidal oscillation parameters. Plane walls only.
    #[serde(default)]
    pub oscillate: Option<OscillateDef>,
    /// Servo controller parameters. Plane walls only.
    #[serde(default)]
    pub servo: Option<ServoDef>,
    /// Region definition for `type = "region"` walls.
    #[serde(default)]
    pub region: Option<Region>,
    /// Wall temperature in K (None = no wall heat transfer).
    #[serde(default)]
    pub temperature: Option<f64>,
}

// ── Runtime types ───────────────────────────────────────────────────────────

/// Wall motion mode at runtime (plane walls only).
pub enum WallMotion {
    /// Wall is stationary.
    Static,
    /// Wall translates at a constant velocity each timestep.
    ConstantVelocity,
    /// Wall oscillates sinusoidally: `displacement = amplitude * sin(2π * frequency * t)`.
    Oscillate {
        /// Peak displacement (meters).
        amplitude: f64,
        /// Oscillation frequency (Hz).
        frequency: f64,
    },
    /// Proportional servo: velocity = clamp(gain × (target_force − measured_force), ±max_velocity).
    Servo {
        /// Desired total contact force (N).
        target_force: f64,
        /// Maximum wall speed (m/s).
        max_velocity: f64,
        /// Proportional gain (m/s per N).
        gain: f64,
    },
}

/// Runtime representation of an infinite plane wall.
///
/// Contact geometry for a plane wall:
/// ```text
///          ·  particle center (px, py, pz)
///         /|
///        / |  distance = dot(P - W, n̂)
///       /  |
///   n̂ ↑   |  delta (overlap) = radius - distance
///      |   |
///  ────W───┴──── wall surface
///      (point_x, point_y, point_z)
/// ```
///
/// The signed distance from the particle center to the wall plane is
/// `dot(pos - point, normal)`. Overlap is `delta = radius - distance`.
/// Force is applied only when `delta > 0` (or within JKR adhesion range).
pub struct WallPlane {
    /// Current X-coordinate of a point on the plane (updated by motion).
    pub point_x: f64,
    /// Current Y-coordinate of a point on the plane (updated by motion).
    pub point_y: f64,
    /// Current Z-coordinate of a point on the plane (updated by motion).
    pub point_z: f64,
    /// X-component of the unit outward normal.
    pub normal_x: f64,
    /// Y-component of the unit outward normal.
    pub normal_y: f64,
    /// Z-component of the unit outward normal.
    pub normal_z: f64,
    /// Index into [`MaterialTable`] for this wall's material properties.
    pub material_index: usize,
    /// Optional name for runtime enable/disable.
    pub name: Option<String>,
    /// Lower X bound of the wall's active region.
    pub bound_x_low: f64,
    /// Upper X bound of the wall's active region.
    pub bound_x_high: f64,
    /// Lower Y bound of the wall's active region.
    pub bound_y_low: f64,
    /// Upper Y bound of the wall's active region.
    pub bound_y_high: f64,
    /// Lower Z bound of the wall's active region.
    pub bound_z_low: f64,
    /// Upper Z bound of the wall's active region.
    pub bound_z_high: f64,
    /// Current wall velocity `[vx, vy, vz]` (m/s).
    pub velocity: [f64; 3],
    /// Motion mode (static, constant velocity, oscillating, or servo).
    pub motion: WallMotion,
    /// Initial position of the wall point (reference for oscillation).
    pub origin: [f64; 3],
    /// Accumulated scalar contact force this timestep (along normal),
    /// used by the servo controller to compute the feedback error.
    pub force_accumulator: f64,
    /// Wall temperature in K (None = no wall heat transfer).
    pub temperature: Option<f64>,
}

impl WallPlane {
    /// Check if atom position is within the wall's bounding region.
    #[inline]
    pub fn in_bounds(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.bound_x_low
            && x <= self.bound_x_high
            && y >= self.bound_y_low
            && y <= self.bound_y_high
            && z >= self.bound_z_low
            && z <= self.bound_z_high
    }
}

/// Runtime representation of a cylindrical wall aligned to one axis.
///
/// Contact geometry (cross-section, `inside = true`):
/// ```text
///        ╭──────────╮  cylinder surface (radius R)
///       ╱            ╲
///      │   ← n̂  ·    │  radial_dist from axis to particle
///      │        ↑     │  gap = R - radial_dist
///       ╲      particle╱  delta = particle_radius - gap
///        ╰──────────╯
/// ```
///
/// For `inside = true`, the normal points inward (toward the axis).
/// For `inside = false`, the normal points outward (away from the axis).
/// Particles outside the `[lo, hi]` axial range are ignored.
pub struct WallCylinder {
    /// Axis index: 0 = X, 1 = Y, 2 = Z.
    pub axis: usize,
    /// Center of the cylinder in the 2D plane perpendicular to the axis.
    /// For a Z-cylinder, this is `[center_x, center_y]`.
    pub center: [f64; 2],
    /// Cylinder radius (meters).
    pub radius: f64,
    /// Lower axial bound — particles below this are ignored.
    pub lo: f64,
    /// Upper axial bound — particles above this are ignored.
    pub hi: f64,
    /// If `true`, particles live inside the cylinder and the contact normal
    /// points inward (toward the axis).
    pub inside: bool,
    /// Index into [`MaterialTable`] for this wall's material properties.
    pub material_index: usize,
    /// Optional name for runtime enable/disable.
    pub name: Option<String>,
    /// Accumulated scalar contact force this timestep.
    pub force_accumulator: f64,
    /// Wall temperature in K (None = no wall heat transfer).
    pub temperature: Option<f64>,
}

/// Runtime representation of a spherical wall.
///
/// Contact geometry (`inside = true`):
/// ```text
///         ╭─────╮  sphere surface (radius R)
///        ╱       ╲
///       │  ← n̂  · │  dist = |pos - center|
///       │       ↑  │  gap = R - dist
///        ╲  particle╱  delta = particle_radius - gap
///         ╰─────╯
/// ```
///
/// For `inside = true`, the normal points inward (toward the center).
/// For `inside = false`, the normal points outward (away from the center).
pub struct WallSphere {
    /// Sphere center `[x, y, z]` (meters).
    pub center: [f64; 3],
    /// Sphere radius (meters).
    pub radius: f64,
    /// If `true`, particles live inside the sphere and the contact normal
    /// points inward (toward the center).
    pub inside: bool,
    /// Index into [`MaterialTable`] for this wall's material properties.
    pub material_index: usize,
    /// Optional name for runtime enable/disable.
    pub name: Option<String>,
    /// Accumulated scalar contact force this timestep.
    pub force_accumulator: f64,
    /// Wall temperature in K (None = no wall heat transfer).
    pub temperature: Option<f64>,
}

/// Runtime representation of a region-surface wall.
///
/// Uses any [`Region`] shape (block, sphere, cylinder, cone, union,
/// intersect, etc.) as a wall surface. Contact detection delegates to
/// [`Region::closest_point_on_surface`] to get the signed distance and
/// outward normal for each particle.
pub struct WallRegion {
    /// The region whose surface acts as the wall.
    pub region: Region,
    /// If `true`, particles live inside the region and the contact normal
    /// points inward (away from the surface, toward the interior).
    pub inside: bool,
    /// Index into [`MaterialTable`] for this wall's material properties.
    pub material_index: usize,
    /// Optional name for runtime enable/disable.
    pub name: Option<String>,
    /// Accumulated scalar contact force this timestep.
    pub force_accumulator: f64,
    /// Wall temperature in K (None = no wall heat transfer).
    pub temperature: Option<f64>,
}

/// Collection of all wall types with per-wall active/inactive flags.
///
/// Stored as a resource in the [`App`]. Individual walls can be enabled or
/// disabled at runtime via [`deactivate_by_name`](Self::deactivate_by_name).
pub struct Walls {
    /// Plane walls.
    pub planes: Vec<WallPlane>,
    /// Per-plane active flags (parallel to `planes`).
    pub active: Vec<bool>,
    /// Cylinder walls.
    pub cylinders: Vec<WallCylinder>,
    /// Per-cylinder active flags (parallel to `cylinders`).
    pub cylinder_active: Vec<bool>,
    /// Sphere walls.
    pub spheres: Vec<WallSphere>,
    /// Per-sphere active flags (parallel to `spheres`).
    pub sphere_active: Vec<bool>,
    /// Region-surface walls.
    pub regions: Vec<WallRegion>,
    /// Per-region active flags (parallel to `regions`).
    pub region_active: Vec<bool>,
    /// Elapsed simulation time (seconds), used for oscillation phase tracking.
    pub time: f64,
}

impl Walls {
    /// Deactivate all walls (of any type) whose `name` matches the given string.
    ///
    /// Deactivated walls are skipped during contact force computation and
    /// motion updates. This is useful for removing walls mid-simulation
    /// (e.g., removing a lid after compaction).
    pub fn deactivate_by_name(&mut self, name: &str) {
        for (i, wall) in self.planes.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.active[i] = false;
            }
        }
        for (i, wall) in self.cylinders.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.cylinder_active[i] = false;
            }
        }
        for (i, wall) in self.spheres.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.sphere_active[i] = false;
            }
        }
        for (i, wall) in self.regions.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.region_active[i] = false;
            }
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that registers wall contact force systems from `[[wall]]` TOML config.
///
/// Parses all `[[wall]]` entries, resolves material indices, and creates the
/// [`Walls`] resource. Registers three systems:
///
/// - [`wall_move`] — updates wall positions (oscillation, servo, constant velocity)
/// - [`wall_zero_force_accumulators`] — zeros per-wall force accumulators before force pass
/// - [`wall_contact_force`] — computes Hertz + damping + adhesion forces for all wall types
///
/// # Dependencies
///
/// Requires `DemAtomPlugin` (provides [`MaterialTable`] and [`DemAtom`]).
pub struct WallPlugin;

impl Plugin for WallPlugin {
    fn dependencies(&self) -> Vec<&str> {
        vec!["DemAtomPlugin"]
    }

    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Wall definitions (uncomment to add walls)
# [[wall]]
# point_x = 0.0
# point_y = 0.0
# point_z = 0.0
# normal_x = 0.0
# normal_y = 0.0
# normal_z = 1.0
# material = "glass"        # must match a [[dem.materials]] name
# name = "floor"            # optional name for runtime enable/disable
# velocity = [0.0, 0.0, -0.01]  # constant velocity (optional)
# oscillate = { amplitude = 0.001, frequency = 50.0 }  # sinusoidal (optional)
# servo = { target_force = 100.0, max_velocity = 0.1, gain = 0.001 }  # servo (optional)"#,
        )
    }

    fn build(&self, app: &mut App) {
        let walls = {
            let config = app
                .get_resource_ref::<Config>()
                .expect("Config resource must exist");
            let wall_defs: Vec<WallDef> = if let Some(val) = config.table.get("wall") {
                match val {
                    toml::Value::Array(arr) => arr
                        .iter()
                        .enumerate()
                        .map(|(idx, v)| {
                            match v.clone().try_into::<WallDef>() {
                                Ok(w) => w,
                                Err(e) => {
                                    eprintln!("ERROR: failed to parse [[wall]] entry {}: {}", idx, e);
                                    std::process::exit(1);
                                }
                            }
                        })
                        .collect(),
                    toml::Value::Table(t) => {
                        match toml::Value::Table(t.clone()).try_into::<WallDef>() {
                            Ok(w) => vec![w],
                            Err(e) => {
                                eprintln!("ERROR: failed to parse [wall] entry: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    _ => {
                        eprintln!("ERROR: [wall] must be a table or array of tables");
                        std::process::exit(1);
                    }
                }
            } else {
                Vec::new()
            };
            drop(config);

            let material_table = app
                .get_resource_ref::<MaterialTable>()
                .expect("MaterialTable must exist before WallPlugin — add DemAtomPlugin first");

            let mut planes = Vec::new();
            let mut cylinders = Vec::new();
            let mut spheres = Vec::new();
            let mut regions: Vec<WallRegion> = Vec::new();

            for w in &wall_defs {
                let mat_idx = match material_table.find_material(&w.material) {
                    Some(idx) => idx as usize,
                    None => {
                        eprintln!(
                            "ERROR: wall material '{}' not found in [[dem.materials]]. Available: {:?}",
                            w.material, material_table.names
                        );
                        std::process::exit(1);
                    }
                };

                match w.wall_type.as_str() {
                    "cylinder" => {
                        let axis_str = w.axis.as_deref().unwrap_or("z");
                        let axis = match axis_str {
                            "x" | "X" => 0,
                            "y" | "Y" => 1,
                            "z" | "Z" => 2,
                            _ => {
                                eprintln!("ERROR: cylinder wall axis must be x, y, or z, got '{}'", axis_str);
                                std::process::exit(1);
                            }
                        };
                        let center_vec = w.center.as_ref().expect("cylinder wall requires 'center' [c0, c1]");
                        if center_vec.len() < 2 {
                            eprintln!("ERROR: cylinder wall 'center' must have 2 elements");
                            std::process::exit(1);
                        }
                        let center = [center_vec[0], center_vec[1]];
                        let radius = w.radius.expect("cylinder wall requires 'radius'");
                        let lo = w.lo.unwrap_or(f64::NEG_INFINITY);
                        let hi = w.hi.unwrap_or(f64::INFINITY);
                        let inside = w.inside.unwrap_or(false);
                        cylinders.push(WallCylinder {
                            axis,
                            center,
                            radius,
                            lo,
                            hi,
                            inside,
                            material_index: mat_idx,
                            name: w.name.clone(),
                            force_accumulator: 0.0,
                            temperature: w.temperature,
                        });
                    }
                    "sphere" => {
                        let center_vec = w.center.as_ref().expect("sphere wall requires 'center' [x, y, z]");
                        if center_vec.len() < 3 {
                            eprintln!("ERROR: sphere wall 'center' must have 3 elements");
                            std::process::exit(1);
                        }
                        let center = [center_vec[0], center_vec[1], center_vec[2]];
                        let radius = w.radius.expect("sphere wall requires 'radius'");
                        let inside = w.inside.unwrap_or(false);
                        spheres.push(WallSphere {
                            center,
                            radius,
                            inside,
                            material_index: mat_idx,
                            name: w.name.clone(),
                            force_accumulator: 0.0,
                            temperature: w.temperature,
                        });
                    }
                    "region" => {
                        let region = w.region.clone().unwrap_or_else(|| {
                            eprintln!("ERROR: region wall requires a 'region' field");
                            std::process::exit(1);
                        });
                        let inside = w.inside.unwrap_or(false);
                        regions.push(WallRegion {
                            region,
                            inside,
                            material_index: mat_idx,
                            name: w.name.clone(),
                            force_accumulator: 0.0,
                            temperature: w.temperature,
                        });
                    }
                    // Default to plane for unrecognized types (backwards compatibility)
                    "plane" | _ => {
                        let mag =
                            (w.normal_x * w.normal_x + w.normal_y * w.normal_y + w.normal_z * w.normal_z)
                                .sqrt();
                        if mag <= 1e-15 {
                            eprintln!("ERROR: wall normal vector must be non-zero (wall material '{}')", w.material);
                            std::process::exit(1);
                        }
                        let nx = w.normal_x / mag;
                        let ny = w.normal_y / mag;
                        let nz = w.normal_z / mag;

                        let (motion, velocity) = if let Some(ref osc) = w.oscillate {
                            (
                                WallMotion::Oscillate {
                                    amplitude: osc.amplitude,
                                    frequency: osc.frequency,
                                },
                                [0.0; 3],
                            )
                        } else if let Some(ref srv) = w.servo {
                            (
                                WallMotion::Servo {
                                    target_force: srv.target_force,
                                    max_velocity: srv.max_velocity,
                                    gain: srv.gain,
                                },
                                [0.0; 3],
                            )
                        } else if let Some(vel) = w.velocity {
                            (WallMotion::ConstantVelocity, vel)
                        } else {
                            (WallMotion::Static, [0.0; 3])
                        };

                        planes.push(WallPlane {
                            point_x: w.point_x,
                            point_y: w.point_y,
                            point_z: w.point_z,
                            normal_x: nx,
                            normal_y: ny,
                            normal_z: nz,
                            material_index: mat_idx,
                            name: w.name.clone(),
                            bound_x_low: w.bound_x_low,
                            bound_x_high: w.bound_x_high,
                            bound_y_low: w.bound_y_low,
                            bound_y_high: w.bound_y_high,
                            bound_z_low: w.bound_z_low,
                            bound_z_high: w.bound_z_high,
                            velocity,
                            motion,
                            origin: [w.point_x, w.point_y, w.point_z],
                            force_accumulator: 0.0,
                            temperature: w.temperature,
                        });
                    }
                }
            }
            drop(material_table);

            let np = planes.len();
            let nc = cylinders.len();
            let ns = spheres.len();
            let nr = regions.len();
            Walls {
                planes,
                active: vec![true; np],
                cylinders,
                cylinder_active: vec![true; nc],
                spheres,
                sphere_active: vec![true; ns],
                regions,
                region_active: vec![true; nr],
                time: 0.0,
            }
        };

        app.add_resource(walls);
        app.add_update_system(wall_move, ScheduleSet::PreInitialIntegration);
        app.add_update_system(wall_zero_force_accumulators, ScheduleSet::PreForce);
        app.add_update_system(wall_contact_force.label("wall_contact"), ScheduleSet::Force);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Update wall positions and velocities according to their motion mode.
///
/// Runs in [`ScheduleSet::PreInitialIntegration`] so walls are moved
/// *before* the integration step each timestep. Advances `walls.time` by `dt`.
pub fn wall_move(mut walls: ResMut<Walls>, atoms: Res<Atom>) {
    let dt = atoms.dt;
    let time = walls.time;

    let nplanes = walls.planes.len();
    for idx in 0..nplanes {
        if !walls.active[idx] {
            continue;
        }

        let wall = &mut walls.planes[idx];
        match wall.motion {
            WallMotion::Static => {}
            WallMotion::ConstantVelocity => {
                wall.point_x += wall.velocity[0] * dt;
                wall.point_y += wall.velocity[1] * dt;
                wall.point_z += wall.velocity[2] * dt;
            }
            WallMotion::Oscillate { amplitude, frequency } => {
                let phase = 2.0 * std::f64::consts::PI * frequency * (time + dt);
                let disp = amplitude * phase.sin();
                wall.point_x = wall.origin[0] + disp * wall.normal_x;
                wall.point_y = wall.origin[1] + disp * wall.normal_y;
                wall.point_z = wall.origin[2] + disp * wall.normal_z;
                // Velocity = d(disp)/dt = amplitude * 2*pi*freq * cos(phase)
                let vel_mag = amplitude * 2.0 * std::f64::consts::PI * frequency * phase.cos();
                wall.velocity = [
                    vel_mag * wall.normal_x,
                    vel_mag * wall.normal_y,
                    vel_mag * wall.normal_z,
                ];
            }
            WallMotion::Servo { target_force, max_velocity, gain } => {
                let error = target_force - wall.force_accumulator;
                let vel_mag = (gain * error).clamp(-max_velocity, max_velocity);
                wall.velocity = [
                    vel_mag * wall.normal_x,
                    vel_mag * wall.normal_y,
                    vel_mag * wall.normal_z,
                ];
                wall.point_x += wall.velocity[0] * dt;
                wall.point_y += wall.velocity[1] * dt;
                wall.point_z += wall.velocity[2] * dt;
            }
        }
    }

    walls.time += dt;
}

/// Zero all per-wall force accumulators before the force computation pass.
///
/// Runs in [`ScheduleSet::PreForce`]. The accumulators are summed during
/// [`wall_contact_force`] and read by servo controllers in the next
/// [`wall_move`] call.
pub fn wall_zero_force_accumulators(mut walls: ResMut<Walls>) {
    for wall in &mut walls.planes {
        wall.force_accumulator = 0.0;
    }
    for wall in &mut walls.cylinders {
        wall.force_accumulator = 0.0;
    }
    for wall in &mut walls.spheres {
        wall.force_accumulator = 0.0;
    }
    for wall in &mut walls.regions {
        wall.force_accumulator = 0.0;
    }
}

/// Compute wall–particle contact forces for all wall types.
///
/// For each local atom and each active wall, this system:
/// 1. Computes the signed distance and contact normal
/// 2. Determines the overlap `delta = radius - gap`
/// 3. Applies Hertz elastic force: `F_n = (4/3) * E_eff * sqrt(R_eff * delta) * delta`
/// 4. Adds viscous damping: `F_diss = 2 * beta * sqrt(5/3) * sqrt(S_n * m) * v_n`
/// 5. Optionally adds adhesion (JKR, DMT, or SJKR cohesion)
/// 6. Applies twisting friction torque (plane walls only)
/// 7. Accumulates the scalar contact force for servo control
///
/// Runs in [`ScheduleSet::Force`].
pub fn wall_contact_force(
    mut atoms: ResMut<Atom>,
    mut walls: ResMut<Walls>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let mut dem = registry.expect_mut::<DemAtom>("wall_contact_force");

    let nlocal = atoms.nlocal as usize;

    // Collect per-wall forces to accumulate after the loop
    let nwalls = walls.planes.len();
    let mut wall_forces = vec![0.0f64; nwalls];

    for (wall_idx, wall) in walls.planes.iter().enumerate() {
        if !walls.active[wall_idx] {
            continue;
        }

        let wall_mat = wall.material_index;

        for i in 0..nlocal {
            let px = atoms.pos[i][0];
            let py = atoms.pos[i][1];
            let pz = atoms.pos[i][2];

            // Check if atom is within the wall's bounding region
            if !wall.in_bounds(px, py, pz) {
                continue;
            }

            // Vector from wall point to atom position
            let dx = px - wall.point_x;
            let dy = py - wall.point_y;
            let dz = pz - wall.point_z;

            // Signed distance from atom to wall plane (positive = on normal side)
            let distance = dx * wall.normal_x + dy * wall.normal_y + dz * wall.normal_z;

            // Only apply force when atom center is on the normal side of the wall
            if distance <= 0.0 {
                continue;
            }

            let radius = dem.radius[i];
            let mat_i = atoms.atom_type[i] as usize;

            // Wall has infinite radius → r_eff = r_particle
            let r_eff = radius;
            let e_eff = material_table.e_eff_ij[mat_i][wall_mat];
            let surface_energy = material_table.surface_energy_ij[mat_i][wall_mat];

            let use_dmt = material_table.adhesion_model == "dmt";

            // JKR pull-off distance for extended interaction range
            // DMT: no extended range (particles separate at delta = 0)
            let delta_pulloff = if surface_energy > 0.0 && !use_dmt {
                let gamma = surface_energy;
                (std::f64::consts::PI * std::f64::consts::PI * gamma * gamma * r_eff
                    / (4.0 * e_eff * e_eff))
                    .cbrt()
            } else {
                0.0
            };

            let delta = (radius - distance).min(0.5 * radius);

            // Skip if no contact and no JKR adhesion range
            if delta <= 0.0 && surface_energy <= 0.0 {
                continue;
            }
            if delta < -delta_pulloff {
                continue;
            }

            // JKR adhesion-only regime; DMT has no adhesion-only regime
            let jkr_adhesion_only = surface_energy > 0.0 && !use_dmt && delta <= 0.0;

            // Hertz stiffness (only when delta > 0)
            let (s_n, k_n) = if delta > 0.0 {
                let sdr = (delta * r_eff).sqrt();
                (2.0 * e_eff * sdr, 4.0 / 3.0 * e_eff * sdr)
            } else {
                (0.0, 0.0)
            };

            // Wall has infinite mass → m_reduced = m_particle
            let m_r = atoms.mass[i];

            // Relative velocity along wall normal (subtract wall velocity)
            let v_rel_x = atoms.vel[i][0] - wall.velocity[0];
            let v_rel_y = atoms.vel[i][1] - wall.velocity[1];
            let v_rel_z = atoms.vel[i][2] - wall.velocity[2];
            let v_n = v_rel_x * wall.normal_x
                + v_rel_y * wall.normal_y
                + v_rel_z * wall.normal_z;

            let beta = material_table.beta_ij[mat_i][wall_mat];
            let cohesion_energy = material_table.cohesion_energy_ij[mat_i][wall_mat];

            let f_net = if surface_energy > 0.0 && use_dmt {
                // DMT model: pure Hertz contact + constant attractive force
                let f_dmt = 2.0 * std::f64::consts::PI * surface_energy * r_eff;
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                k_n * delta - f_diss - f_dmt
            } else if surface_energy > 0.0 {
                // JKR simplified explicit model
                let f_adhesion = 1.5 * std::f64::consts::PI * surface_energy * r_eff;
                if jkr_adhesion_only {
                    -f_adhesion
                } else {
                    let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                    k_n * delta - f_diss - f_adhesion
                }
            } else if cohesion_energy > 0.0 {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                let f_cohesion =
                    cohesion_energy * std::f64::consts::PI * delta * r_eff;
                k_n * delta - f_diss - f_cohesion
            } else {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                (k_n * delta - f_diss).max(0.0)
            };

            // Force direction: along wall normal (pushes atom away from wall)
            atoms.force[i][0] += f_net * wall.normal_x;
            atoms.force[i][1] += f_net * wall.normal_y;
            atoms.force[i][2] += f_net * wall.normal_z;

            // Twisting friction torque (wall-particle)
            if delta > 0.0 {
                let mu_tw = material_table.twisting_friction_ij[mat_i][wall_mat];
                if mu_tw > 0.0 {
                    let twist = dem.omega[i][0] * wall.normal_x
                        + dem.omega[i][1] * wall.normal_y
                        + dem.omega[i][2] * wall.normal_z;
                    if twist.abs() > 1e-30 {
                        let tau = mu_tw * f_net.abs() * r_eff;
                        let sign_tw = if twist > 0.0 { -1.0 } else { 1.0 };
                        dem.torque[i][0] += sign_tw * tau * wall.normal_x;
                        dem.torque[i][1] += sign_tw * tau * wall.normal_y;
                        dem.torque[i][2] += sign_tw * tau * wall.normal_z;
                    }
                }
            }

            // Accumulate wall force for servo control
            wall_forces[wall_idx] += f_net;
        }
    }

    // Write accumulated forces back to walls
    for (idx, &f) in wall_forces.iter().enumerate() {
        walls.planes[idx].force_accumulator += f;
    }

    // ── Cylinder walls ──────────────────────────────────────────────────
    let ncyl = walls.cylinders.len();
    let mut cyl_forces = vec![0.0f64; ncyl];
    for (cyl_idx, cyl) in walls.cylinders.iter().enumerate() {
        if !walls.cylinder_active[cyl_idx] {
            continue;
        }
        let wall_mat = cyl.material_index;
        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];
            let mat_i = atoms.atom_type[i] as usize;

            // Decompose position into axial and radial components.
            // `axial` = coordinate along the cylinder axis.
            // `d0, d1` = displacement from cylinder center in the 2D cross-section plane.
            // For axis=Z: axial=z, d0=x-cx, d1=y-cy.
            let (axial, d0, d1) = match cyl.axis {
                0 => (pos[0], pos[1] - cyl.center[0], pos[2] - cyl.center[1]),
                1 => (pos[1], pos[0] - cyl.center[0], pos[2] - cyl.center[1]),
                _ => (pos[2], pos[0] - cyl.center[0], pos[1] - cyl.center[1]),
            };

            // Check axial bounds
            if axial < cyl.lo || axial > cyl.hi {
                continue;
            }

            let radial_dist = (d0 * d0 + d1 * d1).sqrt();
            if radial_dist < 1e-30 {
                continue;
            }

            // Compute overlap (delta) and 3D contact normal (nx, ny, nz).
            // The gap is the distance from the particle center to the wall surface.
            // The normal always points from the wall surface toward the particle center.
            let inv_r = 1.0 / radial_dist;
            let (delta, nx, ny, nz) = if cyl.inside {
                // Inside: gap = cylinder_radius - radial_distance
                let gap = cyl.radius - radial_dist;
                let delta = radius - gap;
                // Normal points inward (toward axis), pushing particle away from wall
                let (n0, n1) = (-d0 * inv_r, -d1 * inv_r);
                let (nx, ny, nz) = match cyl.axis {
                    0 => (0.0, n0, n1),
                    1 => (n0, 0.0, n1),
                    _ => (n0, n1, 0.0),
                };
                (delta, nx, ny, nz)
            } else {
                // Outside: gap = radial_distance - cylinder_radius
                let gap = radial_dist - cyl.radius;
                let delta = radius - gap;
                // Normal points outward (away from axis), pushing particle away from wall
                let (n0, n1) = (d0 * inv_r, d1 * inv_r);
                let (nx, ny, nz) = match cyl.axis {
                    0 => (0.0, n0, n1),
                    1 => (n0, 0.0, n1),
                    _ => (n0, n1, 0.0),
                };
                (delta, nx, ny, nz)
            };

            if delta <= 0.0 {
                continue;
            }
            let delta = delta.min(0.5 * radius);

            let r_eff = radius;
            let e_eff = material_table.e_eff_ij[mat_i][wall_mat];
            let sdr = (delta * r_eff).sqrt();
            let k_n = 4.0 / 3.0 * e_eff * sdr;
            let s_n = 2.0 * e_eff * sdr;
            let m_r = atoms.mass[i];
            let v_n = atoms.vel[i][0] * nx + atoms.vel[i][1] * ny + atoms.vel[i][2] * nz;
            let beta = material_table.beta_ij[mat_i][wall_mat];
            let cohesion_energy = material_table.cohesion_energy_ij[mat_i][wall_mat];

            let f_net = if cohesion_energy > 0.0 {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
                k_n * delta - f_diss - f_cohesion
            } else {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                (k_n * delta - f_diss).max(0.0)
            };

            atoms.force[i][0] += f_net * nx;
            atoms.force[i][1] += f_net * ny;
            atoms.force[i][2] += f_net * nz;
            cyl_forces[cyl_idx] += f_net;
        }
    }
    for (idx, &f) in cyl_forces.iter().enumerate() {
        walls.cylinders[idx].force_accumulator += f;
    }

    // ── Sphere walls ────────────────────────────────────────────────────
    let nsph = walls.spheres.len();
    let mut sph_forces = vec![0.0f64; nsph];
    for (sph_idx, sph) in walls.spheres.iter().enumerate() {
        if !walls.sphere_active[sph_idx] {
            continue;
        }
        let wall_mat = sph.material_index;
        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];
            let mat_i = atoms.atom_type[i] as usize;

            let dx = pos[0] - sph.center[0];
            let dy = pos[1] - sph.center[1];
            let dz = pos[2] - sph.center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1e-30 {
                continue;
            }

            let inv_dist = 1.0 / dist;
            let (nx, ny, nz, delta) = if sph.inside {
                let gap = sph.radius - dist;
                let delta = radius - gap;
                // Normal points inward (toward center)
                (-dx * inv_dist, -dy * inv_dist, -dz * inv_dist, delta)
            } else {
                let gap = dist - sph.radius;
                let delta = radius - gap;
                // Normal points outward (away from center)
                (dx * inv_dist, dy * inv_dist, dz * inv_dist, delta)
            };

            if delta <= 0.0 {
                continue;
            }
            let delta = delta.min(0.5 * radius);

            let r_eff = radius;
            let e_eff = material_table.e_eff_ij[mat_i][wall_mat];
            let sdr = (delta * r_eff).sqrt();
            let k_n = 4.0 / 3.0 * e_eff * sdr;
            let s_n = 2.0 * e_eff * sdr;
            let m_r = atoms.mass[i];
            let v_n = atoms.vel[i][0] * nx + atoms.vel[i][1] * ny + atoms.vel[i][2] * nz;
            let beta = material_table.beta_ij[mat_i][wall_mat];
            let cohesion_energy = material_table.cohesion_energy_ij[mat_i][wall_mat];

            let f_net = if cohesion_energy > 0.0 {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
                k_n * delta - f_diss - f_cohesion
            } else {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                (k_n * delta - f_diss).max(0.0)
            };

            atoms.force[i][0] += f_net * nx;
            atoms.force[i][1] += f_net * ny;
            atoms.force[i][2] += f_net * nz;
            sph_forces[sph_idx] += f_net;
        }
    }
    for (idx, &f) in sph_forces.iter().enumerate() {
        walls.spheres[idx].force_accumulator += f;
    }

    // ── Region walls ────────────────────────────────────────────────────
    let nreg = walls.regions.len();
    let mut reg_forces = vec![0.0f64; nreg];
    for (reg_idx, reg) in walls.regions.iter().enumerate() {
        if !walls.region_active[reg_idx] {
            continue;
        }
        let wall_mat = reg.material_index;
        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];
            let mat_i = atoms.atom_type[i] as usize;

            let sr = reg.region.closest_point_on_surface(&pos);

            // Compute gap: distance from particle surface to region surface.
            // If inside=true, particles live inside the region, so the wall
            // surface is the region boundary and the gap shrinks as the particle
            // approaches the boundary from inside.
            // sr.distance is positive outside, negative inside.
            let gap = if reg.inside {
                // Particle inside: gap = |signed_dist| when signed_dist < 0 (inside)
                // and gap = -signed_dist (positive when inside, wall distance shrinks to 0)
                -sr.distance
            } else {
                // Particle outside: gap = signed_dist (positive when outside)
                sr.distance
            };

            let delta = radius - gap;
            if delta <= 0.0 {
                continue;
            }
            let delta = delta.min(0.5 * radius);

            // Normal direction: points from wall surface toward particle center
            let (nx, ny, nz) = if reg.inside {
                // Inside: normal points inward (toward particle, away from surface)
                (-sr.normal[0], -sr.normal[1], -sr.normal[2])
            } else {
                // Outside: normal is already outward (toward particle)
                (sr.normal[0], sr.normal[1], sr.normal[2])
            };

            let r_eff = radius;
            let e_eff = material_table.e_eff_ij[mat_i][wall_mat];
            let sdr = (delta * r_eff).sqrt();
            let k_n = 4.0 / 3.0 * e_eff * sdr;
            let s_n = 2.0 * e_eff * sdr;
            let m_r = atoms.mass[i];
            let v_n = atoms.vel[i][0] * nx + atoms.vel[i][1] * ny + atoms.vel[i][2] * nz;
            let beta = material_table.beta_ij[mat_i][wall_mat];
            let cohesion_energy = material_table.cohesion_energy_ij[mat_i][wall_mat];

            let f_net = if cohesion_energy > 0.0 {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                let f_cohesion = cohesion_energy * std::f64::consts::PI * delta * r_eff;
                k_n * delta - f_diss - f_cohesion
            } else {
                let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
                (k_n * delta - f_diss).max(0.0)
            };

            atoms.force[i][0] += f_net * nx;
            atoms.force[i][1] += f_net * ny;
            atoms.force[i][2] += f_net * nz;
            reg_forces[reg_idx] += f_net;
        }
    }
    for (idx, &f) in reg_forces.iter().enumerate() {
        walls.regions[idx].force_accumulator += f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use mddem_core::region::Region;
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_test_utils::{make_material_table, push_dem_test_atom};

    fn make_wall_plane(
        point_x: f64,
        point_y: f64,
        point_z: f64,
        normal_x: f64,
        normal_y: f64,
        normal_z: f64,
    ) -> WallPlane {
        let mag = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();
        WallPlane {
            point_x,
            point_y,
            point_z,
            normal_x: normal_x / mag,
            normal_y: normal_y / mag,
            normal_z: normal_z / mag,
            material_index: 0,
            name: None,
            bound_x_low: f64::NEG_INFINITY,
            bound_x_high: f64::INFINITY,
            bound_y_low: f64::NEG_INFINITY,
            bound_y_high: f64::INFINITY,
            bound_z_low: f64::NEG_INFINITY,
            bound_z_high: f64::INFINITY,
            velocity: [0.0; 3],
            motion: WallMotion::Static,
            origin: [point_x, point_y, point_z],
            force_accumulator: 0.0,
            temperature: None,
        }
    }

    fn make_walls(planes: Vec<WallPlane>) -> Walls {
        let n = planes.len();
        Walls {
            planes,
            active: vec![true; n],
            cylinders: Vec::new(),
            cylinder_active: Vec::new(),
            spheres: Vec::new(),
            sphere_active: Vec::new(),
            regions: Vec::new(),
            region_active: Vec::new(),
            time: 0.0,
        }
    }

    #[test]
    fn wall_repulsive_for_overlap() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, 0.0005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][2] > 0.0,
            "atom should be pushed away from wall, got {}",
            atom.force[0][2]
        );
        assert!((atom.force[0][0]).abs() < 1e-15);
        assert!((atom.force[0][1]).abs() < 1e-15);
    }

    #[test]
    fn wall_zero_for_no_overlap() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, 0.002], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force[0][2]).abs() < 1e-15);
    }

    #[test]
    fn inactive_wall_applies_no_force() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, 0.0005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let mut plane = make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        plane.name = Some("blocker".into());
        let mut walls = make_walls(vec![plane]);
        walls.active[0] = false;

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force[0][2]).abs() < 1e-15,
            "inactive wall should apply no force"
        );
    }

    #[test]
    fn angled_wall_force_direction() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0003, 0.0, 0.0003], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0] > 0.0, "force_x should be positive");
        assert!(atom.force[0][2] > 0.0, "force_z should be positive");
        assert!(
            (atom.force[0][0] - atom.force[0][2]).abs() < 1e-10,
            "force_x and force_z should be equal for 45-degree wall"
        );
        assert!((atom.force[0][1]).abs() < 1e-15);
    }

    #[test]
    fn bounded_wall_ignores_out_of_bounds_atom() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.05, 0.01, 0.0005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let mut wall = make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        wall.bound_x_low = 0.0;
        wall.bound_x_high = 0.04;

        let walls = make_walls(vec![wall]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force[0][2]).abs() < 1e-15,
            "out-of-bounds atom should get no wall force"
        );
    }

    #[test]
    fn wall_cohesion_attractive_for_small_overlap() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, 0.000999], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

        let mut mt = dem_atom::MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 1e9);
        mt.build_pair_tables();

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(mt);
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][2] < 0.0,
            "wall cohesion should produce attractive force, got {}",
            atom.force[0][2]
        );
    }

    // ── Moving wall tests ───────────────────────────────────────────────────

    #[test]
    fn constant_velocity_wall_moves() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.nlocal = 0;
        atom.natoms = 0;

        let mut plane = make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        plane.velocity = [0.0, 0.0, -0.01];
        plane.motion = WallMotion::ConstantVelocity;

        let mut walls = make_walls(vec![plane]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(walls);
        app.add_update_system(wall_move, ScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let walls = app.get_resource_ref::<Walls>().unwrap();
        assert!(
            (walls.planes[0].point_z - (-0.00001)).abs() < 1e-15,
            "wall should move, got {}",
            walls.planes[0].point_z
        );
        assert!((walls.time - 0.001).abs() < 1e-15);
    }

    #[test]
    fn oscillating_wall_follows_sine() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.nlocal = 0;
        atom.natoms = 0;

        let amplitude = 0.002;
        let frequency = 50.0;
        let mut plane = make_wall_plane(0.0, 0.0, 0.1, 0.0, 0.0, 1.0);
        plane.motion = WallMotion::Oscillate { amplitude, frequency };

        let mut walls = make_walls(vec![plane]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(walls);
        app.add_update_system(wall_move, ScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let walls = app.get_resource_ref::<Walls>().unwrap();
        let expected_phase = 2.0 * std::f64::consts::PI * frequency * 0.001;
        let expected_disp = amplitude * expected_phase.sin();
        assert!(
            (walls.planes[0].point_z - (0.1 + expected_disp)).abs() < 1e-12,
            "oscillating wall z={}, expected {}",
            walls.planes[0].point_z,
            0.1 + expected_disp
        );
    }

    #[test]
    fn servo_wall_adjusts_velocity() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.nlocal = 0;
        atom.natoms = 0;

        let mut plane = make_wall_plane(0.0, 0.0, 0.1, 0.0, 0.0, -1.0);
        plane.motion = WallMotion::Servo {
            target_force: 100.0,
            max_velocity: 0.1,
            gain: 0.001,
        };
        // Simulate accumulated force = 50 (below target)
        plane.force_accumulator = 50.0;

        let mut walls = make_walls(vec![plane]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(walls);
        app.add_update_system(wall_move, ScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let walls = app.get_resource_ref::<Walls>().unwrap();
        // error = 100 - 50 = 50, vel_mag = 0.001 * 50 = 0.05 (within max)
        // velocity along normal (-z): vel = 0.05 * [0, 0, -1] = [0, 0, -0.05]
        assert!(
            (walls.planes[0].velocity[2] - (-0.05)).abs() < 1e-10,
            "servo velocity z={}, expected -0.05",
            walls.planes[0].velocity[2]
        );
        // Position should move
        assert!(
            walls.planes[0].point_z < 0.1,
            "servo wall should have moved"
        );
    }

    #[test]
    fn moving_wall_relative_velocity_affects_damping() {
        // A wall moving toward a stationary atom should produce higher force
        // than a static wall with the same overlap
        let radius = 0.001;

        let run_with_wall_vel = |wall_vel: [f64; 3]| -> f64 {
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, 0.0005], radius);
            atom.nlocal = 1;
            atom.natoms = 1;

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);

            let mut plane = make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            plane.velocity = wall_vel;
            plane.motion = WallMotion::ConstantVelocity;
            let walls = make_walls(vec![plane]);

            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let atom = app.get_resource_ref::<Atom>().unwrap();
            atom.force[0][2]
        };

        let f_static = run_with_wall_vel([0.0, 0.0, 0.0]);
        let f_approaching = run_with_wall_vel([0.0, 0.0, 1.0]); // wall moving toward atom

        // Wall approaching means relative velocity is negative (approaching)
        // which increases damping force, so total repulsion should be higher
        assert!(
            f_approaching > f_static,
            "approaching wall should increase repulsive force: static={}, approaching={}",
            f_static,
            f_approaching
        );
    }

    // ── Cylinder wall tests ────────────────────────────────────────────────

    fn make_walls_with_cylinder(cyl: WallCylinder) -> Walls {
        Walls {
            planes: Vec::new(),
            active: Vec::new(),
            cylinders: vec![cyl],
            cylinder_active: vec![true],
            spheres: Vec::new(),
            sphere_active: Vec::new(),
            regions: Vec::new(),
            region_active: Vec::new(),
            time: 0.0,
        }
    }

    fn make_walls_with_sphere(sph: WallSphere) -> Walls {
        Walls {
            planes: Vec::new(),
            active: Vec::new(),
            cylinders: Vec::new(),
            cylinder_active: Vec::new(),
            spheres: vec![sph],
            sphere_active: vec![true],
            regions: Vec::new(),
            region_active: Vec::new(),
            time: 0.0,
        }
    }

    fn make_walls_with_region(reg: WallRegion) -> Walls {
        Walls {
            planes: Vec::new(),
            active: Vec::new(),
            cylinders: Vec::new(),
            cylinder_active: Vec::new(),
            spheres: Vec::new(),
            sphere_active: Vec::new(),
            regions: vec![reg],
            region_active: vec![true],
            time: 0.0,
        }
    }

    #[test]
    fn cylinder_inside_repels_toward_center() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle near the wall of a z-cylinder centered at (0.005, 0.005), radius 0.004
        // Place particle at radial distance 0.0035 from axis (gap = 0.004 - 0.0035 = 0.0005 < radius)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005 + 0.0035, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_cylinder(WallCylinder {
            axis: 2, // Z
            center: [0.005, 0.005],
            radius: 0.004,
            lo: 0.0,
            hi: 0.01,
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Force should push particle toward center (negative x direction)
        assert!(
            atom.force[0][0] < 0.0,
            "cylinder should push particle toward center, got fx={}",
            atom.force[0][0]
        );
        assert!((atom.force[0][1]).abs() < 1e-15, "no y force");
        assert!((atom.force[0][2]).abs() < 1e-15, "no z force");
    }

    #[test]
    fn cylinder_no_force_when_not_touching() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle well inside cylinder (far from wall)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_cylinder(WallCylinder {
            axis: 2,
            center: [0.005, 0.005],
            radius: 0.004,
            lo: 0.0,
            hi: 0.01,
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag = (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        assert!(f_mag < 1e-15, "no force when not touching cylinder wall, got {}", f_mag);
    }

    #[test]
    fn sphere_inside_repels_toward_center() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle near the wall of a sphere centered at (0.005, 0.005, 0.005), radius 0.004
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005 + 0.0035, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_sphere(WallSphere {
            center: [0.005, 0.005, 0.005],
            radius: 0.004,
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Force should push particle toward center (negative x direction)
        assert!(
            atom.force[0][0] < 0.0,
            "sphere should push particle toward center, got fx={}",
            atom.force[0][0]
        );
        assert!((atom.force[0][1]).abs() < 1e-15, "no y force");
        assert!((atom.force[0][2]).abs() < 1e-15, "no z force");
    }

    #[test]
    fn sphere_no_force_when_not_touching() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle at center of sphere
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_sphere(WallSphere {
            center: [0.005, 0.005, 0.005],
            radius: 0.004,
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag = (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        assert!(f_mag < 1e-15, "no force when not touching sphere wall, got {}", f_mag);
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Cylinder wall force direction always points radially
    // For an inside cylinder, the force should always point toward the axis
    // regardless of where the particle is around the circumference.
    // Test at multiple angular positions around a Z-cylinder.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn cylinder_force_always_points_radially_inward() {
        let particle_radius = 0.001;
        let cyl_radius = 0.01;
        let center = [0.005, 0.005];
        // Place particles near the wall at different angles
        let angles: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0];

        for angle in &angles {
            let r = cyl_radius - 0.5 * particle_radius; // overlap = 0.5 * particle_radius
            let px = center[0] + r * angle.cos();
            let py = center[1] + r * angle.sin();
            let pz = 0.005;

            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, [px, py, pz], particle_radius);
            atom.nlocal = 1;
            atom.natoms = 1;

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);

            let walls = make_walls_with_cylinder(WallCylinder {
                axis: 2,
                center,
                radius: cyl_radius,
                lo: 0.0,
                hi: 0.01,
                inside: true,
                material_index: 0,
                name: None,
                force_accumulator: 0.0,
                temperature: None,
            });

            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let atom = app.get_resource_ref::<Atom>().unwrap();
            let fx = atom.force[0][0];
            let fy = atom.force[0][1];
            let fz = atom.force[0][2];

            // Force should be purely radial (no z component)
            assert!(
                fz.abs() < 1e-12,
                "angle={:.1}: no z force expected, got {:.6e}", angle, fz
            );

            // Force direction should point toward axis center
            let dx = px - center[0];
            let dy = py - center[1];
            let r_actual = (dx * dx + dy * dy).sqrt();
            // Radial unit vector (outward): (dx/r, dy/r)
            // Force should oppose this (inward): dot(f, r_hat) < 0
            let f_dot_r = fx * dx / r_actual + fy * dy / r_actual;
            assert!(
                f_dot_r < 0.0,
                "angle={:.1}: force should point inward, f·r_hat={:.6e}",
                angle, f_dot_r
            );

            // Force magnitude should be nonzero
            let f_mag = (fx * fx + fy * fy).sqrt();
            assert!(f_mag > 0.0, "angle={:.1}: force magnitude should be nonzero", angle);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Cylinder axial bounds are enforced
    // Particles outside lo/hi should not get any force from the cylinder.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn cylinder_axial_bounds_enforced() {
        let particle_radius = 0.001;
        let cyl_radius = 0.01;
        let center = [0.005, 0.005];

        // Place particle near the wall but outside axial bounds (below lo)
        let r = cyl_radius - 0.5 * particle_radius;
        let px = center[0] + r;
        let py = center[1];
        let pz = -0.001; // below lo=0.0

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        push_dem_test_atom(&mut atom, &mut dem, 0, [px, py, pz], particle_radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_cylinder(WallCylinder {
            axis: 2,
            center,
            radius: cyl_radius,
            lo: 0.0,
            hi: 0.01,
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag = (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        assert!(
            f_mag < 1e-15,
            "No force outside axial bounds: f_mag={:.6e}",
            f_mag
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Sphere wall force direction at multiple positions
    // For an inside sphere, force should always point toward the center.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn sphere_force_points_toward_center_at_multiple_positions() {
        let particle_radius = 0.001;
        let sph_radius = 0.01;
        let sph_center = [0.005, 0.005, 0.005];

        // Test positions along different axes
        let directions: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        for dir in &directions {
            let mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let nd = [dir[0] / mag, dir[1] / mag, dir[2] / mag];
            let r = sph_radius - 0.5 * particle_radius;
            let px = sph_center[0] + r * nd[0];
            let py = sph_center[1] + r * nd[1];
            let pz = sph_center[2] + r * nd[2];

            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, [px, py, pz], particle_radius);
            atom.nlocal = 1;
            atom.natoms = 1;

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);

            let walls = make_walls_with_sphere(WallSphere {
                center: sph_center,
                radius: sph_radius,
                inside: true,
                material_index: 0,
                name: None,
                force_accumulator: 0.0,
                temperature: None,
            });

            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let atom = app.get_resource_ref::<Atom>().unwrap();
            let fx = atom.force[0][0];
            let fy = atom.force[0][1];
            let fz = atom.force[0][2];
            let f_mag = (fx * fx + fy * fy + fz * fz).sqrt();

            assert!(f_mag > 0.0, "dir={:?}: force should be nonzero", dir);

            // Force should point toward center: dot(f, r_hat) < 0
            let dx = px - sph_center[0];
            let dy = py - sph_center[1];
            let dz = pz - sph_center[2];
            let r_actual = (dx * dx + dy * dy + dz * dz).sqrt();
            let f_dot_r = fx * dx / r_actual + fy * dy / r_actual + fz * dz / r_actual;
            assert!(
                f_dot_r < 0.0,
                "dir={:?}: force should point toward center, f·r_hat={:.6e}",
                dir, f_dot_r
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Wall force at exact contact (delta=0) should be zero
    // When a particle's surface just touches the wall (no overlap),
    // the elastic force should be zero.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn wall_zero_force_at_exact_contact() {
        let radius = 0.001;

        // Place particle center at exactly radius from wall -> delta = 0
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, radius], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag = (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        assert!(
            f_mag < 1e-10,
            "Force at exact contact should be ~zero, got {:.6e}",
            f_mag
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Wall Hertz force scales as delta^(3/2) for plane walls
    // Same as particle-particle Hertz, but with R_eff = R_particle.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn wall_hertz_force_scales_as_delta_three_halves() {
        let radius = 0.001;

        let wall_force_at = |delta: f64| -> f64 {
            let distance = radius - delta; // signed distance from wall to center
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, [0.01, 0.01, distance], radius);
            atom.nlocal = 1;
            atom.natoms = 1;

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);

            let walls = make_walls(vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let atom = app.get_resource_ref::<Atom>().unwrap();
            atom.force[0][2].abs()
        };

        let deltas = [1e-5, 2e-5, 4e-5, 6e-5, 8e-5];
        let forces: Vec<f64> = deltas.iter().map(|d| wall_force_at(*d)).collect();

        for i in 1..deltas.len() {
            let expected_ratio = (deltas[i] / deltas[0]).powf(1.5);
            let actual_ratio = forces[i] / forces[0];
            let rel_err = ((actual_ratio - expected_ratio) / expected_ratio).abs();
            assert!(
                rel_err < 0.01,
                "Wall Hertz scaling: delta ratio {:.1}, expected F ratio {:.4}, got {:.4} (rel err {:.4})",
                deltas[i] / deltas[0], expected_ratio, actual_ratio, rel_err
            );
        }
    }

    // ── Region wall tests ─────────────────────────────────────────────────

    #[test]
    fn region_sphere_inside_repels() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle near sphere wall surface (inside sphere of radius 0.004)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005 + 0.0035, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_region(WallRegion {
            region: Region::Sphere {
                center: [0.005, 0.005, 0.005],
                radius: 0.004,
            },
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][0] < 0.0,
            "region sphere wall should push particle toward center, got fx={}",
            atom.force[0][0]
        );
        assert!((atom.force[0][1]).abs() < 1e-15, "no y force");
        assert!((atom.force[0][2]).abs() < 1e-15, "no z force");
    }

    #[test]
    fn region_sphere_no_force_when_far() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle at center of sphere (far from wall)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_region(WallRegion {
            region: Region::Sphere {
                center: [0.005, 0.005, 0.005],
                radius: 0.004,
            },
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let f_mag = (atom.force[0][0].powi(2) + atom.force[0][1].powi(2) + atom.force[0][2].powi(2)).sqrt();
        assert!(f_mag < 1e-15, "no force when far from region wall, got {}", f_mag);
    }

    #[test]
    fn region_block_inside_repels() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Particle near the +z face of a block (inside, close to top)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005, 0.005, 0.0095], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_region(WallRegion {
            region: Region::Block {
                min: [0.0, 0.0, 0.0],
                max: [0.01, 0.01, 0.01],
            },
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Should be pushed away from the +z face (downward)
        assert!(
            atom.force[0][2] < 0.0,
            "region block wall should push particle away from +z face, got fz={}",
            atom.force[0][2]
        );
    }

    #[test]
    fn region_cone_inside_repels() {
        use mddem_core::region::Axis;
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.0005;

        // Cone: z-axis, rad_lo=0.004 at z=0, rad_hi=0.002 at z=0.01
        // At z=0.005, radius = 0.003
        // Place particle at radial distance 0.0028 from axis (gap = 0.003 - 0.0028 = 0.0002 < radius)
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.005 + 0.0028, 0.005, 0.005], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = make_walls_with_region(WallRegion {
            region: Region::Cone {
                center: [0.005, 0.005],
                axis: Axis::Z,
                rad_lo: 0.004,
                rad_hi: 0.002,
                lo: 0.0,
                hi: 0.01,
            },
            inside: true,
            material_index: 0,
            name: None,
            force_accumulator: 0.0,
            temperature: None,
        });

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Force should push toward center (negative x direction)
        assert!(
            atom.force[0][0] < 0.0,
            "cone wall should push particle toward center, got fx={}",
            atom.force[0][0]
        );
    }

    #[test]
    fn region_wall_force_matches_dedicated_sphere() {
        // Verify that a region sphere wall produces the same force as the dedicated sphere wall
        let radius = 0.001;
        let sphere_center = [0.005, 0.005, 0.005];
        let sphere_radius = 0.004;
        let particle_pos = [0.005 + 0.0035, 0.005, 0.005];

        // Run with dedicated sphere wall
        let f_dedicated = {
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, particle_pos, radius);
            atom.nlocal = 1;
            atom.natoms = 1;
            let mut registry = AtomDataRegistry::new();
            registry.register(dem);
            let walls = make_walls_with_sphere(WallSphere {
                center: sphere_center,
                radius: sphere_radius,
                inside: true,
                material_index: 0,
                name: None,
                force_accumulator: 0.0,
                temperature: None,
            });
            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            [atom.force[0][0], atom.force[0][1], atom.force[0][2]]
        };

        // Run with region sphere wall
        let f_region = {
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            push_dem_test_atom(&mut atom, &mut dem, 0, particle_pos, radius);
            atom.nlocal = 1;
            atom.natoms = 1;
            let mut registry = AtomDataRegistry::new();
            registry.register(dem);
            let walls = make_walls_with_region(WallRegion {
                region: Region::Sphere {
                    center: sphere_center,
                    radius: sphere_radius,
                },
                inside: true,
                material_index: 0,
                name: None,
                force_accumulator: 0.0,
                temperature: None,
            });
            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(registry);
            app.add_resource(make_material_table());
            app.add_resource(walls);
            app.add_update_system(wall_contact_force, ScheduleSet::Force);
            app.organize_systems();
            app.run();
            let atom = app.get_resource_ref::<Atom>().unwrap();
            [atom.force[0][0], atom.force[0][1], atom.force[0][2]]
        };

        for d in 0..3 {
            assert!(
                (f_dedicated[d] - f_region[d]).abs() < 1e-6 * f_dedicated[d].abs().max(1e-15),
                "force mismatch in dim {}: dedicated={}, region={}",
                d,
                f_dedicated[d],
                f_region[d]
            );
        }
    }

    #[test]
    fn static_wall_unaffected_by_motion_systems() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.nlocal = 0;
        atom.natoms = 0;

        let plane = make_wall_plane(0.0, 0.0, 0.5, 0.0, 0.0, 1.0);
        let mut walls = make_walls(vec![plane]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(walls);
        app.add_update_system(wall_move, ScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let walls = app.get_resource_ref::<Walls>().unwrap();
        assert!(
            (walls.planes[0].point_z - 0.5).abs() < 1e-15,
            "static wall should not move"
        );
    }
}
