//! Plane wall contact forces for DEM particles (Hertz normal + Coulomb friction).
//! Supports static, constant-velocity, oscillating, and servo-controlled walls.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::{DemAtom, MaterialTable, SQRT_5_3};
use mddem_core::{Atom, AtomDataRegistry, Config};

fn default_neg_inf() -> f64 {
    f64::NEG_INFINITY
}
fn default_pos_inf() -> f64 {
    f64::INFINITY
}

// ── Config structs ──────────────────────────────────────────────────────────

/// Oscillation parameters for a wall.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OscillateDef {
    pub amplitude: f64,
    pub frequency: f64,
}

/// Servo control parameters for a wall.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ServoDef {
    pub target_force: f64,
    pub max_velocity: f64,
    pub gain: f64,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML definition of a wall plane: point, normal, material, and optional bounding box.
pub struct WallDef {
    pub point_x: f64,
    pub point_y: f64,
    pub point_z: f64,
    pub normal_x: f64,
    pub normal_y: f64,
    pub normal_z: f64,
    pub material: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_neg_inf")]
    pub bound_x_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_x_high: f64,
    #[serde(default = "default_neg_inf")]
    pub bound_y_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_y_high: f64,
    #[serde(default = "default_neg_inf")]
    pub bound_z_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_z_high: f64,
    /// Constant wall velocity [vx, vy, vz].
    #[serde(default)]
    pub velocity: Option<[f64; 3]>,
    /// Sinusoidal oscillation along the wall normal.
    #[serde(default)]
    pub oscillate: Option<OscillateDef>,
    /// Servo control: adjust velocity to reach target force.
    #[serde(default)]
    pub servo: Option<ServoDef>,
}

// ── Runtime types ───────────────────────────────────────────────────────────

/// Wall motion type at runtime.
pub enum WallMotion {
    Static,
    ConstantVelocity,
    Oscillate { amplitude: f64, frequency: f64 },
    Servo { target_force: f64, max_velocity: f64, gain: f64 },
}

/// Runtime representation of a wall plane with resolved material index.
pub struct WallPlane {
    pub point_x: f64,
    pub point_y: f64,
    pub point_z: f64,
    pub normal_x: f64,
    pub normal_y: f64,
    pub normal_z: f64,
    pub material_index: usize,
    pub name: Option<String>,
    pub bound_x_low: f64,
    pub bound_x_high: f64,
    pub bound_y_low: f64,
    pub bound_y_high: f64,
    pub bound_z_low: f64,
    pub bound_z_high: f64,
    /// Current wall velocity.
    pub velocity: [f64; 3],
    /// Motion mode.
    pub motion: WallMotion,
    /// Origin point (initial position, for oscillation reference).
    pub origin: [f64; 3],
    /// Accumulated contact force this step (scalar, along normal).
    pub force_accumulator: f64,
}

impl WallPlane {
    /// Check if atom position is within the wall's bounding region.
    #[inline]
    fn in_bounds(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.bound_x_low
            && x <= self.bound_x_high
            && y >= self.bound_y_low
            && y <= self.bound_y_high
            && z >= self.bound_z_low
            && z <= self.bound_z_high
    }
}

/// Collection of wall planes with per-wall active/inactive flags.
pub struct Walls {
    pub planes: Vec<WallPlane>,
    pub active: Vec<bool>,
    /// Elapsed simulation time (for oscillation phase tracking).
    pub time: f64,
}

impl Walls {
    pub fn deactivate_by_name(&mut self, name: &str) {
        for (i, wall) in self.planes.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.active[i] = false;
            }
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers wall contact force system from `[[wall]]` TOML config.
pub struct WallPlugin;

impl Plugin for WallPlugin {
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

            let mut planes = Vec::with_capacity(wall_defs.len());
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

                // Determine motion mode and initial velocity
                let (motion, velocity) = if let Some(ref osc) = w.oscillate {
                    (
                        WallMotion::Oscillate {
                            amplitude: osc.amplitude,
                            frequency: osc.frequency,
                        },
                        [0.0; 3], // velocity set each step by wall_move
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
                });
            }
            drop(material_table);

            let n = planes.len();
            Walls {
                planes,
                active: vec![true; n],
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

/// Move walls according to their motion mode.
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

/// Zero wall force accumulators before force computation.
pub fn wall_zero_force_accumulators(mut walls: ResMut<Walls>) {
    for wall in &mut walls.planes {
        wall.force_accumulator = 0.0;
    }
}

pub fn wall_contact_force(
    mut atoms: ResMut<Atom>,
    mut walls: ResMut<Walls>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let dem = registry.expect::<DemAtom>("wall_contact_force");

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

            // JKR pull-off distance for extended interaction range
            let delta_pulloff = if surface_energy > 0.0 {
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

            let jkr_adhesion_only = surface_energy > 0.0 && delta <= 0.0;

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

            let f_net = if surface_energy > 0.0 {
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

            // Accumulate wall force for servo control
            wall_forces[wall_idx] += f_net;
        }
    }

    // Write accumulated forces back to walls
    for (idx, &f) in wall_forces.iter().enumerate() {
        walls.planes[idx].force_accumulator += f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
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
        }
    }

    fn make_walls(planes: Vec<WallPlane>) -> Walls {
        let n = planes.len();
        Walls {
            planes,
            active: vec![true; n],
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
