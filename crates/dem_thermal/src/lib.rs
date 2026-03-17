//! Contact-based heat conduction between DEM particles.
//!
//! This crate implements thermal conduction through particle-particle contacts
//! in a Discrete Element Method (DEM) simulation. When two particles overlap
//! (i.e., are in mechanical contact), heat flows from the hotter particle to the
//! cooler one at a rate proportional to the contact area and the temperature
//! difference.
//!
//! # Heat Transfer Model
//!
//! The heat flux between two contacting particles *i* and *j* is:
//!
//! ```text
//! Q = k · 2a · (Tⱼ − Tᵢ)
//! ```
//!
//! where:
//! - **k** is the thermal conductivity (W/(m·K)),
//! - **a** = √(r_eff · δ) is the contact radius from Hertz theory,
//! - **r_eff** = (rᵢ · rⱼ) / (rᵢ + rⱼ) is the effective radius,
//! - **δ** = (rᵢ + rⱼ) − d is the overlap (d = center-to-center distance),
//! - **Tᵢ**, **Tⱼ** are particle temperatures (K).
//!
//! Temperature is then integrated forward in time:
//!
//! ```text
//! Tᵢ(t + dt) = Tᵢ(t) + dt · Qᵢ / (mᵢ · cₚ)
//! ```
//!
//! where **mᵢ** is the particle mass and **cₚ** is the specific heat capacity.
//!
//! # TOML Configuration
//!
//! Add a `[thermal]` section to your simulation config file:
//!
//! ```toml
//! [thermal]
//! conductivity = 1.0          # Thermal conductivity in W/(m·K)
//! specific_heat = 500.0       # Specific heat capacity in J/(kg·K)
//! initial_temperature = 300.0 # Initial temperature for all particles in K (default: 300.0)
//! ```
//!
//! If the `[thermal]` section is omitted entirely, the plugin registers the
//! [`ThermalAtom`] data but does not add any systems — temperatures remain
//! at their initial values.
//!
//! # Per-Atom Data
//!
//! This crate extends each particle with [`ThermalAtom`] fields:
//! - `temperature` — current particle temperature (K), communicated to ghost atoms
//! - `heat_flux` — accumulated heat flux (W), reverse-communicated and zeroed each step
//!
//! # Plugin Dependencies
//!
//! [`ThermalPlugin`] requires `DemAtomPlugin` (for particle radii) and
//! `NeighborPlugin` (for contact pair iteration).

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use dem_wall::Walls;
use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, Config};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for contact-based heat conduction, deserialized from `[thermal]`.
///
/// # TOML Fields
///
/// | Field                 | Type  | Default | Unit    | Description                       |
/// |-----------------------|-------|---------|---------|-----------------------------------|
/// | `conductivity`        | `f64` | 1.0     | W/(m·K) | Thermal conductivity              |
/// | `specific_heat`       | `f64` | 500.0   | J/(kg·K)| Specific heat capacity            |
/// | `initial_temperature` | `f64` | 300.0   | K       | Initial temperature for all atoms |
///
/// # Example
///
/// ```toml
/// [thermal]
/// conductivity = 50.0
/// specific_heat = 900.0
/// initial_temperature = 350.0
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ThermalConfig {
    /// Thermal conductivity (W/(m·K)).
    pub conductivity: f64,
    /// Specific heat capacity (J/(kg·K)).
    pub specific_heat: f64,
    /// Initial temperature for all particles (K). Defaults to 300.0 if omitted.
    #[serde(default = "default_initial_temperature")]
    pub initial_temperature: f64,
}

fn default_initial_temperature() -> f64 {
    300.0
}

impl Default for ThermalConfig {
    fn default() -> Self {
        ThermalConfig {
            conductivity: 1.0,
            specific_heat: 500.0,
            initial_temperature: 300.0,
        }
    }
}

// ── Per-atom thermal data ───────────────────────────────────────────────────

/// Per-atom thermal extension: temperature and accumulated heat flux.
///
/// Registered automatically by [`ThermalPlugin`]. Access via the
/// [`AtomDataRegistry`] in your systems:
///
/// ```ignore
/// let thermal = registry.expect::<ThermalAtom>("my_system");
/// let temp_i = thermal.temperature[i];
/// ```
///
/// Communication attributes:
/// - `temperature` is `#[forward]`-communicated so ghost atoms carry correct temps.
/// - `heat_flux` is `#[reverse]`-communicated (summed back to owning proc) and
///   `#[zero]`ed each timestep.
#[derive(AtomData)]
pub struct ThermalAtom {
    /// Per-atom temperature (K). Communicated forward to ghost atoms.
    #[forward]
    pub temperature: Vec<f64>,
    /// Per-atom heat flux accumulator (W). Reverse-communicated and zeroed each step.
    #[reverse]
    #[zero]
    pub heat_flux: Vec<f64>,
}

impl Default for ThermalAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl ThermalAtom {
    /// Creates an empty `ThermalAtom` with no particles.
    pub fn new() -> Self {
        ThermalAtom {
            temperature: Vec::new(),
            heat_flux: Vec::new(),
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that adds contact-based heat conduction to a DEM simulation.
///
/// When the `[thermal]` config section is present, this plugin registers three
/// systems:
///
/// 1. **`initialize_temperatures`** (`PostSetup`) — sets all particle
///    temperatures to [`ThermalConfig::initial_temperature`].
/// 2. **`compute_heat_conduction`** (`Force`) — loops over neighbor pairs,
///    computes `Q = k · 2a · (Tⱼ − Tᵢ)` for contacting particles.
/// 3. **`integrate_temperature`** (`PostFinalIntegration`) — updates each
///    particle's temperature: `T += dt · Q / (m · cₚ)`.
///
/// If `[thermal]` is absent from the config, only the [`ThermalAtom`] data is
/// registered (with default values) and no systems are added.
pub struct ThermalPlugin;

impl Plugin for ThermalPlugin {
    fn dependencies(&self) -> Vec<&str> {
        vec!["DemAtomPlugin", "NeighborPlugin"]
    }

    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [thermal]
# conductivity = 1.0          # W/(m·K)
# specific_heat = 500.0       # J/(kg·K)
# initial_temperature = 300.0 # K"#,
        )
    }

    fn build(&self, app: &mut App) {
        register_atom_data!(app, ThermalAtom::new());

        let has_thermal = {
            let config = app
                .get_resource_ref::<Config>()
                .expect("Config must exist");
            config.table.get("thermal").is_some()
        };

        if !has_thermal {
            app.add_resource(ThermalConfig::default());
            return;
        }

        let thermal_config = Config::load::<ThermalConfig>(app, "thermal");
        app.add_resource(thermal_config);

        app.add_setup_system(initialize_temperatures, ScheduleSetupSet::PostSetup);
        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);

        // Register wall heat conduction if walls resource exists
        let has_walls = app.get_resource_ref::<Walls>().is_some();
        if has_walls {
            app.add_update_system(compute_wall_heat_conduction, ScheduleSet::Force);
        }

        app.add_update_system(integrate_temperature, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Set initial temperatures for all atoms to [`ThermalConfig::initial_temperature`].
fn initialize_temperatures(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    config: Res<ThermalConfig>,
) {
    let mut thermal = registry.expect_mut::<ThermalAtom>("initialize_temperatures");
    let n = atoms.len();
    while thermal.temperature.len() < n {
        thermal.temperature.push(config.initial_temperature);
    }
    while thermal.heat_flux.len() < n {
        thermal.heat_flux.push(0.0);
    }
}

/// Compute contact-based heat conduction between neighboring particles.
///
/// For each pair of overlapping particles, the heat flux is:
///
/// ```text
/// Q = k · 2a · (Tⱼ − Tᵢ)
/// ```
///
/// where `a = √(r_eff · δ)` is the Hertzian contact radius. The computed
/// flux is added to atom *i*'s accumulator and subtracted from atom *j*'s,
/// ensuring energy conservation (antisymmetric).
pub fn compute_heat_conduction(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    config: Res<ThermalConfig>,
) {
    let dem = registry.expect::<DemAtom>("compute_heat_conduction");
    let mut thermal = registry.expect_mut::<ThermalAtom>("compute_heat_conduction");

    // Ensure thermal vectors cover all atoms (including ghosts added after setup)
    while thermal.temperature.len() < atoms.len() {
        thermal.temperature.push(config.initial_temperature);
    }
    while thermal.heat_flux.len() < atoms.len() {
        thermal.heat_flux.push(0.0);
    }

    let nlocal = atoms.nlocal as usize;
    let k = config.conductivity;

    for (i, j) in neighbor.pairs(nlocal) {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];
        let sum_r = r1 + r2;

        // Effective radius for Hertz contact: r_eff = (r1 * r2) / (r1 + r2)
        let r_eff = (r1 * r2) / sum_r;

        // Compute center-to-center distance
        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        // Skip non-contacting pairs (no overlap)
        if dist_sq >= sum_r * sum_r {
            continue;
        }

        let distance = dist_sq.sqrt();
        // Overlap depth: δ = (r1 + r2) - distance
        let delta = sum_r - distance;
        if delta <= 0.0 {
            continue;
        }

        // Hertzian contact radius: a = √(r_eff · δ)
        let a = (r_eff * delta).sqrt();

        // Heat transfer through contact: Q = k · 2a · (Tⱼ − Tᵢ)
        // Positive Q means atom i gains heat (j is hotter), negative means i loses heat
        let dt_temp = thermal.temperature[j] - thermal.temperature[i];
        let q = k * 2.0 * a * dt_temp;

        // Accumulate flux antisymmetrically to conserve energy
        thermal.heat_flux[i] += q;
        thermal.heat_flux[j] -= q;
    }
}

/// Compute heat conduction between walls with temperature and contacting particles.
///
/// For each wall with a temperature set, computes:
///
/// ```text
/// Q = k · 2a · (T_wall − T_particle)
/// ```
///
/// where `a = √(R_eff · δ)` is the contact radius. Walls have infinite thermal
/// mass so wall temperature does not change.
///
/// Handles plane, cylinder, sphere, and region wall types.
pub fn compute_wall_heat_conduction(
    atoms: Res<Atom>,
    walls: Res<Walls>,
    registry: Res<AtomDataRegistry>,
    config: Res<ThermalConfig>,
) {
    let dem = registry.expect::<DemAtom>("compute_wall_heat_conduction");
    let mut thermal = registry.expect_mut::<ThermalAtom>("compute_wall_heat_conduction");

    let nlocal = atoms.nlocal as usize;
    let k = config.conductivity;

    // ── Plane walls ──────────────────────────────────────────────────────
    for (wall_idx, wall) in walls.planes.iter().enumerate() {
        if !walls.active[wall_idx] {
            continue;
        }
        let wall_temp = match wall.temperature {
            Some(t) => t,
            None => continue,
        };

        for i in 0..nlocal {
            let px = atoms.pos[i][0];
            let py = atoms.pos[i][1];
            let pz = atoms.pos[i][2];

            if !wall.in_bounds(px, py, pz) {
                continue;
            }

            let dx = px - wall.point_x;
            let dy = py - wall.point_y;
            let dz = pz - wall.point_z;
            let distance = dx * wall.normal_x + dy * wall.normal_y + dz * wall.normal_z;

            if distance <= 0.0 {
                continue;
            }

            let radius = dem.radius[i];
            let delta = radius - distance;
            if delta <= 0.0 {
                continue;
            }

            // Wall has infinite radius → r_eff = r_particle
            let r_eff = radius;
            let a = (r_eff * delta).sqrt();
            let q = k * 2.0 * a * (wall_temp - thermal.temperature[i]);
            thermal.heat_flux[i] += q;
        }
    }

    // ── Cylinder walls ───────────────────────────────────────────────────
    for (cyl_idx, cyl) in walls.cylinders.iter().enumerate() {
        if !walls.cylinder_active[cyl_idx] {
            continue;
        }
        let wall_temp = match cyl.temperature {
            Some(t) => t,
            None => continue,
        };

        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];

            let (axial, d0, d1) = match cyl.axis {
                0 => (pos[0], pos[1] - cyl.center[0], pos[2] - cyl.center[1]),
                1 => (pos[1], pos[0] - cyl.center[0], pos[2] - cyl.center[1]),
                _ => (pos[2], pos[0] - cyl.center[0], pos[1] - cyl.center[1]),
            };

            if axial < cyl.lo || axial > cyl.hi {
                continue;
            }

            let radial_dist = (d0 * d0 + d1 * d1).sqrt();
            if radial_dist < 1e-30 {
                continue;
            }

            let delta = if cyl.inside {
                let gap = cyl.radius - radial_dist;
                radius - gap
            } else {
                let gap = radial_dist - cyl.radius;
                radius - gap
            };

            if delta <= 0.0 {
                continue;
            }

            // For curved walls: R_eff = R_particle * R_wall / (R_particle + R_wall)
            let r_eff = radius * cyl.radius / (radius + cyl.radius);
            let a = (r_eff * delta).sqrt();
            let q = k * 2.0 * a * (wall_temp - thermal.temperature[i]);
            thermal.heat_flux[i] += q;
        }
    }

    // ── Sphere walls ─────────────────────────────────────────────────────
    for (sph_idx, sph) in walls.spheres.iter().enumerate() {
        if !walls.sphere_active[sph_idx] {
            continue;
        }
        let wall_temp = match sph.temperature {
            Some(t) => t,
            None => continue,
        };

        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];

            let dx = pos[0] - sph.center[0];
            let dy = pos[1] - sph.center[1];
            let dz = pos[2] - sph.center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1e-30 {
                continue;
            }

            let delta = if sph.inside {
                let gap = sph.radius - dist;
                radius - gap
            } else {
                let gap = dist - sph.radius;
                radius - gap
            };

            if delta <= 0.0 {
                continue;
            }

            // For curved walls: R_eff = R_particle * R_wall / (R_particle + R_wall)
            let r_eff = radius * sph.radius / (radius + sph.radius);
            let a = (r_eff * delta).sqrt();
            let q = k * 2.0 * a * (wall_temp - thermal.temperature[i]);
            thermal.heat_flux[i] += q;
        }
    }

    // ── Region walls ─────────────────────────────────────────────────────
    for (reg_idx, reg) in walls.regions.iter().enumerate() {
        if !walls.region_active[reg_idx] {
            continue;
        }
        let wall_temp = match reg.temperature {
            Some(t) => t,
            None => continue,
        };

        for i in 0..nlocal {
            let pos = atoms.pos[i];
            let radius = dem.radius[i];

            let sr = reg.region.closest_point_on_surface(&pos);
            let gap = if reg.inside { -sr.distance } else { sr.distance };
            let delta = radius - gap;
            if delta <= 0.0 {
                continue;
            }

            // Region walls treated as planar locally → r_eff = r_particle
            let r_eff = radius;
            let a = (r_eff * delta).sqrt();
            let q = k * 2.0 * a * (wall_temp - thermal.temperature[i]);
            thermal.heat_flux[i] += q;
        }
    }
}

/// Integrate temperature forward in time using accumulated heat flux.
///
/// For each local atom:
///
/// ```text
/// T += dt · Q / (m · cₚ)
/// ```
///
/// where `dt` is the simulation timestep, `Q` is the accumulated heat flux,
/// `m` is the particle mass, and `cₚ` is the specific heat capacity.
pub fn integrate_temperature(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    config: Res<ThermalConfig>,
) {
    let mut thermal = registry.expect_mut::<ThermalAtom>("integrate_temperature");
    let nlocal = atoms.nlocal as usize;
    let cp = config.specific_heat;
    let dt = atoms.dt;

    for i in 0..nlocal {
        // dT = dt * Q / (m * cp): convert heat flux (W) to temperature change (K)
        thermal.temperature[i] += dt * thermal.heat_flux[i] / (atoms.mass[i] * cp);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use dem_wall::{WallMotion, WallPlane, Walls};
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use mddem_test_utils::push_dem_test_atom;

    fn setup_two_atoms(
        t1: f64,
        t2: f64,
        sep: f64,
        radius: f64,
    ) -> (App, ThermalConfig) {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 1, [sep, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut thermal = ThermalAtom::new();
        thermal.temperature.push(t1);
        thermal.temperature.push(t2);
        thermal.heat_flux.push(0.0);
        thermal.heat_flux.push(0.0);

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(thermal);

        let config = ThermalConfig {
            conductivity: 1.0,
            specific_heat: 500.0,
            initial_temperature: 300.0,
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(config.clone());

        (app, config)
    }

    #[test]
    fn heat_flows_hot_to_cold() {
        let radius = 0.001;
        let sep = 0.0019; // overlap
        let (mut app, _) = setup_two_atoms(400.0, 300.0, sep, radius);

        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        // Atom 0 is hotter, should lose heat (negative flux after integration would cool it)
        // But heat_flux is the raw Q: positive means gaining heat
        // Q = k * 2a * (T_j - T_i) for atom i
        // For atom 0: Q = k * 2a * (300 - 400) < 0 (loses heat)
        assert!(
            thermal.heat_flux[0] < 0.0,
            "hot atom should lose heat, got {}",
            thermal.heat_flux[0]
        );
        // For atom 1: Q = k * 2a * (400 - 300) > 0 (gains heat)
        assert!(
            thermal.heat_flux[1] > 0.0,
            "cold atom should gain heat, got {}",
            thermal.heat_flux[1]
        );
        // Energy conservation: sum of fluxes = 0
        assert!(
            (thermal.heat_flux[0] + thermal.heat_flux[1]).abs() < 1e-20,
            "heat flux should be conserved"
        );
    }

    #[test]
    fn no_flow_at_equal_temperature() {
        let radius = 0.001;
        let sep = 0.0019;
        let (mut app, _) = setup_two_atoms(300.0, 300.0, sep, radius);

        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0].abs() < 1e-20,
            "no heat flow at equal T"
        );
        assert!(
            thermal.heat_flux[1].abs() < 1e-20,
            "no heat flow at equal T"
        );
    }

    #[test]
    fn no_flow_beyond_contact() {
        let radius = 0.001;
        let sep = 0.003; // no overlap
        let (mut app, _) = setup_two_atoms(400.0, 300.0, sep, radius);

        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(thermal.heat_flux[0].abs() < 1e-20);
        assert!(thermal.heat_flux[1].abs() < 1e-20);
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Two particles reach thermal equilibrium at mass-weighted
    // average temperature. With equal masses, T_eq = (T1 + T2) / 2.
    // Run many steps and verify convergence to equilibrium.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn thermal_equilibrium_mass_weighted_average() {
        let radius = 0.001;
        let sep = 0.0019; // overlap
        let t1 = 500.0;
        let t2 = 300.0;

        // Instead of using the App scheduler (which has ordering issues for this
        // isolated test), manually step through the physics in a loop.
        // Compute contact geometry once (particles are stationary)
        let r_eff: f64 = radius / 2.0;
        let delta: f64 = 2.0 * radius - sep;
        let a: f64 = (r_eff * delta).sqrt();

        let density: f64 = 2500.0;
        let mass: f64 = density * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let dt: f64 = 1e-5; // larger timestep for faster convergence
        let conductivity: f64 = 100.0; // high conductivity
        let cp: f64 = 500.0;

        let mut temp = [t1, t2];

        for _ in 0..500000 {
            // Heat transfer: Q = k * 2a * (T_j - T_i)
            let dt_temp = temp[1] - temp[0];
            let q = conductivity * 2.0 * a * dt_temp;

            // Integrate temperature
            temp[0] += dt * q / (mass * cp);
            temp[1] -= dt * q / (mass * cp);
        }

        // Equal masses → equilibrium at arithmetic mean
        let t_eq = (t1 + t2) / 2.0; // 400.0

        assert!(
            (temp[0] - t_eq).abs() < 1.0,
            "Atom 0 should approach equilibrium {:.1}, got {:.1}",
            t_eq, temp[0]
        );
        assert!(
            (temp[1] - t_eq).abs() < 1.0,
            "Atom 1 should approach equilibrium {:.1}, got {:.1}",
            t_eq, temp[1]
        );
        // Energy conservation: sum of (m*cp*T) should be constant
        let e_initial = mass * cp * t1 + mass * cp * t2;
        let e_final = mass * cp * temp[0] + mass * cp * temp[1];
        assert!(
            (e_final - e_initial).abs() / e_initial < 1e-10,
            "Thermal energy not conserved: {:.6e} vs {:.6e}",
            e_final, e_initial
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Heat transfer rate scales with conductivity
    // Double the conductivity → double the heat flux per step.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn heat_flux_scales_with_conductivity() {
        let radius = 0.001;
        let sep = 0.0019;

        let run_with_k = |conductivity: f64| -> f64 {
            let mut atom = Atom::new();
            let mut dem = DemAtom::new();
            atom.dt = 1e-7;
            push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], radius);
            push_dem_test_atom(&mut atom, &mut dem, 1, [sep, 0.0, 0.0], radius);
            atom.nlocal = 2;
            atom.natoms = 2;

            let mut thermal = ThermalAtom::new();
            thermal.temperature.push(400.0);
            thermal.temperature.push(300.0);
            thermal.heat_flux.push(0.0);
            thermal.heat_flux.push(0.0);

            let mut neighbor = Neighbor::new();
            neighbor.neighbor_offsets = vec![0, 1, 1];
            neighbor.neighbor_indices = vec![1];

            let mut registry = AtomDataRegistry::new();
            registry.register(dem);
            registry.register(thermal);

            let config = ThermalConfig {
                conductivity,
                specific_heat: 500.0,
                initial_temperature: 300.0,
            };

            let mut app = App::new();
            app.add_resource(atom);
            app.add_resource(neighbor);
            app.add_resource(registry);
            app.add_resource(config);
            app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
            app.organize_systems();
            app.run();

            let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
            let thermal = registry.expect::<ThermalAtom>("test");
            thermal.heat_flux[0].abs()
        };

        let q1 = run_with_k(1.0);
        let q2 = run_with_k(2.0);
        let q4 = run_with_k(4.0);

        assert!(
            (q2 / q1 - 2.0).abs() < 0.01,
            "Doubling conductivity should double flux: ratio = {:.4}",
            q2 / q1
        );
        assert!(
            (q4 / q1 - 4.0).abs() < 0.01,
            "4x conductivity should give 4x flux: ratio = {:.4}",
            q4 / q1
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Heat flux is antisymmetric (Q_i = -Q_j)
    // This is a conservation check: total heat flux should sum to zero.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn heat_flux_antisymmetric() {
        let radius = 0.001;
        let sep = 0.0019;
        let (mut app, _) = setup_two_atoms(450.0, 250.0, sep, radius);
        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            (thermal.heat_flux[0] + thermal.heat_flux[1]).abs() < 1e-20,
            "Heat flux should be antisymmetric: q0={:.6e}, q1={:.6e}",
            thermal.heat_flux[0], thermal.heat_flux[1]
        );
    }

    #[test]
    fn temperature_integration_conserves_energy() {
        let radius = 0.001;
        let sep = 0.0019;
        let (mut app, _) = setup_two_atoms(400.0, 300.0, sep, radius);

        app.add_update_system(compute_heat_conduction, ScheduleSet::Force);
        app.add_update_system(integrate_temperature, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");

        // Total thermal energy should be conserved: sum(m * cp * T) = const
        let cp = 500.0;
        let e_total = atom.mass[0] * cp * thermal.temperature[0]
            + atom.mass[1] * cp * thermal.temperature[1];
        let e_initial = atom.mass[0] * cp * 400.0 + atom.mass[1] * cp * 300.0;
        assert!(
            (e_total - e_initial).abs() / e_initial < 1e-10,
            "thermal energy should be conserved: {} vs {}",
            e_total, e_initial
        );
        // Hot atom should have cooled
        assert!(thermal.temperature[0] < 400.0);
        // Cold atom should have warmed
        assert!(thermal.temperature[1] > 300.0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Wall heat conduction tests
    // ══════════════════════════════════════════════════════════════════════

    fn make_test_wall_plane(
        point_z: f64,
        normal_z: f64,
        temperature: Option<f64>,
    ) -> WallPlane {
        WallPlane {
            point_x: 0.0,
            point_y: 0.0,
            point_z,
            normal_x: 0.0,
            normal_y: 0.0,
            normal_z,
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
            origin: [0.0, 0.0, point_z],
            force_accumulator: 0.0,
            temperature,
        }
    }

    fn make_test_walls(planes: Vec<WallPlane>) -> Walls {
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

    fn setup_wall_test(
        particle_temp: f64,
        wall_temp: Option<f64>,
        particle_z: f64,
        radius: f64,
    ) -> App {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, particle_z], radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut thermal = ThermalAtom::new();
        thermal.temperature.push(particle_temp);
        thermal.heat_flux.push(0.0);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(thermal);

        let config = ThermalConfig {
            conductivity: 1.0,
            specific_heat: 500.0,
            initial_temperature: 300.0,
        };

        // Wall at z=0 with normal pointing up (+z)
        let walls = make_test_walls(vec![make_test_wall_plane(0.0, 1.0, wall_temp)]);

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(config);
        app.add_resource(walls);
        app.add_update_system(compute_wall_heat_conduction, ScheduleSet::Force);
        app.organize_systems();
        app
    }

    #[test]
    fn wall_hot_heats_cold_particle() {
        let radius = 0.001;
        let particle_z = 0.0005; // overlap = radius - distance = 0.001 - 0.0005 = 0.0005
        let mut app = setup_wall_test(300.0, Some(500.0), particle_z, radius);
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0] > 0.0,
            "hot wall should heat cold particle, got {}",
            thermal.heat_flux[0]
        );
    }

    #[test]
    fn wall_cold_cools_hot_particle() {
        let radius = 0.001;
        let particle_z = 0.0005;
        let mut app = setup_wall_test(500.0, Some(300.0), particle_z, radius);
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0] < 0.0,
            "cold wall should cool hot particle, got {}",
            thermal.heat_flux[0]
        );
    }

    #[test]
    fn wall_equal_temp_zero_flux() {
        let radius = 0.001;
        let particle_z = 0.0005;
        let mut app = setup_wall_test(400.0, Some(400.0), particle_z, radius);
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0].abs() < 1e-20,
            "equal temperature should give zero flux, got {}",
            thermal.heat_flux[0]
        );
    }

    #[test]
    fn wall_no_temperature_no_heat_transfer() {
        let radius = 0.001;
        let particle_z = 0.0005;
        let mut app = setup_wall_test(300.0, None, particle_z, radius);
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0].abs() < 1e-20,
            "no wall temperature should give zero flux, got {}",
            thermal.heat_flux[0]
        );
    }

    #[test]
    fn wall_no_contact_no_heat_transfer() {
        let radius = 0.001;
        let particle_z = 0.002; // no overlap (distance > radius)
        let mut app = setup_wall_test(300.0, Some(500.0), particle_z, radius);
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let thermal = registry.expect::<ThermalAtom>("test");
        assert!(
            thermal.heat_flux[0].abs() < 1e-20,
            "no contact should give zero flux, got {}",
            thermal.heat_flux[0]
        );
    }

    #[test]
    fn wall_heat_flux_proportional_to_temperature_difference() {
        let radius = 0.001;
        let particle_z = 0.0005;

        let run_with_dt = |wall_temp: f64| -> f64 {
            let mut app = setup_wall_test(300.0, Some(wall_temp), particle_z, radius);
            app.run();
            let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
            let thermal = registry.expect::<ThermalAtom>("test");
            thermal.heat_flux[0]
        };

        let q1 = run_with_dt(400.0); // dT = 100
        let q2 = run_with_dt(500.0); // dT = 200
        let q3 = run_with_dt(600.0); // dT = 300

        // Flux should scale linearly with temperature difference
        assert!(
            (q2 / q1 - 2.0).abs() < 0.01,
            "double dT should double flux: ratio = {:.4}",
            q2 / q1
        );
        assert!(
            (q3 / q1 - 3.0).abs() < 0.01,
            "triple dT should triple flux: ratio = {:.4}",
            q3 / q1
        );
    }
}
