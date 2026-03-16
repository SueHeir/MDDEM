//! Contact-based heat conduction for DEM particles.
//!
//! Per-atom temperature with heat transfer through contact area:
//! `Q = conductivity * 2*a * (Tj - Ti)` where `a = sqrt(r_eff * delta)`.

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, Config};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

/// TOML `[thermal]` — heat conduction configuration.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ThermalConfig {
    /// Thermal conductivity (W/(m·K)).
    pub conductivity: f64,
    /// Specific heat capacity (J/(kg·K)).
    pub specific_heat: f64,
    /// Initial temperature for all particles (K).
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

/// Per-atom thermal extension: temperature and heat flux.
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
    pub fn new() -> Self {
        ThermalAtom {
            temperature: Vec::new(),
            heat_flux: Vec::new(),
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers thermal per-atom data and heat conduction systems.
pub struct ThermalPlugin;

impl Plugin for ThermalPlugin {
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
        app.add_update_system(integrate_temperature, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Set initial temperatures for all atoms.
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

/// Compute heat conduction through contacts: Q = conductivity * 2*a * (Tj - Ti).
pub fn compute_heat_conduction(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    config: Res<ThermalConfig>,
) {
    let dem = registry.expect::<DemAtom>("compute_heat_conduction");
    let mut thermal = registry.expect_mut::<ThermalAtom>("compute_heat_conduction");

    // Ensure thermal vectors cover all atoms
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
        let r_eff = (r1 * r2) / sum_r;

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq >= sum_r * sum_r {
            continue;
        }

        let distance = dist_sq.sqrt();
        let delta = sum_r - distance;
        if delta <= 0.0 {
            continue;
        }

        // Contact radius: a = sqrt(r_eff * delta)
        let a = (r_eff * delta).sqrt();

        // Heat transfer: Q = conductivity * 2*a * (Tj - Ti)
        let dt_temp = thermal.temperature[j] - thermal.temperature[i];
        let q = k * 2.0 * a * dt_temp;

        thermal.heat_flux[i] += q;
        thermal.heat_flux[j] -= q;
    }
}

/// Integrate temperature: T += dt * heat_flux / (mass * cp).
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
        thermal.temperature[i] += dt * thermal.heat_flux[i] / (atoms.mass[i] * cp);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
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
}
