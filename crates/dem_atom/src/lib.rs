//! DEM atom properties: named material types, per-pair mixing tables, per-atom radius/density,
//! and particle insertion (random, rate-based, file-based) from `[[particles.insert]]` config.

pub mod insert;
pub mod radius;

pub use insert::*;
pub use radius::*;

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{register_atom_data, Atom, AtomData, AtomPlugin, Config};

// ── Shared physics constants ────────────────────────────────────────────────

/// √(5/3) — viscoelastic damping coefficient used in Hertz contact models.
pub const SQRT_5_3: f64 = 0.9128709291752768;

// ── Config structs ──────────────────────────────────────────────────────────

fn default_friction() -> f64 {
    0.4
}

fn default_contact_model() -> String {
    "hertz".to_string()
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// A single material definition from `[[dem.materials]]`.
pub struct MaterialConfig {
    /// Material name, referenced by particle insert blocks.
    pub name: String,
    /// Young's modulus (Pa).
    pub youngs_mod: f64,
    /// Poisson's ratio (dimensionless, 0–0.5).
    pub poisson_ratio: f64,
    /// Coefficient of restitution (0–1).
    pub restitution: f64,
    /// Coulomb friction coefficient.
    #[serde(default = "default_friction")]
    pub friction: f64,
    /// Rolling friction coefficient (0 = disabled).
    #[serde(default)]
    pub rolling_friction: f64,
    /// Cohesion energy density (J/m², 0 = disabled). SJKR model.
    #[serde(default)]
    pub cohesion_energy: f64,
    /// Surface energy (J/m², 0 = disabled). Activates JKR adhesion model.
    /// Cannot be used together with cohesion_energy on the same material.
    #[serde(default)]
    pub surface_energy: f64,
    /// Twisting friction coefficient (0 = disabled).
    #[serde(default)]
    pub twisting_friction: f64,
    /// Linear normal stiffness for Hooke model (N/m, 0 = use Hertz).
    #[serde(default)]
    pub kn: f64,
    /// Linear tangential stiffness for Hooke model (N/m, 0 = use Mindlin).
    #[serde(default)]
    pub kt: f64,
}

/// TOML `[dem]` — top-level DEM configuration containing material definitions.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct DemConfig {
    pub materials: Option<Vec<MaterialConfig>>,
    /// Contact model: "hertz" (default) or "hooke".
    #[serde(default = "default_contact_model")]
    pub contact_model: String,
}

// ── MaterialTable — per-material and per-pair precomputed properties ────────

/// Per-material properties and per-pair precomputed mixing tables (geometric-mean).
pub struct MaterialTable {
    pub names: Vec<String>,
    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
    pub friction: Vec<f64>,
    pub restitution: Vec<f64>,
    pub rolling_friction: Vec<f64>,
    pub cohesion_energy: Vec<f64>,
    pub surface_energy: Vec<f64>,
    pub beta_ij: Vec<Vec<f64>>,
    pub friction_ij: Vec<Vec<f64>>,
    pub rolling_friction_ij: Vec<Vec<f64>>,
    pub cohesion_energy_ij: Vec<Vec<f64>>,
    /// Per-pair surface energy for JKR adhesion (geometric mean mixing).
    pub surface_energy_ij: Vec<Vec<f64>>,
    /// Precomputed effective Young's modulus for each material pair (Hertz contact).
    pub e_eff_ij: Vec<Vec<f64>>,
    /// Precomputed effective shear modulus for each material pair (Mindlin contact).
    pub g_eff_ij: Vec<Vec<f64>>,
    /// Per-pair twisting friction (geometric mean mixing).
    pub twisting_friction_ij: Vec<Vec<f64>>,
    /// Per-material linear normal stiffness for Hooke model.
    pub kn: Vec<f64>,
    /// Per-material linear tangential stiffness for Hooke model.
    pub kt: Vec<f64>,
    /// Per-pair Hooke normal stiffness (harmonic mean: 2*ki*kj/(ki+kj)).
    pub kn_ij: Vec<Vec<f64>>,
    /// Per-pair Hooke tangential stiffness (harmonic mean).
    pub kt_ij: Vec<Vec<f64>>,
    /// Contact model: "hertz" or "hooke".
    pub contact_model: String,
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialTable {
    pub fn new() -> Self {
        MaterialTable {
            names: Vec::new(),
            youngs_mod: Vec::new(),
            poisson_ratio: Vec::new(),
            friction: Vec::new(),
            restitution: Vec::new(),
            rolling_friction: Vec::new(),
            cohesion_energy: Vec::new(),
            surface_energy: Vec::new(),
            beta_ij: Vec::new(),
            friction_ij: Vec::new(),
            rolling_friction_ij: Vec::new(),
            cohesion_energy_ij: Vec::new(),
            surface_energy_ij: Vec::new(),
            e_eff_ij: Vec::new(),
            g_eff_ij: Vec::new(),
            twisting_friction_ij: Vec::new(),
            kn: Vec::new(),
            kt: Vec::new(),
            kn_ij: Vec::new(),
            kt_ij: Vec::new(),
            contact_model: "hertz".to_string(),
        }
    }

    pub fn add_material(
        &mut self,
        name: &str,
        youngs_mod: f64,
        poisson_ratio: f64,
        restitution: f64,
        friction: f64,
        rolling_friction: f64,
        cohesion_energy: f64,
    ) -> u32 {
        self.add_material_full(name, youngs_mod, poisson_ratio, restitution, friction, rolling_friction, cohesion_energy, 0.0)
    }

    pub fn add_material_full(
        &mut self,
        name: &str,
        youngs_mod: f64,
        poisson_ratio: f64,
        restitution: f64,
        friction: f64,
        rolling_friction: f64,
        cohesion_energy: f64,
        surface_energy: f64,
    ) -> u32 {
        if cohesion_energy > 0.0 && surface_energy > 0.0 {
            eprintln!(
                "ERROR: material '{}' has both cohesion_energy and surface_energy > 0. Use only one.",
                name
            );
            std::process::exit(1);
        }
        let idx = self.names.len() as u32;
        self.names.push(name.to_string());
        self.youngs_mod.push(youngs_mod);
        self.poisson_ratio.push(poisson_ratio);
        self.restitution.push(restitution);
        self.friction.push(friction);
        self.rolling_friction.push(rolling_friction);
        self.cohesion_energy.push(cohesion_energy);
        self.surface_energy.push(surface_energy);
        idx
    }

    pub fn find_material(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }

    pub fn build_pair_tables(&mut self) {
        let n = self.names.len();
        self.beta_ij = vec![vec![0.0; n]; n];
        self.friction_ij = vec![vec![0.0; n]; n];
        self.rolling_friction_ij = vec![vec![0.0; n]; n];
        self.cohesion_energy_ij = vec![vec![0.0; n]; n];
        self.surface_energy_ij = vec![vec![0.0; n]; n];
        self.e_eff_ij = vec![vec![0.0; n]; n];
        self.g_eff_ij = vec![vec![0.0; n]; n];
        // Pad surface_energy if add_material (old API) was used
        while self.surface_energy.len() < n {
            self.surface_energy.push(0.0);
        }
        for i in 0..n {
            for j in 0..n {
                // Geometric mean mixing for restitution
                let e_ij = (self.restitution[i] * self.restitution[j]).sqrt();
                let log_e = e_ij.ln();
                self.beta_ij[i][j] = -log_e / (PI * PI + log_e * log_e).sqrt();

                // Geometric mean mixing for friction
                self.friction_ij[i][j] = (self.friction[i] * self.friction[j]).sqrt();

                // Geometric mean mixing for rolling friction
                self.rolling_friction_ij[i][j] =
                    (self.rolling_friction[i] * self.rolling_friction[j]).sqrt();

                // Geometric mean mixing for cohesion energy
                self.cohesion_energy_ij[i][j] =
                    (self.cohesion_energy[i] * self.cohesion_energy[j]).sqrt();

                // Geometric mean mixing for surface energy (JKR)
                self.surface_energy_ij[i][j] =
                    (self.surface_energy[i] * self.surface_energy[j]).sqrt();

                // Effective Young's modulus (Hertz)
                let nu_i = self.poisson_ratio[i];
                let nu_j = self.poisson_ratio[j];
                self.e_eff_ij[i][j] = 1.0
                    / ((1.0 - nu_i * nu_i) / self.youngs_mod[i]
                        + (1.0 - nu_j * nu_j) / self.youngs_mod[j]);

                // Effective shear modulus (Mindlin)
                self.g_eff_ij[i][j] = 1.0
                    / (2.0 * (2.0 - nu_i) * (1.0 + nu_i) / self.youngs_mod[i]
                        + 2.0 * (2.0 - nu_j) * (1.0 + nu_j) / self.youngs_mod[j]);
            }
        }
    }
}

// ── DemAtom per-atom data ────────────────────────────────────────────────────

/// Per-atom DEM extension data: particle radius, density, inverse inertia, and rotational fields.
#[derive(AtomData)]
pub struct DemAtom {
    pub radius: Vec<f64>,
    pub density: Vec<f64>,
    pub inv_inertia: Vec<f64>,
    pub quaternion: Vec<[f64; 4]>,
    #[forward]
    pub omega: Vec<[f64; 3]>,
    pub ang_mom: Vec<[f64; 3]>,
    #[reverse]
    #[zero]
    pub torque: Vec<[f64; 3]>,
}

impl Default for DemAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl DemAtom {
    pub fn new() -> Self {
        DemAtom {
            radius: Vec::new(),
            density: Vec::new(),
            inv_inertia: Vec::new(),
            quaternion: Vec::new(),
            omega: Vec::new(),
            ang_mom: Vec::new(),
            torque: Vec::new(),
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`DemAtom`] extension and [`MaterialTable`] from `[[dem.materials]]` config.
pub struct DemAtomPlugin;

impl Plugin for DemAtomPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Material definitions for DEM particles
[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4
# rolling_friction = 0.1      # rolling resistance coefficient (default 0.0 = disabled)
# cohesion_energy = 0.05       # SJKR cohesion energy density J/m² (default 0.0 = disabled)
# surface_energy = 0.05        # JKR surface energy J/m² (default 0.0 = disabled)

# Additional materials can be added:
# [[dem.materials]]
# name = "steel"
# youngs_mod = 200e9
# poisson_ratio = 0.28
# restitution = 0.8
# friction = 0.3"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_plugins(AtomPlugin);

        register_atom_data!(app, DemAtom::new());

        let dem_config = Config::load::<DemConfig>(app, "dem");

        // Build MaterialTable from config at plugin build time
        let mut material_table = MaterialTable::new();

        if let Some(ref materials) = dem_config.materials {
            for mat in materials {
                material_table.add_material_full(
                    &mat.name,
                    mat.youngs_mod,
                    mat.poisson_ratio,
                    mat.restitution,
                    mat.friction,
                    mat.rolling_friction,
                    mat.cohesion_energy,
                    mat.surface_energy,
                );
            }
            material_table.build_pair_tables();
        }

        app.add_resource(material_table);
        app.add_setup_system(set_dem_ntypes, ScheduleSetupSet::Setup);
    }
}

fn set_dem_ntypes(mut atoms: ResMut<Atom>, material_table: Res<MaterialTable>) {
    if !material_table.names.is_empty() {
        atoms.ntypes = material_table.names.len();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_material_beta_and_friction() {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0);
        mt.build_pair_tables();

        let e = 0.95_f64;
        let log_e = e.ln();
        let expected_beta = -log_e / (PI * PI + log_e * log_e).sqrt();
        assert!(
            (mt.beta_ij[0][0] - expected_beta).abs() < 1e-12,
            "beta should be {}, got {}",
            expected_beta,
            mt.beta_ij[0][0]
        );
        assert!(
            (mt.friction_ij[0][0] - 0.4).abs() < 1e-12,
            "friction should be 0.4, got {}",
            mt.friction_ij[0][0]
        );
    }

    #[test]
    fn multi_material_mixing_symmetry() {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0);
        mt.add_material("steel", 200e9, 0.28, 0.8, 0.3, 0.0, 0.0);
        mt.build_pair_tables();

        // Symmetry
        assert!(
            (mt.beta_ij[0][1] - mt.beta_ij[1][0]).abs() < 1e-15,
            "beta_ij should be symmetric"
        );
        assert!(
            (mt.friction_ij[0][1] - mt.friction_ij[1][0]).abs() < 1e-15,
            "friction_ij should be symmetric"
        );

        // Geometric mean mixing for friction
        let expected_friction = (0.4_f64 * 0.3).sqrt();
        assert!(
            (mt.friction_ij[0][1] - expected_friction).abs() < 1e-12,
            "friction_ij should be geometric mean {}, got {}",
            expected_friction,
            mt.friction_ij[0][1]
        );

        // Geometric mean mixing for restitution -> beta
        let e_mix = (0.95_f64 * 0.8).sqrt();
        let log_e = e_mix.ln();
        let expected_beta = -log_e / (PI * PI + log_e * log_e).sqrt();
        assert!(
            (mt.beta_ij[0][1] - expected_beta).abs() < 1e-12,
            "beta_ij should use geometric mean restitution"
        );

        // e_eff and g_eff symmetry
        assert!(
            (mt.e_eff_ij[0][1] - mt.e_eff_ij[1][0]).abs() < 1e-6,
            "e_eff_ij should be symmetric"
        );
        assert!(
            (mt.g_eff_ij[0][1] - mt.g_eff_ij[1][0]).abs() < 1e-6,
            "g_eff_ij should be symmetric"
        );
        assert!(mt.e_eff_ij[0][0] > 0.0, "e_eff should be positive");
        assert!(mt.g_eff_ij[0][0] > 0.0, "g_eff should be positive");
    }

    #[test]
    fn e_eff_matches_manual_computation() {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0);
        mt.build_pair_tables();

        let nu = 0.3_f64;
        let e = 8.7e9_f64;
        let expected = 1.0 / (2.0 * (1.0 - nu * nu) / e);
        assert!(
            (mt.e_eff_ij[0][0] - expected).abs() < 1.0,
            "e_eff_ij[0][0] should be {}, got {}",
            expected,
            mt.e_eff_ij[0][0]
        );
    }
}
