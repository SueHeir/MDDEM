//! DEM atom properties: named material types, per-pair mixing tables, and per-atom radius/density.

use std::any::TypeId;
use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use serde::Deserialize;

use mddem_core::{AtomData, AtomDataRegistry, AtomPlugin, Config};

// ── Config structs ──────────────────────────────────────────────────────────

fn default_friction() -> f64 {
    0.4
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
}

/// TOML `[dem]` — top-level DEM configuration containing material definitions.
#[derive(Deserialize, Clone, Default)]
pub struct DemConfig {
    pub materials: Option<Vec<MaterialConfig>>,
}

// ── MaterialTable — per-material and per-pair precomputed properties ────────

/// Per-material properties and per-pair precomputed mixing tables (geometric-mean).
pub struct MaterialTable {
    pub names: Vec<String>,
    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
    pub friction: Vec<f64>,
    pub restitution: Vec<f64>,
    pub beta_ij: Vec<Vec<f64>>,
    pub friction_ij: Vec<Vec<f64>>,
    /// Precomputed effective Young's modulus for each material pair (Hertz contact).
    pub e_eff_ij: Vec<Vec<f64>>,
    /// Precomputed effective shear modulus for each material pair (Mindlin contact).
    pub g_eff_ij: Vec<Vec<f64>>,
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
            beta_ij: Vec::new(),
            friction_ij: Vec::new(),
            e_eff_ij: Vec::new(),
            g_eff_ij: Vec::new(),
        }
    }

    pub fn add_material(
        &mut self,
        name: &str,
        youngs_mod: f64,
        poisson_ratio: f64,
        restitution: f64,
        friction: f64,
    ) -> u32 {
        let idx = self.names.len() as u32;
        self.names.push(name.to_string());
        self.youngs_mod.push(youngs_mod);
        self.poisson_ratio.push(poisson_ratio);
        self.restitution.push(restitution);
        self.friction.push(friction);
        idx
    }

    pub fn find_material(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }

    pub fn build_pair_tables(&mut self) {
        let n = self.names.len();
        self.beta_ij = vec![vec![0.0; n]; n];
        self.friction_ij = vec![vec![0.0; n]; n];
        self.e_eff_ij = vec![vec![0.0; n]; n];
        self.g_eff_ij = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                // Geometric mean mixing for restitution
                let e_ij = (self.restitution[i] * self.restitution[j]).sqrt();
                let log_e = e_ij.ln();
                self.beta_ij[i][j] = -log_e / (PI * PI + log_e * log_e).sqrt();

                // Geometric mean mixing for friction
                self.friction_ij[i][j] = (self.friction[i] * self.friction[j]).sqrt();

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

/// Per-atom DEM extension data: particle radius, density, and precomputed inverse inertia.
#[derive(AtomData)]
pub struct DemAtom {
    pub radius: Vec<f64>,
    pub density: Vec<f64>,
    pub inv_inertia: Vec<f64>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_material_beta_and_friction() {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
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
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
        mt.add_material("steel", 200e9, 0.28, 0.8, 0.3);
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
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
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

        if let Some(registry_option) = app.get_mut_resource(TypeId::of::<AtomDataRegistry>()) {
            let mut registry_binder = registry_option.borrow_mut();
            let registry = registry_binder.downcast_mut::<AtomDataRegistry>().unwrap();
            registry.register(DemAtom::new());
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }

        let dem_config = Config::load::<DemConfig>(app, "dem");

        // Build MaterialTable from config at plugin build time
        let mut material_table = MaterialTable::new();

        if let Some(ref materials) = dem_config.materials {
            for mat in materials {
                material_table.add_material(
                    &mat.name,
                    mat.youngs_mod,
                    mat.poisson_ratio,
                    mat.restitution,
                    mat.friction,
                );
            }
            material_table.build_pair_tables();
        }

        app.add_resource(material_table);
    }
}
