//! # dem_atom — Per-atom DEM data and material property tables
//!
//! This crate provides the core per-atom data structures and material property
//! system for Discrete Element Method (DEM) simulations in MDDEM:
//!
//! - **[`DemAtom`]** — Per-atom extension data (radius, density, angular velocity, torque, etc.)
//!   registered via the `AtomData` derive macro with `#[forward]`, `#[reverse]`, and `#[zero]`
//!   attributes that control MPI pack/unpack and per-timestep zeroing behavior.
//! - **[`MaterialTable`]** — Named materials with mechanical properties (Young's modulus,
//!   Poisson's ratio, restitution, friction) and precomputed per-pair mixing tables
//!   (geometric-mean or harmonic-mean) for efficient contact force evaluation.
//! - **[`DemAtomInsertPlugin`]** — Particle insertion from `[[particles.insert]]` TOML config:
//!   random placement, rate-based trickle insertion, or file-based loading (CSV, LAMMPS dump/data).
//! - **[`RadiusSpec`]** — Particle radius specification: fixed value or statistical distribution
//!   (uniform, gaussian, lognormal, discrete).
//!
//! ## TOML Configuration
//!
//! Materials are defined under `[[dem.materials]]`:
//!
//! ```toml
//! [dem]
//! contact_model = "hertz"  # "hertz" (default) or "hooke"
//!
//! [[dem.materials]]
//! name = "glass"
//! youngs_mod = 8.7e9       # Young's modulus (Pa)
//! poisson_ratio = 0.3      # Poisson's ratio (dimensionless, 0–0.5)
//! restitution = 0.95       # Coefficient of restitution (0–1)
//! friction = 0.4           # Coulomb friction coefficient (default 0.4)
//! ```
//!
//! ## AtomData Derive Attributes
//!
//! The `#[forward]`, `#[reverse]`, and `#[zero]` attributes on [`DemAtom`] fields
//! control how data is handled during MPI communication and timestep integration:
//!
//! - **`#[forward]`** — Field is packed and sent forward from owning processor to ghost
//!   atoms on neighboring processors (e.g., angular velocity `omega`).
//! - **`#[reverse]`** — Field is accumulated in reverse: ghost contributions are sent back
//!   and summed into the owning atom's value (e.g., `torque`).
//! - **`#[zero]`** — Field is zeroed at the start of each force computation step (e.g., `torque`),
//!   before new contact forces accumulate.

pub mod insert;
pub mod radius;

pub use insert::*;
pub use radius::*;

use std::f64::consts::PI;

use grass_app::prelude::*;
use mddem_derive::AtomData;
use grass_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{register_atom_data, Atom, AtomData, AtomPlugin, Config, ScheduleSetupSet};

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
    /// Rolling spring stiffness for SDS rolling model (N·m/rad, 0 = use constant model).
    #[serde(default)]
    pub rolling_stiffness: f64,
    /// Rolling viscous damping coefficient for SDS rolling model.
    #[serde(default)]
    pub rolling_damping: f64,
    /// Twisting spring stiffness for SDS twisting model (N·m/rad, 0 = use constant model).
    #[serde(default)]
    pub twisting_stiffness: f64,
    /// Twisting viscous damping coefficient for SDS twisting model.
    #[serde(default)]
    pub twisting_damping: f64,
}

fn default_adhesion_model() -> String {
    "jkr".to_string()
}

fn default_rolling_model() -> String {
    "constant".to_string()
}

fn default_twisting_model() -> String {
    "constant".to_string()
}

/// TOML `[dem]` — top-level DEM configuration containing material definitions.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct DemConfig {
    /// List of material definitions, each corresponding to a `[[dem.materials]]` block.
    pub materials: Option<Vec<MaterialConfig>>,
    /// Contact model: "hertz" (default) or "hooke".
    #[serde(default = "default_contact_model")]
    pub contact_model: String,
    /// Adhesion model when `surface_energy > 0`: "jkr" (default) or "dmt".
    ///
    /// - **JKR** (Johnson-Kendall-Roberts): Modified contact area with pull-off force = 1.5*pi*gamma*R*.
    ///   Suitable for soft materials with high surface energy (Tabor parameter > 5).
    /// - **DMT** (Derjaguin-Muller-Toporov): Pure Hertz contact area with constant attractive
    ///   force = 2*pi*gamma*R*. Suitable for stiff materials with low surface energy (Tabor parameter < 0.1).
    #[serde(default = "default_adhesion_model")]
    pub adhesion_model: String,
    /// Rolling resistance model: "constant" (default) or "sds".
    #[serde(default = "default_rolling_model")]
    pub rolling_model: String,
    /// Twisting friction model: "constant" (default) or "sds".
    #[serde(default = "default_twisting_model")]
    pub twisting_model: String,
}

// ── MaterialTable — per-material and per-pair precomputed properties ────────

/// Per-material properties and precomputed per-pair mixing tables for contact force evaluation.
///
/// Pair properties are computed in [`build_pair_tables`](Self::build_pair_tables) using:
/// - **Geometric mean** for friction, restitution, cohesion/surface energy, rolling/twisting friction
/// - **Harmonic mean** (2·ki·kj/(ki+kj)) for Hooke stiffnesses and SDS spring stiffnesses
/// - **Effective modulus** formulas for Hertz (`E*`) and Mindlin (`G*`) contact models
///
/// Indexed by material index (returned by [`add_material`](Self::add_material) and
/// [`find_material`](Self::find_material)). Pair tables are indexed as `table_ij[i][j]`.
pub struct MaterialTable {
    pub names: Vec<String>,
    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
    pub friction: Vec<f64>,
    pub restitution: Vec<f64>,
    pub rolling_friction: Vec<f64>,
    pub twisting_friction: Vec<f64>,
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
    /// Adhesion model: "jkr" (default) or "dmt".
    pub adhesion_model: String,
    /// Rolling resistance model: "constant" or "sds".
    pub rolling_model: String,
    /// Twisting friction model: "constant" or "sds".
    pub twisting_model: String,
    /// Per-material rolling spring stiffness (SDS model).
    pub rolling_stiffness: Vec<f64>,
    /// Per-material rolling damping coefficient (SDS model).
    pub rolling_damping: Vec<f64>,
    /// Per-material twisting spring stiffness (SDS model).
    pub twisting_stiffness: Vec<f64>,
    /// Per-material twisting damping coefficient (SDS model).
    pub twisting_damping: Vec<f64>,
    /// Per-pair rolling stiffness (harmonic mean).
    pub rolling_stiffness_ij: Vec<Vec<f64>>,
    /// Per-pair rolling damping (geometric mean).
    pub rolling_damping_ij: Vec<Vec<f64>>,
    /// Per-pair twisting stiffness (harmonic mean).
    pub twisting_stiffness_ij: Vec<Vec<f64>>,
    /// Per-pair twisting damping (geometric mean).
    pub twisting_damping_ij: Vec<Vec<f64>>,
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialTable {
    /// Creates an empty `MaterialTable` with default contact/adhesion/rolling/twisting models.
    pub fn new() -> Self {
        MaterialTable {
            names: Vec::new(),
            youngs_mod: Vec::new(),
            poisson_ratio: Vec::new(),
            friction: Vec::new(),
            restitution: Vec::new(),
            rolling_friction: Vec::new(),
            twisting_friction: Vec::new(),
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
            adhesion_model: "jkr".to_string(),
            rolling_model: "constant".to_string(),
            twisting_model: "constant".to_string(),
            rolling_stiffness: Vec::new(),
            rolling_damping: Vec::new(),
            twisting_stiffness: Vec::new(),
            twisting_damping: Vec::new(),
            rolling_stiffness_ij: Vec::new(),
            rolling_damping_ij: Vec::new(),
            twisting_stiffness_ij: Vec::new(),
            twisting_damping_ij: Vec::new(),
        }
    }

    /// Adds a material with basic properties. Returns the material index.
    ///
    /// This is a convenience wrapper around [`add_material_full`](Self::add_material_full)
    /// with `surface_energy = 0.0` (no adhesion).
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

    /// Adds a material with basic properties plus surface energy. Returns the material index.
    ///
    /// Wraps [`add_material_extended`](Self::add_material_extended) with twisting/Hooke stiffness = 0.
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
        self.add_material_extended(
            name, youngs_mod, poisson_ratio, restitution, friction,
            rolling_friction, cohesion_energy, surface_energy, 0.0, 0.0, 0.0,
        )
    }

    /// Adds a material with twisting friction and Hooke stiffnesses. Returns the material index.
    ///
    /// Wraps [`add_material_with_sds`](Self::add_material_with_sds) with SDS parameters = 0.
    pub fn add_material_extended(
        &mut self,
        name: &str,
        youngs_mod: f64,
        poisson_ratio: f64,
        restitution: f64,
        friction: f64,
        rolling_friction: f64,
        cohesion_energy: f64,
        surface_energy: f64,
        twisting_friction: f64,
        kn: f64,
        kt: f64,
    ) -> u32 {
        self.add_material_with_sds(
            name, youngs_mod, poisson_ratio, restitution, friction,
            rolling_friction, cohesion_energy, surface_energy, twisting_friction,
            kn, kt, 0.0, 0.0, 0.0, 0.0,
        )
    }

    /// Add a material with all fields including SDS rolling/twisting parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn add_material_with_sds(
        &mut self,
        name: &str,
        youngs_mod: f64,
        poisson_ratio: f64,
        restitution: f64,
        friction: f64,
        rolling_friction: f64,
        cohesion_energy: f64,
        surface_energy: f64,
        twisting_friction: f64,
        kn: f64,
        kt: f64,
        rolling_stiffness: f64,
        rolling_damping: f64,
        twisting_stiffness: f64,
        twisting_damping: f64,
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
        self.twisting_friction.push(twisting_friction);
        self.cohesion_energy.push(cohesion_energy);
        self.surface_energy.push(surface_energy);
        self.kn.push(kn);
        self.kt.push(kt);
        self.rolling_stiffness.push(rolling_stiffness);
        self.rolling_damping.push(rolling_damping);
        self.twisting_stiffness.push(twisting_stiffness);
        self.twisting_damping.push(twisting_damping);
        idx
    }

    /// Looks up a material by name, returning its index if found.
    pub fn find_material(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }

    /// Computes all per-pair mixing tables from the registered per-material properties.
    ///
    /// Must be called after all materials have been added. Populates `*_ij` fields using
    /// geometric-mean or harmonic-mean mixing rules as appropriate for each property.
    pub fn build_pair_tables(&mut self) {
        let n = self.names.len();
        self.beta_ij = vec![vec![0.0; n]; n];
        self.friction_ij = vec![vec![0.0; n]; n];
        self.rolling_friction_ij = vec![vec![0.0; n]; n];
        self.cohesion_energy_ij = vec![vec![0.0; n]; n];
        self.surface_energy_ij = vec![vec![0.0; n]; n];
        self.e_eff_ij = vec![vec![0.0; n]; n];
        self.g_eff_ij = vec![vec![0.0; n]; n];
        self.twisting_friction_ij = vec![vec![0.0; n]; n];
        self.kn_ij = vec![vec![0.0; n]; n];
        self.kt_ij = vec![vec![0.0; n]; n];
        self.rolling_stiffness_ij = vec![vec![0.0; n]; n];
        self.rolling_damping_ij = vec![vec![0.0; n]; n];
        self.twisting_stiffness_ij = vec![vec![0.0; n]; n];
        self.twisting_damping_ij = vec![vec![0.0; n]; n];
        // Pad optional fields if old API was used
        while self.surface_energy.len() < n {
            self.surface_energy.push(0.0);
        }
        while self.twisting_friction.len() < n {
            self.twisting_friction.push(0.0);
        }
        while self.kn.len() < n {
            self.kn.push(0.0);
        }
        while self.kt.len() < n {
            self.kt.push(0.0);
        }
        while self.rolling_stiffness.len() < n {
            self.rolling_stiffness.push(0.0);
        }
        while self.rolling_damping.len() < n {
            self.rolling_damping.push(0.0);
        }
        while self.twisting_stiffness.len() < n {
            self.twisting_stiffness.push(0.0);
        }
        while self.twisting_damping.len() < n {
            self.twisting_damping.push(0.0);
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

                // Geometric mean mixing for twisting friction
                self.twisting_friction_ij[i][j] =
                    (self.twisting_friction[i].max(0.0) * self.twisting_friction[j].max(0.0)).sqrt();

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

                // Harmonic mean mixing for Hooke stiffness
                let ki = self.kn[i];
                let kj = self.kn[j];
                self.kn_ij[i][j] = if ki > 0.0 && kj > 0.0 {
                    2.0 * ki * kj / (ki + kj)
                } else {
                    0.0
                };
                let kti = self.kt[i];
                let ktj = self.kt[j];
                self.kt_ij[i][j] = if kti > 0.0 && ktj > 0.0 {
                    2.0 * kti * ktj / (kti + ktj)
                } else {
                    0.0
                };

                // SDS rolling stiffness (harmonic mean)
                let kri = self.rolling_stiffness[i];
                let krj = self.rolling_stiffness[j];
                self.rolling_stiffness_ij[i][j] = if kri > 0.0 && krj > 0.0 {
                    2.0 * kri * krj / (kri + krj)
                } else if kri > 0.0 {
                    kri
                } else {
                    krj
                };

                // SDS rolling damping (geometric mean)
                self.rolling_damping_ij[i][j] =
                    (self.rolling_damping[i].max(0.0) * self.rolling_damping[j].max(0.0)).sqrt();

                // SDS twisting stiffness (harmonic mean)
                let kwi = self.twisting_stiffness[i];
                let kwj = self.twisting_stiffness[j];
                self.twisting_stiffness_ij[i][j] = if kwi > 0.0 && kwj > 0.0 {
                    2.0 * kwi * kwj / (kwi + kwj)
                } else if kwi > 0.0 {
                    kwi
                } else {
                    kwj
                };

                // SDS twisting damping (geometric mean)
                self.twisting_damping_ij[i][j] =
                    (self.twisting_damping[i].max(0.0) * self.twisting_damping[j].max(0.0)).sqrt();
            }
        }
    }
}

// ── DemAtom per-atom data ────────────────────────────────────────────────────

/// Per-atom DEM extension data: particle radius, density, inverse inertia, and rotational fields.
///
/// Registered via [`register_atom_data!`] in [`DemAtomPlugin::build`]. The `AtomData` derive
/// macro generates pack/unpack methods for MPI communication based on field attributes:
///
/// - **`#[forward]`** — Sent from owner to ghost atoms each timestep (e.g., `omega`).
/// - **`#[reverse]`** — Accumulated from ghosts back to owner (e.g., `torque`).
/// - **`#[zero]`** — Zeroed before each force computation (e.g., `torque`).
///
/// Fields without attributes are only communicated during atom migration (ownership transfer).
#[derive(AtomData)]
pub struct DemAtom {
    /// Particle radius (m). Set at insertion time; used for contact detection and force calculation.
    pub radius: Vec<f64>,
    /// Particle material density (kg/m³). Used with radius to compute mass and moment of inertia.
    pub density: Vec<f64>,
    /// Inverse moment of inertia (1/(I) where I = 2/5 * m * r²) (1/(kg·m²)).
    /// Precomputed at insertion for efficient torque-to-angular-acceleration conversion.
    pub inv_inertia: Vec<f64>,
    /// Orientation quaternion [w, x, y, z] (unit quaternion). Initialized to [1, 0, 0, 0].
    pub quaternion: Vec<[f64; 4]>,
    /// Angular velocity (rad/s) in [x, y, z] components.
    /// Marked `#[forward]`: communicated from owner to ghost atoms each timestep.
    #[forward]
    pub omega: Vec<[f64; 3]>,
    /// Angular momentum (kg·m²/s) in [x, y, z] components.
    pub ang_mom: Vec<[f64; 3]>,
    /// Torque (N·m) in [x, y, z] components.
    /// Marked `#[reverse]`: ghost contributions are summed back to the owning atom.
    /// Marked `#[zero]`: zeroed before each force computation step.
    #[reverse]
    #[zero]
    pub torque: Vec<[f64; 3]>,
    /// Rigid body ID for clump/multisphere membership. 0.0 = independent particle.
    /// Positive values indicate sub-spheres of the same rigid body (same value = same body).
    #[forward]
    pub body_id: Vec<f64>,
}

impl Default for DemAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl DemAtom {
    /// Creates an empty `DemAtom` with no particles. Particle data is appended during insertion.
    pub fn new() -> Self {
        DemAtom {
            radius: Vec::new(),
            density: Vec::new(),
            inv_inertia: Vec::new(),
            quaternion: Vec::new(),
            omega: Vec::new(),
            ang_mom: Vec::new(),
            torque: Vec::new(),
            body_id: Vec::new(),
        }
    }
}

/// Returns true if atoms `i` and `j` belong to the same rigid body.
#[inline]
pub fn same_body(dem: &DemAtom, i: usize, j: usize) -> bool {
    let bi = dem.body_id[i];
    let bj = dem.body_id[j];
    bi > 0.0 && bj > 0.0 && (bi - bj).abs() < 0.5
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`DemAtom`] extension and [`MaterialTable`] from `[[dem.materials]]` config.
pub struct DemAtomPlugin;

impl Plugin for DemAtomPlugin {
    fn provides(&self) -> Vec<&str> {
        vec!["dem_particles"]
    }

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
# surface_energy = 0.05        # JKR/DMT surface energy J/m² (default 0.0 = disabled)
# adhesion_model = "jkr"       # "jkr" (default) or "dmt" when surface_energy > 0

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

        material_table.contact_model = dem_config.contact_model.clone();
        material_table.adhesion_model = dem_config.adhesion_model.clone();
        material_table.rolling_model = dem_config.rolling_model.clone();
        material_table.twisting_model = dem_config.twisting_model.clone();

        if let Some(ref materials) = dem_config.materials {
            for mat in materials {
                material_table.add_material_with_sds(
                    &mat.name,
                    mat.youngs_mod,
                    mat.poisson_ratio,
                    mat.restitution,
                    mat.friction,
                    mat.rolling_friction,
                    mat.cohesion_energy,
                    mat.surface_energy,
                    mat.twisting_friction,
                    mat.kn,
                    mat.kt,
                    mat.rolling_stiffness,
                    mat.rolling_damping,
                    mat.twisting_stiffness,
                    mat.twisting_damping,
                );
            }
            material_table.build_pair_tables();
        }

        app.add_resource(material_table);
        app.add_setup_system(set_dem_ntypes, ScheduleSetupSet::Setup);
    }
}

/// Setup system that sets `Atom::ntypes` from the number of registered materials.
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
