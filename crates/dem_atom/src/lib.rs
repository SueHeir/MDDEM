//! DEM atom properties: named material types, per-pair mixing tables, per-atom radius/density,
//! and random particle insertion from `[[particles.insert]]` config blocks.

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, AtomPlugin, CommResource, Config, Domain, Region, RunConfig, StageOverrides};

// ── Shared physics constants ────────────────────────────────────────────────

/// √(5/3) — viscoelastic damping coefficient used in Hertz contact models.
pub const SQRT_5_3: f64 = 0.9128709291752768;

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
#[serde(deny_unknown_fields)]
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

        register_atom_data!(app, DemAtom::new());

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
        app.add_setup_system(set_dem_ntypes, ScheduleSetupSet::Setup);
    }
}

fn set_dem_ntypes(mut atoms: ResMut<Atom>, material_table: Res<MaterialTable>) {
    if !material_table.names.is_empty() {
        atoms.ntypes = material_table.names.len();
    }
}

// ── Particle insertion ─────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// A single particle insertion block from `[[particles.insert]]`.
pub struct InsertConfig {
    pub material: String,
    pub count: u32,
    pub radius: f64,
    pub density: f64,
    pub velocity: Option<f64>,
    pub velocity_x: Option<f64>,
    pub velocity_y: Option<f64>,
    pub velocity_z: Option<f64>,
    /// Insertion region. Defaults to domain bounds (inset by particle radius).
    #[serde(default)]
    pub region: Option<Region>,
}

/// TOML `[particles]` — contains a list of insertion blocks.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ParticlesConfig {
    pub insert: Option<Vec<InsertConfig>>,
}

/// Inserts DEM particles at setup time based on `[[particles.insert]]` config blocks.
pub struct DemAtomInsertPlugin;

impl Plugin for DemAtomInsertPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Particle insertion blocks (one per material/group)
[[particles.insert]]
material = "glass"          # must match a [[dem.materials]] name
count = 100
radius = 0.001
density = 2500.0
# velocity = 0.1            # random velocity magnitude (Gaussian)
# velocity_x = 0.0          # directional velocity (additive with random)
# velocity_y = 0.0
# velocity_z = 0.0
# region = { type = "block", min = [0.0, 0.0, 0.0], max = [1.0, 1.0, 1.0] }  # defaults to domain bounds"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_setup_system(dem_insert_atoms, ScheduleSetupSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSetupSet::PostSetup);
    }
}

pub fn dem_insert_atoms(
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    stage_overrides: Res<StageOverrides>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let index = scheduler_manager.index;

    // Determine if this stage should insert particles:
    // - First stage: use top-level [particles] (backward compat) or stage overrides
    // - Later stages: only if the stage's [[run]] block explicitly has particles
    let has_stage_particles = index < run_config.num_stages()
        && run_config.current_stage(index).overrides.contains_key("particles");

    let particles_config: ParticlesConfig = if has_stage_particles || index == 0 {
        Config::load_stage_aware(&stage_overrides, "particles")
    } else {
        ParticlesConfig::default()
    };

    // Insert particles per insert block
    if let Some(ref inserts) = particles_config.insert {
        if comm.rank() == 0 {
            let mut dem_data = registry.expect_mut::<DemAtom>("dem_insert_atoms");
            let mut rng = rand::rng();
            let mut max_tag = atom.get_max_tag();

            for insert in inserts {
                // Find material config by name
                let mat_idx = match material_table.find_material(&insert.material) {
                    Some(idx) => idx,
                    None => {
                        eprintln!(
                            "ERROR: Unknown material '{}' in [[particles.insert]]. Available: {:?}",
                            insert.material, material_table.names
                        );
                        std::process::exit(1);
                    }
                };

                let radius = insert.radius;
                let density = insert.density;

                println!("DemAtomInsert: inserting {} particles of material '{}' (r={}, rho={}, E={}, nu={})",
                    insert.count, insert.material, radius, density,
                    material_table.youngs_mod[mat_idx as usize],
                    material_table.poisson_ratio[mat_idx as usize]);

                // Use explicit region or default to domain bounds inset by radius.
                let region = insert.region.clone().unwrap_or_else(|| Region::Block {
                    min: [
                        domain.boundaries_low[0] + radius,
                        domain.boundaries_low[1] + radius,
                        domain.boundaries_low[2] + radius,
                    ],
                    max: [
                        domain.boundaries_high[0] - radius,
                        domain.boundaries_high[1] - radius,
                        domain.boundaries_high[2] - radius,
                    ],
                });

                let mut count = 0u32;
                while count < insert.count {
                    let [x, y, z] = region.random_point_inside(&mut rng);

                    let mut no_overlap = true;
                    for i in 0..atom.len() {
                        let dx = x - atom.pos[i][0];
                        let dy = y - atom.pos[i][1];
                        let dz = z - atom.pos[i][2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                        if distance <= (radius + dem_data.radius[i]) * 1.1 {
                            no_overlap = false;
                            break;
                        }
                    }

                    if no_overlap {
                        count += 1;
                        atom.natoms += 1;
                        atom.nlocal += 1;
                        atom.tag.push(max_tag);
                        atom.origin_index.push(0);
                        atom.cutoff_radius.push(radius);
                        atom.is_ghost.push(false);
                                                max_tag += 1;
                        atom.pos.push([x, y, z]);
                        atom.vel.push([0.0; 3]);
                        atom.force.push([0.0; 3]);
                        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
                        atom.mass.push(mass);
                        atom.inv_mass.push(1.0 / mass);

                        atom.atom_type.push(mat_idx);
                        dem_data.radius.push(radius);
                        dem_data.density.push(density);
                        dem_data.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
                        dem_data.quaternion.push([1.0, 0.0, 0.0, 0.0]);
                        dem_data.omega.push([0.0; 3]);
                        dem_data.ang_mom.push([0.0; 3]);
                        dem_data.torque.push([0.0; 3]);
                    }
                }

                // Apply per-insert velocity to this batch
                let total_len = atom.vel.len();
                let start = total_len - insert.count as usize;
                if let Some(rand_vel) = insert.velocity {
                    if rand_vel < 0.0 {
                        eprintln!("ERROR: velocity in [[particles.insert]] must be non-negative, got {}", rand_vel);
                        std::process::exit(1);
                    }
                    let normal = Normal::new(0.0, rand_vel).unwrap();
                    for i in start..total_len {
                        atom.vel[i][0] = normal.sample(&mut rng);
                        atom.vel[i][1] = normal.sample(&mut rng);
                        atom.vel[i][2] = normal.sample(&mut rng);
                    }
                }
                // Apply directional velocity components (additive with random)
                let vx = insert.velocity_x.unwrap_or(0.0);
                let vy = insert.velocity_y.unwrap_or(0.0);
                let vz = insert.velocity_z.unwrap_or(0.0);
                if vx != 0.0 || vy != 0.0 || vz != 0.0 {
                    for i in start..total_len {
                        atom.vel[i][0] += vx;
                        atom.vel[i][1] += vy;
                        atom.vel[i][2] += vz;
                    }
                }
            }
        }
    }
}

fn calculate_delta_time(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let dem = registry.expect::<DemAtom>("calculate_delta_time");
    let mut dt: f64 = 0.001;

    for i in 0..dem.radius.len() {
        let mat_idx = atoms.atom_type[i] as usize;
        let youngs_mod = material_table.youngs_mod[mat_idx];
        let poisson_ratio = material_table.poisson_ratio[mat_idx];
        let g = youngs_mod / (2.0 * (1.0 + poisson_ratio));
        let alpha = 0.1631 * poisson_ratio + 0.876605;
        let delta = PI * dem.radius[i] / alpha * (dem.density[i] / g).sqrt();
        dt = delta.min(dt);
    }

    dt = comm.all_reduce_min_f64(dt);

    if comm.rank() == 0 {
        println!("Using {} for delta time", dt * 0.15);
    }
    atoms.dt = dt * 0.15;
}
