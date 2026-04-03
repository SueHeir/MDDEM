//! Particle insertion: random, rate-based, and file-based from `[[particles.insert]]` config.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use mddem_core::{
    Atom, AtomDataRegistry, CommResource, Config, Domain, Region, RunConfig, RunState,
    ParticleSimScheduleSet, ScheduleSetupSet, StageOverrides,
};

use crate::{DemAtom, MaterialTable, RadiusSpec};

// ── Particle insertion ─────────────────────────────────────────────────────

fn default_source() -> String {
    "random".to_string()
}

/// Column index mapping for CSV file-based insertion.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ColumnMapping {
    #[serde(default)]
    pub x: Option<usize>,
    #[serde(default)]
    pub y: Option<usize>,
    #[serde(default)]
    pub z: Option<usize>,
    #[serde(default)]
    pub radius: Option<usize>,
    #[serde(default)]
    pub vx: Option<usize>,
    #[serde(default)]
    pub vy: Option<usize>,
    #[serde(default)]
    pub vz: Option<usize>,
    #[serde(default)]
    pub atom_type: Option<usize>,
}

impl Default for ColumnMapping {
    fn default() -> Self {
        ColumnMapping {
            x: Some(0),
            y: Some(1),
            z: Some(2),
            radius: Some(3),
            vx: None,
            vy: None,
            vz: None,
            atom_type: None,
        }
    }
}

#[derive(Deserialize, Clone)]
/// A single particle insertion block from `[[particles.insert]]`.
///
/// Three modes determined by config fields:
/// - **Random** (default): `source = "random"` (or omitted), requires `material`, `count`, `radius`, `density`
/// - **Rate-based**: random insertion with `rate` field present — registers for periodic insertion
/// - **File-based**: `source = "file"`, requires `file` and `format`
pub struct InsertConfig {
    /// Insertion source: `"random"` (default) or `"file"`.
    #[serde(default = "default_source")]
    pub source: String,
    /// Material name (must match a `[[dem.materials]]` entry). Required for random/rate modes.
    pub material: Option<String>,
    /// Number of particles to insert at setup time. Required for random mode without `rate`.
    pub count: Option<u32>,
    /// Particle radius: fixed value or distribution. Required for random/rate modes.
    pub radius: Option<RadiusSpec>,
    /// Particle density (kg/m³). Required for random/rate modes.
    pub density: Option<f64>,
    /// Random velocity magnitude (Gaussian distribution).
    pub velocity: Option<f64>,
    /// Directional velocity components (additive with random velocity).
    pub velocity_x: Option<f64>,
    pub velocity_y: Option<f64>,
    pub velocity_z: Option<f64>,
    /// Insertion region. Defaults to domain bounds (inset by particle radius).
    #[serde(default)]
    pub region: Option<Region>,
    // ── Rate-based insertion fields ──
    /// Particles to insert per interval. Presence of this field triggers rate mode.
    pub rate: Option<u32>,
    /// Insert every N timesteps (default: 1).
    pub rate_interval: Option<usize>,
    /// First timestep to begin insertion (default: 0).
    pub rate_start: Option<usize>,
    /// Last timestep for insertion (optional, no default = run forever).
    pub rate_end: Option<usize>,
    /// Maximum total particles to insert (optional).
    pub rate_limit: Option<u32>,
    // ── File-based insertion fields ──
    /// Path to particle data file.
    pub file: Option<String>,
    /// File format: `"csv"` or `"lammps_dump"`.
    pub format: Option<String>,
    /// Column index mapping for CSV files.
    pub columns: Option<ColumnMapping>,
    /// Explicit mapping from integer atom types (in file) to named materials.
    /// e.g. `type_map = { 1 = "glass", 2 = "steel" }`.
    /// When present, overrides implicit type-to-material mapping.
    /// Keys are strings in TOML but parsed as u32 integers.
    #[serde(default)]
    pub type_map: Option<HashMap<String, String>>,
    /// LAMMPS data file atom style: `"atomic"`, `"sphere"`, `"bpm/sphere"`.
    /// Auto-detected from `Atoms # style` header if not specified.
    pub atom_style: Option<String>,
}

/// TOML `[particles]` — contains a list of insertion blocks.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ParticlesConfig {
    pub insert: Option<Vec<InsertConfig>>,
}

// ── Rate-based insertion state ──────────────────────────────────────────────

/// Tracks a single rate-based insertion configuration and its progress.
pub struct RateInsertEntry {
    pub config: InsertConfig,
    pub mat_idx: u32,
    pub total_inserted: u32,
}

/// Resource holding all active rate-based insertion entries.
pub struct RateInsertState {
    pub entries: Vec<RateInsertEntry>,
}

impl Default for RateInsertState {
    fn default() -> Self {
        RateInsertState {
            entries: Vec::new(),
        }
    }
}

// ── SpatialHash for O(1) overlap checking ───────────────────────────────────

/// Grid-based spatial hash for fast overlap detection during particle insertion.
///
/// Divides space into cubic cells of size `cell_size` (typically ~2× max particle diameter).
/// Overlap queries check the 3×3×3 neighborhood of the candidate cell, ensuring all
/// potential overlaps are found without a full O(N²) scan.
/// Periodic boundary info for minimum-image overlap checks during insertion.
struct PeriodicBox {
    is_periodic: [bool; 3],
    box_size: [f64; 3],
}

impl PeriodicBox {
    fn from_domain(domain: &Domain) -> Self {
        PeriodicBox {
            is_periodic: domain.periodic_flags(),
            box_size: domain.size,
        }
    }

    /// Compute minimum-image squared distance between two positions.
    fn min_image_dist_sq(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let mut dist_sq = 0.0;
        for d in 0..3 {
            let mut delta = a[d] - b[d];
            if self.is_periodic[d] {
                let half = 0.5 * self.box_size[d];
                if delta > half {
                    delta -= self.box_size[d];
                } else if delta < -half {
                    delta += self.box_size[d];
                }
            }
            dist_sq += delta * delta;
        }
        dist_sq
    }
}

struct SpatialHash {
    cell_size: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHash {
    fn new(cell_size: f64) -> Self {
        SpatialHash {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_key(&self, pos: &[f64; 3]) -> (i64, i64, i64) {
        (
            (pos[0] / self.cell_size).floor() as i64,
            (pos[1] / self.cell_size).floor() as i64,
            (pos[2] / self.cell_size).floor() as i64,
        )
    }

    fn insert(&mut self, idx: usize, pos: &[f64; 3]) {
        let key = self.cell_key(pos);
        self.cells.entry(key).or_default().push(idx);
    }

    fn has_overlap(
        &self,
        pos: &[f64; 3],
        radius: f64,
        positions: &[[f64; 3]],
        radii: &[f64],
        pbc: &PeriodicBox,
    ) -> bool {
        // Collect all cell keys to check: 3x3x3 neighborhood plus periodic images
        let key = self.cell_key(pos);
        let min_dist_check = radius * 2.2; // conservative check radius

        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    let neighbor_key = (key.0 + di, key.1 + dj, key.2 + dk);
                    if let Some(indices) = self.cells.get(&neighbor_key) {
                        for &idx in indices {
                            let dist_sq = pbc.min_image_dist_sq(pos, &positions[idx]);
                            let min_dist = (radius + radii[idx]) * 1.1;
                            if dist_sq <= min_dist * min_dist {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // For periodic axes where the box is small (< 3 cell sizes), the standard
        // 3x3x3 neighborhood may miss periodic images. Do a brute-force check
        // against all atoms using minimum-image distances.
        let needs_pbc_check = (0..3).any(|d| {
            pbc.is_periodic[d] && pbc.box_size[d] < 3.0 * self.cell_size + min_dist_check
        });
        if needs_pbc_check {
            for idx in 0..positions.len() {
                let dist_sq = pbc.min_image_dist_sq(pos, &positions[idx]);
                let min_dist = (radius + radii[idx]) * 1.1;
                if dist_sq <= min_dist * min_dist {
                    return true;
                }
            }
        }

        false
    }
}

// ── DemAtomInsertPlugin ─────────────────────────────────────────────────────

/// Inserts DEM particles at setup time and registers rate-based insertion for runtime.
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
# region = { type = "block", min = [0.0, 0.0, 0.0], max = [1.0, 1.0, 1.0] }  # defaults to domain bounds
#
# Size distributions (instead of fixed radius):
# radius = { distribution = "uniform", min = 0.0008, max = 0.0012 }
# radius = { distribution = "gaussian", mean = 0.001, std = 0.0001 }
# radius = { distribution = "lognormal", mean = 0.001, std = 0.0001 }
# radius = { distribution = "discrete", values = [0.001, 0.0015], weights = [0.7, 0.3] }
#
# Rate-based insertion (insert particles over time):
# rate = 10              # particles per interval
# rate_interval = 100    # insert every N timesteps
# rate_start = 0         # first timestep (default 0)
# rate_end = 500000      # last timestep (optional)
# rate_limit = 5000      # total max particles (optional)
#
# File-based insertion:
# source = "file"
# file = "particles.csv"
# format = "csv"
# material = "glass"
# density = 2500.0
# columns = { x = 0, y = 1, z = 2, radius = 3 }"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_resource(RateInsertState::default());
        app.add_setup_system(dem_insert_atoms, ScheduleSetupSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSetupSet::PostSetup)
            .add_update_system(dem_rate_insert, ParticleSimScheduleSet::PreInitialIntegration);
    }
}

// ── Helper: insert a single particle ────────────────────────────────────────

/// Appends a single DEM particle to both the shared `Atom` arrays and the `DemAtom` extension.
///
/// Computes mass from density and radius (solid sphere: m = ρ·4/3·π·r³), and inverse
/// moment of inertia (I = 2/5·m·r² for a solid sphere). Initializes quaternion to identity
/// and angular velocity/momentum/torque to zero.
fn insert_single_particle(
    atom: &mut Atom,
    dem_data: &mut DemAtom,
    pos: [f64; 3],
    vel: [f64; 3],
    radius: f64,
    density: f64,
    mat_idx: u32,
    tag: u32,
) {
    atom.natoms += 1;
    atom.nlocal += 1;
    atom.tag.push(tag);
    atom.origin_index.push(0);
    atom.cutoff_radius.push(radius);
    atom.image.push([0, 0, 0]);
    atom.is_ghost.push(false);
    atom.pos.push(pos);
    atom.vel.push(vel);
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
    dem_data.body_id.push(0.0);
}

// ── Helper: resolve material index ──────────────────────────────────────────

fn resolve_material(material_table: &MaterialTable, name: &str) -> u32 {
    match material_table.find_material(name) {
        Some(idx) => idx,
        None => {
            eprintln!(
                "ERROR: Unknown material '{}' in [[particles.insert]]. Available: {:?}",
                name, material_table.names
            );
            std::process::exit(1);
        }
    }
}

// ── Helper: resolve type_map to index map ────────────────────────────────────

/// Validates material names in `type_map` and builds a `HashMap<u32, u32>` mapping
/// file atom types to material indices. Called once per file load.
fn resolve_type_map(
    type_map: &HashMap<String, String>,
    material_table: &MaterialTable,
) -> HashMap<u32, u32> {
    let mut index_map = HashMap::new();
    for (key_str, mat_name) in type_map {
        let file_type: u32 = key_str.parse().unwrap_or_else(|_| {
            eprintln!(
                "ERROR: type_map key '{}' is not a valid integer atom type",
                key_str
            );
            std::process::exit(1);
        });
        let mat_idx = resolve_material(material_table, mat_name);
        index_map.insert(file_type, mat_idx);
    }
    index_map
}

/// Look up material index for a given file atom type.
/// Checks type_map first, then falls back to the default material index.
fn lookup_material_for_type(
    file_type: u32,
    type_index_map: Option<&HashMap<u32, u32>>,
    default_mat_idx: u32,
) -> u32 {
    if let Some(map) = type_index_map {
        if let Some(&idx) = map.get(&file_type) {
            return idx;
        }
    }
    default_mat_idx
}

// ── Setup system: dem_insert_atoms ──────────────────────────────────────────

/// Setup system that processes all `[[particles.insert]]` blocks at simulation start.
///
/// For each block: immediate random insertion places particles with overlap checking,
/// file-based insertion loads from CSV/LAMMPS files, and rate-based insertion registers
/// entries in [`RateInsertState`] for periodic insertion during the run.
pub fn dem_insert_atoms(
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    stage_overrides: Res<StageOverrides>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    mut rate_state: ResMut<RateInsertState>,
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
                if insert.source == "file" {
                    // ── File-based insertion ──
                    insert_from_file(
                        insert,
                        &mut atom,
                        &mut dem_data,
                        &material_table,
                        &mut max_tag,
                    );
                } else if insert.rate.is_some() {
                    // ── Rate-based: register for runtime insertion ──
                    let mat_name = insert.material.as_deref().unwrap_or_else(|| {
                        eprintln!("ERROR: Rate-based [[particles.insert]] requires 'material'");
                        std::process::exit(1);
                    });
                    let mat_idx = resolve_material(&material_table, mat_name);
                    if insert.radius.is_none() {
                        eprintln!("ERROR: Rate-based [[particles.insert]] requires 'radius'");
                        std::process::exit(1);
                    }
                    if insert.density.is_none() {
                        eprintln!("ERROR: Rate-based [[particles.insert]] requires 'density'");
                        std::process::exit(1);
                    }
                    println!(
                        "DemAtomInsert: registering rate-based insertion for material '{}' (rate={}/every {})",
                        mat_name,
                        insert.rate.expect("rate already validated above"),
                        insert.rate_interval.unwrap_or(1),
                    );
                    rate_state.entries.push(RateInsertEntry {
                        config: insert.clone(),
                        mat_idx,
                        total_inserted: 0,
                    });
                } else {
                    // ── Immediate random insertion ──
                    let mat_name = insert.material.as_deref().unwrap_or_else(|| {
                        eprintln!(
                            "ERROR: [[particles.insert]] requires 'material' for random insertion"
                        );
                        std::process::exit(1);
                    });
                    let mat_idx = resolve_material(&material_table, mat_name);
                    let count = insert.count.unwrap_or_else(|| {
                        eprintln!("ERROR: [[particles.insert]] requires 'count' for random insertion (without rate)");
                        std::process::exit(1);
                    });
                    let radius_spec = insert.radius.as_ref().unwrap_or_else(|| {
                        eprintln!(
                            "ERROR: [[particles.insert]] requires 'radius' for random insertion"
                        );
                        std::process::exit(1);
                    });
                    let density = insert.density.unwrap_or_else(|| {
                        eprintln!(
                            "ERROR: [[particles.insert]] requires 'density' for random insertion"
                        );
                        std::process::exit(1);
                    });

                    let max_r = radius_spec.max_radius();
                    println!(
                        "DemAtomInsert: inserting {} particles of material '{}' (r={}, rho={}, E={}, nu={})",
                        count,
                        mat_name,
                        max_r,
                        density,
                        material_table.youngs_mod[mat_idx as usize],
                        material_table.poisson_ratio[mat_idx as usize]
                    );

                    // Use explicit region or default to domain bounds inset by max radius.
                    let region = insert.region.clone().unwrap_or_else(|| Region::Block {
                        min: [
                            domain.boundaries_low[0] + max_r,
                            domain.boundaries_low[1] + max_r,
                            domain.boundaries_low[2] + max_r,
                        ],
                        max: [
                            domain.boundaries_high[0] - max_r,
                            domain.boundaries_high[1] - max_r,
                            domain.boundaries_high[2] - max_r,
                        ],
                    });

                    let pbc = PeriodicBox::from_domain(&domain);
                    let start_idx = atom.len();
                    let mut inserted = 0u32;
                    let mut attempts = 0u64;
                    let max_attempts = count as u64 * 1_000_000;
                    while inserted < count && attempts < max_attempts {
                        attempts += 1;
                        let [x, y, z] = region.random_point_inside(&mut rng);
                        let radius = radius_spec.sample(&mut rng);

                        let mut no_overlap = true;
                        let candidate = [x, y, z];
                        for i in 0..atom.len() {
                            let dist_sq = pbc.min_image_dist_sq(&candidate, &atom.pos[i]);
                            let min_dist = (radius + dem_data.radius[i]) * 1.1;
                            if dist_sq <= min_dist * min_dist {
                                no_overlap = false;
                                break;
                            }
                        }

                        if no_overlap {
                            insert_single_particle(
                                &mut atom,
                                &mut dem_data,
                                [x, y, z],
                                [0.0; 3],
                                radius,
                                density,
                                mat_idx,
                                max_tag,
                            );
                            max_tag += 1;
                            inserted += 1;
                        }
                    }
                    if inserted < count {
                        eprintln!(
                            "WARNING: Could only insert {}/{} particles after {} attempts. \
                             Increase domain size or reduce particle count.",
                            inserted, count, max_attempts
                        );
                    }

                    // Apply per-insert velocity to this batch
                    let total_len = atom.vel.len();
                    let start = start_idx;
                    if let Some(rand_vel) = insert.velocity {
                        if rand_vel < 0.0 {
                            eprintln!(
                                "ERROR: velocity in [[particles.insert]] must be non-negative, got {}",
                                rand_vel
                            );
                            std::process::exit(1);
                        }
                        let normal = Normal::new(0.0, rand_vel)
                            .expect("velocity must be non-negative for Normal distribution");
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
}

// ── File-based insertion ────────────────────────────────────────────────────

fn insert_from_file(
    insert: &InsertConfig,
    atom: &mut Atom,
    dem_data: &mut DemAtom,
    material_table: &MaterialTable,
    max_tag: &mut u32,
) {
    let file_path = insert.file.as_deref().unwrap_or_else(|| {
        eprintln!("ERROR: source = \"file\" requires 'file' field in [[particles.insert]]");
        std::process::exit(1);
    });
    let format = insert.format.as_deref().unwrap_or_else(|| {
        eprintln!("ERROR: source = \"file\" requires 'format' field in [[particles.insert]]");
        std::process::exit(1);
    });

    match format {
        "csv" => read_csv_particles(insert, file_path, atom, dem_data, material_table, max_tag),
        "lammps_dump" => {
            read_lammps_dump_particles(insert, file_path, atom, dem_data, material_table, max_tag)
        }
        "lammps_data" => {
            read_lammps_data_particles(insert, file_path, atom, dem_data, material_table, max_tag)
        }
        other => {
            eprintln!(
                "ERROR: Unknown file format '{}' in [[particles.insert]]. Supported: csv, lammps_dump, lammps_data",
                other
            );
            std::process::exit(1);
        }
    }
}

fn read_csv_particles(
    insert: &InsertConfig,
    file_path: &str,
    atom: &mut Atom,
    dem_data: &mut DemAtom,
    material_table: &MaterialTable,
    max_tag: &mut u32,
) {
    let mat_name = insert.material.as_deref().unwrap_or_else(|| {
        eprintln!("ERROR: CSV source = \"file\" requires 'material' in [[particles.insert]]");
        std::process::exit(1);
    });
    let mat_idx = resolve_material(material_table, mat_name);

    let type_index_map = insert
        .type_map
        .as_ref()
        .map(|tm| resolve_type_map(tm, material_table));

    let density = insert.density.unwrap_or_else(|| {
        eprintln!("ERROR: CSV source = \"file\" requires 'density' in [[particles.insert]]");
        std::process::exit(1);
    });

    let cols = insert.columns.clone().unwrap_or_default();
    let col_x = cols.x.unwrap_or(0);
    let col_y = cols.y.unwrap_or(1);
    let col_z = cols.z.unwrap_or(2);
    let col_radius = cols.radius;
    let col_vx = cols.vx;
    let col_vy = cols.vy;
    let col_vz = cols.vz;
    let col_atom_type = cols.atom_type;

    let default_radius = match &insert.radius {
        Some(RadiusSpec::Fixed(r)) => Some(*r),
        _ => None,
    };

    let file = File::open(file_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open CSV file '{}': {}", file_path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut count = 0u32;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            eprintln!(
                "ERROR: Failed to read line {} of '{}': {}",
                line_num + 1,
                file_path,
                e
            );
            std::process::exit(1);
        });
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Skip header line if it starts with a letter
        if line_num == 0 && trimmed.chars().next().map_or(false, |c| c.is_alphabetic()) {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        let parse = |idx: usize, name: &str| -> f64 {
            fields.get(idx).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!(
                    "ERROR: Failed to parse {} (column {}) at line {} of '{}'",
                    name,
                    idx,
                    line_num + 1,
                    file_path
                );
                std::process::exit(1);
            })
        };

        let x = parse(col_x, "x");
        let y = parse(col_y, "y");
        let z = parse(col_z, "z");
        let radius = col_radius
            .map(|c| parse(c, "radius"))
            .or(default_radius)
            .unwrap_or_else(|| {
                eprintln!(
                    "ERROR: No radius column or default radius at line {} of '{}'",
                    line_num + 1,
                    file_path
                );
                std::process::exit(1);
            });
        let vx = col_vx.map(|c| parse(c, "vx")).unwrap_or(0.0);
        let vy = col_vy.map(|c| parse(c, "vy")).unwrap_or(0.0);
        let vz = col_vz.map(|c| parse(c, "vz")).unwrap_or(0.0);

        // Determine material: type_map lookup (if atom_type column present) → default material
        let row_mat_idx = match col_atom_type {
            Some(col) => {
                let file_type = parse(col, "atom_type") as u32;
                lookup_material_for_type(file_type, type_index_map.as_ref(), mat_idx)
            }
            None => mat_idx,
        };

        insert_single_particle(
            atom,
            dem_data,
            [x, y, z],
            [vx, vy, vz],
            radius,
            density,
            row_mat_idx,
            *max_tag,
        );
        *max_tag += 1;
        count += 1;
    }

    println!(
        "DemAtomInsert: loaded {} particles from CSV '{}'",
        count, file_path
    );
}

fn read_lammps_dump_particles(
    insert: &InsertConfig,
    file_path: &str,
    atom: &mut Atom,
    dem_data: &mut DemAtom,
    material_table: &MaterialTable,
    max_tag: &mut u32,
) {
    let mat_name = insert.material.as_deref().unwrap_or_else(|| {
        eprintln!(
            "ERROR: lammps_dump source = \"file\" requires 'material' in [[particles.insert]]"
        );
        std::process::exit(1);
    });
    let mat_idx = resolve_material(material_table, mat_name);

    let type_index_map = insert
        .type_map
        .as_ref()
        .map(|tm| resolve_type_map(tm, material_table));

    let density = insert.density.unwrap_or_else(|| {
        eprintln!(
            "ERROR: lammps_dump source = \"file\" requires 'density' in [[particles.insert]]"
        );
        std::process::exit(1);
    });

    let default_radius = match &insert.radius {
        Some(RadiusSpec::Fixed(r)) => Some(*r),
        _ => None,
    };

    let file = File::open(file_path).unwrap_or_else(|e| {
        eprintln!(
            "ERROR: Failed to open LAMMPS dump file '{}': {}",
            file_path, e
        );
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse LAMMPS dump format
    let mut n_atoms: usize = 0;
    let mut column_names: Vec<String> = Vec::new();
    let mut reading_atoms = false;
    let mut count = 0u32;

    // Helper to find column index by name
    let find_col = |names: &[String], name: &str| -> Option<usize> {
        names.iter().position(|n| n == name)
    };

    while let Some(Ok(line)) = lines.next() {
        let trimmed = line.trim();

        if trimmed == "ITEM: NUMBER OF ATOMS" {
            if let Some(Ok(next)) = lines.next() {
                n_atoms = next.trim().parse().unwrap_or(0);
            }
            continue;
        }

        if trimmed.starts_with("ITEM: ATOMS") {
            // Parse column names from header: "ITEM: ATOMS id type x y z ..."
            column_names = trimmed
                .strip_prefix("ITEM: ATOMS")
                .unwrap_or("")
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            reading_atoms = true;
            continue;
        }

        if trimmed.starts_with("ITEM:") {
            reading_atoms = false;
            continue;
        }

        if reading_atoms && !trimmed.is_empty() {
            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.len() < column_names.len() {
                continue;
            }

            let parse_col = |name: &str| -> Option<f64> {
                find_col(&column_names, name).and_then(|i| fields.get(i)?.parse().ok())
            };

            let x = parse_col("x").unwrap_or(0.0);
            let y = parse_col("y").unwrap_or(0.0);
            let z = parse_col("z").unwrap_or(0.0);
            let vx = parse_col("vx").unwrap_or(0.0);
            let vy = parse_col("vy").unwrap_or(0.0);
            let vz = parse_col("vz").unwrap_or(0.0);
            let radius = parse_col("radius")
                .or(default_radius)
                .unwrap_or_else(|| {
                    eprintln!(
                        "ERROR: No 'radius' column in LAMMPS dump and no default radius in config"
                    );
                    std::process::exit(1);
                });

            // Determine material: type_map override → default material
            let row_mat_idx = match parse_col("type") {
                Some(t) => lookup_material_for_type(t as u32, type_index_map.as_ref(), mat_idx),
                None => mat_idx,
            };

            insert_single_particle(
                atom,
                dem_data,
                [x, y, z],
                [vx, vy, vz],
                radius,
                density,
                row_mat_idx,
                *max_tag,
            );
            *max_tag += 1;
            count += 1;
        }
    }

    let _ = n_atoms; // used for format validation if needed
    println!(
        "DemAtomInsert: loaded {} particles from LAMMPS dump '{}'",
        count, file_path
    );
}

/// Parse a field from a LAMMPS data file, with a user-friendly error on failure.
fn parse_field<T: std::str::FromStr>(value: &str, field_name: &str, line_num: usize, file_path: &str) -> T
where
    T::Err: std::fmt::Display,
{
    value.parse::<T>().unwrap_or_else(|e| {
        eprintln!(
            "ERROR: Failed to parse {} '{}' at line {} of '{}': {}",
            field_name, value, line_num, file_path, e
        );
        std::process::exit(1);
    })
}

fn read_lammps_data_particles(
    insert: &InsertConfig,
    file_path: &str,
    atom: &mut Atom,
    dem_data: &mut DemAtom,
    material_table: &MaterialTable,
    max_tag: &mut u32,
) {
    let mat_name = insert.material.as_deref().unwrap_or_else(|| {
        eprintln!(
            "ERROR: lammps_data source = \"file\" requires 'material' in [[particles.insert]]"
        );
        std::process::exit(1);
    });
    let mat_idx = resolve_material(material_table, mat_name);

    let type_index_map = insert
        .type_map
        .as_ref()
        .map(|tm| resolve_type_map(tm, material_table));

    let default_density = insert.density;
    let default_radius = match &insert.radius {
        Some(RadiusSpec::Fixed(r)) => Some(*r),
        _ => None,
    };

    let file = File::open(file_path).unwrap_or_else(|e| {
        eprintln!(
            "ERROR: Failed to open LAMMPS data file '{}': {}",
            file_path, e
        );
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .enumerate()
        .map(|(i, l)| {
            l.unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to read line {} of '{}': {}", i + 1, file_path, e);
                std::process::exit(1);
            })
        })
        .collect();

    // Detect atom style from config or from "Atoms # style" header
    let config_style = insert.atom_style.as_deref();

    // Find section start indices
    let mut atoms_start = None;
    let mut atoms_style = None;
    let mut velocities_start = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("Atoms") {
            atoms_start = Some(i + 1);
            // Try to detect style from "Atoms # style" comment
            if let Some(comment) = trimmed.strip_prefix("Atoms") {
                let comment = comment.trim();
                if let Some(style) = comment.strip_prefix('#') {
                    let style = style.trim();
                    if !style.is_empty() {
                        atoms_style = Some(style.to_string());
                    }
                }
            }
        } else if trimmed == "Velocities" {
            velocities_start = Some(i + 1);
        }
    }

    let atom_style = config_style
        .map(|s| s.to_string())
        .or(atoms_style)
        .unwrap_or_else(|| "atomic".to_string());

    let atoms_start = atoms_start.unwrap_or_else(|| {
        eprintln!(
            "ERROR: No 'Atoms' section found in LAMMPS data file '{}'",
            file_path
        );
        std::process::exit(1);
    });

    // Parse Atoms section
    struct ParsedAtom {
        id: u32,
        atom_type: u32,
        pos: [f64; 3],
        radius: f64,
        density: f64,
    }

    let section_headers = [
        "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Masses", "Pair Coeffs",
    ];
    let is_section_header = |line: &str| -> bool {
        let trimmed = line.trim();
        section_headers.iter().any(|h| trimmed.starts_with(h))
    };

    let mut parsed_atoms: Vec<ParsedAtom> = Vec::new();

    for i in atoms_start..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_section_header(trimmed) {
            break;
        }
        // Skip comment lines
        if trimmed.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();

        match atom_style.as_str() {
            "atomic" => {
                // id type x y z
                if fields.len() < 5 {
                    eprintln!(
                        "ERROR: Expected at least 5 columns for atomic style at line {} of '{}'",
                        i + 1,
                        file_path
                    );
                    std::process::exit(1);
                }
                let id: u32 = parse_field(fields[0], "atom id", i + 1, file_path);
                let atype: u32 = parse_field(fields[1], "atom type", i + 1, file_path);
                let x: f64 = parse_field(fields[2], "x coordinate", i + 1, file_path);
                let y: f64 = parse_field(fields[3], "y coordinate", i + 1, file_path);
                let z: f64 = parse_field(fields[4], "z coordinate", i + 1, file_path);
                let radius = default_radius.unwrap_or_else(|| {
                    eprintln!("ERROR: 'radius' required in config for atomic style LAMMPS data");
                    std::process::exit(1);
                });
                let density = default_density.unwrap_or_else(|| {
                    eprintln!("ERROR: 'density' required in config for atomic style LAMMPS data");
                    std::process::exit(1);
                });
                parsed_atoms.push(ParsedAtom {
                    id,
                    atom_type: atype,
                    pos: [x, y, z],
                    radius,
                    density,
                });
            }
            "sphere" | "bpm/sphere" => {
                // id type diameter density x y z
                if fields.len() < 7 {
                    eprintln!(
                        "ERROR: Expected at least 7 columns for {} style at line {} of '{}'",
                        atom_style,
                        i + 1,
                        file_path
                    );
                    std::process::exit(1);
                }
                let id: u32 = parse_field(fields[0], "atom id", i + 1, file_path);
                let atype: u32 = parse_field(fields[1], "atom type", i + 1, file_path);
                let diameter: f64 = parse_field(fields[2], "diameter", i + 1, file_path);
                let density: f64 = parse_field(fields[3], "density", i + 1, file_path);
                let x: f64 = parse_field(fields[4], "x coordinate", i + 1, file_path);
                let y: f64 = parse_field(fields[5], "y coordinate", i + 1, file_path);
                let z: f64 = parse_field(fields[6], "z coordinate", i + 1, file_path);
                parsed_atoms.push(ParsedAtom {
                    id,
                    atom_type: atype,
                    pos: [x, y, z],
                    radius: diameter / 2.0,
                    density,
                });
            }
            other => {
                eprintln!(
                    "ERROR: Unsupported atom_style '{}' in LAMMPS data file. Supported: atomic, sphere, bpm/sphere",
                    other
                );
                std::process::exit(1);
            }
        }
    }

    // Parse Velocities section (optional) — build id → [vx, vy, vz] map
    let mut velocity_map: HashMap<u32, [f64; 3]> = HashMap::new();
    if let Some(vel_start) = velocities_start {
        for i in vel_start..lines.len() {
            let trimmed = lines[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            if is_section_header(trimmed) {
                break;
            }
            if trimmed.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.len() >= 4 {
                let id: u32 = parse_field(fields[0], "atom id (Velocities)", i + 1, file_path);
                let vx: f64 = parse_field(fields[1], "vx", i + 1, file_path);
                let vy: f64 = parse_field(fields[2], "vy", i + 1, file_path);
                let vz: f64 = parse_field(fields[3], "vz", i + 1, file_path);
                velocity_map.insert(id, [vx, vy, vz]);
            }
        }
    }

    // Insert all parsed atoms
    let count = parsed_atoms.len();
    for pa in parsed_atoms {
        let vel = velocity_map.get(&pa.id).copied().unwrap_or([0.0; 3]);
        let row_mat_idx = lookup_material_for_type(pa.atom_type, type_index_map.as_ref(), mat_idx);
        insert_single_particle(
            atom,
            dem_data,
            pa.pos,
            vel,
            pa.radius,
            pa.density,
            row_mat_idx,
            *max_tag,
        );
        *max_tag += 1;
    }

    println!(
        "DemAtomInsert: loaded {} particles from LAMMPS data file '{}' (style: {})",
        count, file_path, atom_style
    );
}

// ── Update system: rate-based insertion ─────────────────────────────────────

/// Update system for rate-based particle insertion during the simulation run.
///
/// Checks each registered [`RateInsertEntry`] against the current timestep, interval,
/// start/end bounds, and total limit. Uses a [`SpatialHash`] for O(1) overlap detection
/// when placing new particles. Runs in `ParticleSimScheduleSet::PreInitialIntegration`.
#[allow(clippy::too_many_arguments)]
pub fn dem_rate_insert(
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    mut rate_state: ResMut<RateInsertState>,
) {
    if rate_state.entries.is_empty() || comm.rank() != 0 {
        return;
    }

    let step = run_state.total_cycle;
    let mut any_to_insert = false;

    // Quick check if any entry needs insertion this step (before stripping ghosts)

    // Quick check if any entry needs insertion this step
    for entry in rate_state.entries.iter() {
        let interval = entry.config.rate_interval.unwrap_or(1);
        let start = entry.config.rate_start.unwrap_or(0);
        if step < start {
            continue;
        }
        if let Some(end) = entry.config.rate_end {
            if step > end {
                continue;
            }
        }
        if let Some(limit) = entry.config.rate_limit {
            if entry.total_inserted >= limit {
                continue;
            }
        }
        let steps_since_start = step - start;
        if interval == 0 || steps_since_start % interval == 0 {
            any_to_insert = true;
            break;
        }
    }

    if !any_to_insert {
        return;
    }

    // Strip ghost atoms before inserting new local atoms.
    // New atoms are appended at atom.len(), which must equal nlocal so that
    // the subsequent borders() truncate_to_nlocal() doesn't discard them.
    if atom.nghost > 0 {
        atom.truncate_to_nlocal();
        registry.truncate_all(atom.nlocal as usize);
        atom.nghost = 0;
    }

    let mut dem_data = registry.expect_mut::<DemAtom>("dem_rate_insert");
    let mut rng = rand::rng();
    let mut next_tag = if atom.tag.is_empty() {
        0
    } else {
        atom.get_max_tag() + 1
    };

    for entry_idx in 0..rate_state.entries.len() {
        let interval = rate_state.entries[entry_idx]
            .config
            .rate_interval
            .unwrap_or(1);
        let start = rate_state.entries[entry_idx]
            .config
            .rate_start
            .unwrap_or(0);
        let rate = rate_state.entries[entry_idx]
            .config
            .rate
            .expect("rate-based insertion entry must have 'rate' field");

        if step < start {
            continue;
        }
        if let Some(end) = rate_state.entries[entry_idx].config.rate_end {
            if step > end {
                continue;
            }
        }
        if let Some(limit) = rate_state.entries[entry_idx].config.rate_limit {
            if rate_state.entries[entry_idx].total_inserted >= limit {
                continue;
            }
        }
        let steps_since_start = step - start;
        if interval > 0 && steps_since_start % interval != 0 {
            continue;
        }

        // How many to insert this step
        let mut to_insert = rate;
        if let Some(limit) = rate_state.entries[entry_idx].config.rate_limit {
            let remaining = limit - rate_state.entries[entry_idx].total_inserted;
            to_insert = to_insert.min(remaining);
        }

        let radius_spec = rate_state.entries[entry_idx]
            .config
            .radius
            .as_ref()
            .expect("rate-based insertion entry must have 'radius' field");
        let density = rate_state.entries[entry_idx]
            .config
            .density
            .expect("rate-based insertion entry must have 'density' field");
        let mat_idx = rate_state.entries[entry_idx].mat_idx;

        let max_r = radius_spec.max_radius();
        let region = rate_state.entries[entry_idx]
            .config
            .region
            .clone()
            .unwrap_or_else(|| Region::Block {
                min: [
                    domain.boundaries_low[0] + max_r,
                    domain.boundaries_low[1] + max_r,
                    domain.boundaries_low[2] + max_r,
                ],
                max: [
                    domain.boundaries_high[0] - max_r,
                    domain.boundaries_high[1] - max_r,
                    domain.boundaries_high[2] - max_r,
                ],
            });

        // Build spatial hash from all existing atoms
        let cell_size = (2.0 * max_r * 1.1).max(1e-10);
        let pbc = PeriodicBox::from_domain(&domain);
        let mut spatial_hash = SpatialHash::new(cell_size);
        for i in 0..atom.len() {
            spatial_hash.insert(i, &atom.pos[i]);
        }

        let start_len = atom.len();
        let mut inserted = 0u32;
        let mut attempts = 0u32;
        let max_attempts = to_insert * 100;

        while inserted < to_insert && attempts < max_attempts {
            attempts += 1;
            let [x, y, z] = region.random_point_inside(&mut rng);
            let radius = radius_spec.sample(&mut rng);

            if !spatial_hash.has_overlap(
                &[x, y, z],
                radius,
                &atom.pos,
                &dem_data.radius,
                &pbc,
            ) {
                let new_idx = atom.len();
                insert_single_particle(
                    &mut atom,
                    &mut dem_data,
                    [x, y, z],
                    [0.0; 3],
                    radius,
                    density,
                    mat_idx,
                    next_tag,
                );
                spatial_hash.insert(new_idx, &atom.pos[new_idx]);
                next_tag += 1;
                inserted += 1;
            }
        }

        rate_state.entries[entry_idx].total_inserted += inserted;

        // Apply velocity to newly inserted particles
        let total_len = atom.vel.len();
        let config = &rate_state.entries[entry_idx].config;
        if let Some(rand_vel) = config.velocity {
            if rand_vel > 0.0 {
                let normal = Normal::new(0.0, rand_vel)
                    .expect("velocity must be non-negative for Normal distribution");
                for i in start_len..total_len {
                    atom.vel[i][0] = normal.sample(&mut rng);
                    atom.vel[i][1] = normal.sample(&mut rng);
                    atom.vel[i][2] = normal.sample(&mut rng);
                }
            }
        }
        let vx = config.velocity_x.unwrap_or(0.0);
        let vy = config.velocity_y.unwrap_or(0.0);
        let vz = config.velocity_z.unwrap_or(0.0);
        if vx != 0.0 || vy != 0.0 || vz != 0.0 {
            for i in start_len..total_len {
                atom.vel[i][0] += vx;
                atom.vel[i][1] += vy;
                atom.vel[i][2] += vz;
            }
        }

        if inserted > 0 {
            // Force full ghost rebuild since atom count changed
            atom.communicate_only = false;
        }
        if inserted > 0 && attempts >= max_attempts {
            eprintln!(
                "WARNING: Rate insertion at step {} only placed {}/{} particles (max attempts reached)",
                step, inserted, to_insert
            );
        }
    }
}

// ── Delta time calculation ──────────────────────────────────────────────────

/// Computes a stable timestep from the Rayleigh wave speed criterion.
///
/// For each particle, estimates the Rayleigh wave transit time across the particle
/// diameter using `dt_R = π·r / α · √(ρ/G)`, where α ≈ 0.1631·ν + 0.8766 and
/// G = E / (2·(1+ν)). The final timestep is 15% of the minimum across all particles.
fn calculate_delta_time(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    // If the current stage specifies an explicit dt, use it directly.
    let index = scheduler_manager.index;
    let config_dt = run_config.current_stage(index).dt;
    if config_dt > 0.0 {
        atoms.dt = config_dt;
        if comm.rank() == 0 {
            println!("Using {} for delta time (from config)", config_dt);
        }
        return;
    }

    // Auto-compute from Rayleigh wave speed criterion.
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RadiusDistribution;
    use mddem_core::toml;

    // ── SpatialHash tests ───────────────────────────────────────────────────

    fn no_pbc() -> PeriodicBox {
        PeriodicBox {
            is_periodic: [false; 3],
            box_size: [1.0; 3],
        }
    }

    #[test]
    fn spatial_hash_no_overlap() {
        let mut hash = SpatialHash::new(0.1);
        let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        let radii = vec![0.01, 0.01];
        for (i, pos) in positions.iter().enumerate() {
            hash.insert(i, pos);
        }
        // Far away — no overlap
        assert!(!hash.has_overlap(&[1.0, 1.0, 1.0], 0.01, &positions, &radii, &no_pbc()));
    }

    #[test]
    fn spatial_hash_detects_overlap() {
        let mut hash = SpatialHash::new(0.1);
        let positions = vec![[0.0, 0.0, 0.0]];
        let radii = vec![0.05];
        hash.insert(0, &positions[0]);
        // Close enough to overlap
        assert!(hash.has_overlap(&[0.05, 0.0, 0.0], 0.05, &positions, &radii, &no_pbc()));
    }

    #[test]
    fn spatial_hash_near_boundary() {
        let mut hash = SpatialHash::new(0.1);
        let positions = vec![[0.09, 0.0, 0.0]];
        let radii = vec![0.04];
        hash.insert(0, &positions[0]);
        // Just across cell boundary — should still detect overlap
        assert!(hash.has_overlap(&[0.11, 0.0, 0.0], 0.04, &positions, &radii, &no_pbc()));
    }

    #[test]
    fn spatial_hash_periodic_overlap() {
        // Particles near opposite edges of a periodic box should overlap
        let pbc = PeriodicBox {
            is_periodic: [false, true, false],
            box_size: [1.0, 0.1, 1.0],
        };
        let mut hash = SpatialHash::new(0.05);
        let positions = vec![[0.0, 0.005, 0.0]]; // near y=0 edge
        let radii = vec![0.02];
        hash.insert(0, &positions[0]);
        // Near y=0.095 edge — through PBC, distance is 0.01 < 2*0.02
        assert!(hash.has_overlap(&[0.0, 0.095, 0.0], 0.02, &positions, &radii, &pbc));
    }

    // ── InsertConfig deserialization tests ───────────────────────────────────

    #[test]
    fn insert_config_backward_compat() {
        let toml_str = r#"
material = "glass"
count = 100
radius = 0.001
density = 2500.0
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.material.as_deref(), Some("glass"));
        assert_eq!(config.count, Some(100));
        assert_eq!(config.density, Some(2500.0));
        assert!(matches!(config.radius, Some(RadiusSpec::Fixed(r)) if (r - 0.001).abs() < 1e-15));
        assert!(config.rate.is_none());
        assert_eq!(config.source, "random");
    }

    #[test]
    fn insert_config_with_distribution() {
        let toml_str = r#"
material = "glass"
count = 500
density = 2500.0
radius = { distribution = "uniform", min = 0.0008, max = 0.0012 }
velocity_z = -1.0
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert!(matches!(
            config.radius,
            Some(RadiusSpec::Distribution(RadiusDistribution::Uniform { .. }))
        ));
        assert_eq!(config.velocity_z, Some(-1.0));
    }

    #[test]
    fn insert_config_rate_based() {
        let toml_str = r#"
material = "glass"
density = 2500.0
radius = { distribution = "uniform", min = 0.0008, max = 0.0012 }
velocity_z = -1.0
rate = 10
rate_interval = 100
rate_start = 0
rate_end = 500000
rate_limit = 5000
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rate, Some(10));
        assert_eq!(config.rate_interval, Some(100));
        assert_eq!(config.rate_start, Some(0));
        assert_eq!(config.rate_end, Some(500000));
        assert_eq!(config.rate_limit, Some(5000));
    }

    #[test]
    fn insert_config_file_based_csv() {
        let toml_str = r#"
source = "file"
file = "particles.csv"
format = "csv"
material = "glass"
density = 2500.0
columns = { x = 0, y = 1, z = 2, radius = 3 }
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.source, "file");
        assert_eq!(config.file.as_deref(), Some("particles.csv"));
        assert_eq!(config.format.as_deref(), Some("csv"));
        let cols = config.columns.unwrap();
        assert_eq!(cols.x, Some(0));
        assert_eq!(cols.radius, Some(3));
    }

    #[test]
    fn insert_config_file_based_lammps() {
        let toml_str = r#"
source = "file"
file = "dump.lammpstrj"
format = "lammps_dump"
material = "glass"
density = 2500.0
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.source, "file");
        assert_eq!(config.format.as_deref(), Some("lammps_dump"));
    }

    #[test]
    fn insert_config_with_type_map() {
        let toml_str = r#"
source = "file"
file = "dump.lammpstrj"
format = "lammps_dump"
material = "glass"
density = 2500.0
type_map = { 1 = "glass", 2 = "steel" }
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        let tm = config.type_map.unwrap();
        assert_eq!(tm.len(), 2);
        assert_eq!(tm["1"], "glass");
        assert_eq!(tm["2"], "steel");
        assert_eq!(config.material.as_deref(), Some("glass"));
    }

    #[test]
    fn insert_config_type_map_with_fallback() {
        let toml_str = r#"
source = "file"
file = "particles.csv"
format = "csv"
density = 2500.0
material = "glass"
type_map = { 2 = "steel" }
columns = { x = 0, y = 1, z = 2, radius = 3, atom_type = 4 }
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.material.as_deref(), Some("glass"));
        let tm = config.type_map.unwrap();
        assert_eq!(tm.len(), 1);
        assert_eq!(tm["2"], "steel");
    }

    #[test]
    fn insert_config_no_type_map_backward_compat() {
        let toml_str = r#"
source = "file"
file = "dump.lammpstrj"
format = "lammps_dump"
material = "glass"
density = 2500.0
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert!(config.type_map.is_none());
    }

    #[test]
    fn insert_config_lammps_data() {
        let toml_str = r#"
source = "file"
file = "data.lammps"
format = "lammps_data"
material = "glass"
density = 2500.0
radius = 0.001
type_map = { 1 = "glass", 2 = "steel" }
atom_style = "atomic"
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.source, "file");
        assert_eq!(config.format.as_deref(), Some("lammps_data"));
        assert_eq!(config.atom_style.as_deref(), Some("atomic"));
        let tm = config.type_map.unwrap();
        assert_eq!(tm.len(), 2);
    }

    #[test]
    fn insert_config_lammps_data_sphere_style() {
        let toml_str = r#"
source = "file"
file = "data.lammps"
format = "lammps_data"
material = "glass"
atom_style = "bpm/sphere"
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.atom_style.as_deref(), Some("bpm/sphere"));
        // No density/radius required for sphere style (per-atom in file)
        assert!(config.density.is_none());
        assert!(config.radius.is_none());
    }

    #[test]
    fn insert_config_with_region() {
        let toml_str = r#"
material = "glass"
count = 100
radius = 0.001
density = 2500.0
region = { type = "cylinder", center = [0.01, 0.01], radius = 0.008, axis = "z", lo = 0.04, hi = 0.05 }
"#;
        let config: InsertConfig = toml::from_str(toml_str).unwrap();
        assert!(config.region.is_some());
    }
}
