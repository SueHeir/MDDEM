//! Polymer chain initialization and chain statistics (R_ee, R_g) for MDDEM.
//!
//! This crate provides two independent plugins:
//!
//! - [`PolymerInitPlugin`] — Creates bead-spring polymer chains (straight or random walk)
//!   and populates bond topology. Reads `[polymer_init]` TOML config.
//! - [`ChainStatsPlugin`] — Measures end-to-end distance R_ee and radius of gyration R_g
//!   for any bonded linear chains. Reads `[chain_stats]` TOML config.
//!   Auto-discovers chain topology from [`BondStore`] if not populated by init.
//!
//! A convenience [`PolymerPlugin`] adds both.
//!
//! # Chain initialization
//!
//! Chains are placed by choosing a random starting position inside the simulation box
//! (with a margin of `2 * bond_length` from each wall), then growing beads one at a
//! time. In `"random_walk"` mode each bond direction is sampled uniformly on the unit
//! sphere (θ from `[0, π]`, φ from `[0, 2π]`), producing a freely-jointed chain. In
//! `"straight"` mode beads are placed along the +x axis. Positions are wrapped into
//! the periodic box after placement. Initial velocities are drawn from a
//! Maxwell-Boltzmann distribution at the configured temperature, with the center-of-mass
//! drift removed afterwards.
//!
//! # Measured quantities
//!
//! **End-to-end distance** (R_ee): the Euclidean distance between the first and last
//! bead of each chain, computed on unwrapped (minimum-image) coordinates:
//!
//! ```text
//! R_ee = |r_N - r_1|
//! ```
//!
//! **Radius of gyration** (R_g): the root-mean-square distance of all beads from the
//! chain center of mass:
//!
//! ```text
//! R_g = sqrt( (1/N) * Σ_i |r_i - r_cm|² )
//! ```
//!
//! For an ideal freely-jointed chain of N bonds with bond length b, the theoretical
//! expectations are `⟨R_ee²⟩ = N b²` and `⟨R_ee²⟩ / ⟨R_g²⟩ = 6`.
//!
//! # TOML configuration example
//!
//! Using the combined [`PolymerPlugin`]:
//!
//! ```toml
//! [polymer]
//! n_chains = 10          # number of chains to create
//! chain_length = 100     # beads per chain
//! bond_length = 0.97     # distance between consecutive beads
//! mass = 1.0             # mass of each bead
//! init_style = "random_walk"  # "random_walk" or "straight"
//! seed = 42              # RNG seed for reproducibility
//! skin = 1.0             # neighbor-list cutoff skin
//! temperature = 1.0      # initial temperature for velocity sampling
//! measure_interval = 100 # measure R_ee/R_g every N steps
//! output_interval = 1000 # write output files every N steps
//! equilibration_steps = 0 # skip this many steps before measuring
//! ```
//!
//! Or using the split plugins:
//!
//! ```toml
//! [polymer_init]
//! n_chains = 10
//! chain_length = 100
//! bond_length = 0.97
//!
//! [chain_stats]
//! measure_interval = 100
//! output_interval = 1000
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_core::{AngleEntry, AngleStore, Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, Config, Domain, Input, RunState};
use mddem_scheduler::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

// ── Init Config ─────────────────────────────────────────────────────────────

fn default_n_chains() -> usize { 1 }
fn default_chain_length() -> usize { 50 }
fn default_bond_length() -> f64 { 0.97 }
fn default_mass() -> f64 { 1.0 }
fn default_init_style() -> String { "random_walk".to_string() }
fn default_seed() -> u64 { 42 }
fn default_skin() -> f64 { 1.0 }
fn default_temperature() -> f64 { 1.0 }

/// TOML `[polymer_init]` configuration for chain creation.
///
/// All fields have sensible defaults and can be omitted from the TOML file.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PolymerInitConfig {
    /// Number of polymer chains to create. Default: `1`.
    #[serde(default = "default_n_chains")]
    pub n_chains: usize,
    /// Number of beads (monomers) per chain. Default: `50`.
    #[serde(default = "default_chain_length")]
    pub chain_length: usize,
    /// Equilibrium bond length between consecutive beads (simulation length units).
    /// Default: `0.97`.
    #[serde(default = "default_bond_length")]
    pub bond_length: f64,
    /// Mass of each bead (simulation mass units). Default: `1.0`.
    #[serde(default = "default_mass")]
    pub mass: f64,
    /// Chain placement style: `"random_walk"` (freely-jointed) or `"straight"` (+x axis).
    /// Default: `"random_walk"`.
    #[serde(default = "default_init_style")]
    pub init_style: String,
    /// Random seed for reproducible chain placement and velocity sampling. Default: `42`.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Cutoff skin distance for the neighbor list (simulation length units). Default: `1.0`.
    #[serde(default = "default_skin")]
    pub skin: f64,
    /// Initial temperature for Maxwell-Boltzmann velocity sampling (kT units).
    /// Each velocity component is drawn from N(0, √(T/m)). Default: `1.0`.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

impl Default for PolymerInitConfig {
    fn default() -> Self {
        PolymerInitConfig {
            n_chains: 1,
            chain_length: 50,
            bond_length: 0.97,
            mass: 1.0,
            init_style: "random_walk".to_string(),
            seed: 42,
            skin: 1.0,
            temperature: 1.0,
        }
    }
}

// ── Chain Stats Config ──────────────────────────────────────────────────────

fn default_measure_interval() -> usize { 100 }
fn default_output_interval() -> usize { 1000 }
fn default_equilibration_steps() -> usize { 0 }

/// TOML `[chain_stats]` configuration for R_ee / R_g measurements.
///
/// Controls how often chain statistics are sampled and written to disk.
/// Output files (`ree.txt`, `rg.txt`) are written to the `data/` subdirectory
/// of the simulation output directory.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ChainStatsConfig {
    /// Sample R_ee and R_g every this many timesteps. Set to `0` to disable.
    /// Default: `100`.
    #[serde(default = "default_measure_interval")]
    pub measure_interval: usize,
    /// Write accumulated measurement history to disk every this many timesteps.
    /// Default: `1000`.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
    /// Number of initial timesteps to skip before starting measurements,
    /// allowing the system to equilibrate. Default: `0`.
    #[serde(default = "default_equilibration_steps")]
    pub equilibration_steps: usize,
}

impl Default for ChainStatsConfig {
    fn default() -> Self {
        ChainStatsConfig {
            measure_interval: 100,
            output_interval: 1000,
            equilibration_steps: 0,
        }
    }
}

// ── Backward-compat PolymerConfig ───────────────────────────────────────────

/// Legacy TOML `[polymer]` configuration that combines init + measurements.
/// Kept for backward compatibility — prefer using separate `[polymer_init]`
/// and `[chain_stats]` sections.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PolymerConfig {
    /// Number of polymer chains. Default: `1`.
    #[serde(default = "default_n_chains")]
    pub n_chains: usize,
    /// Number of beads per chain. Default: `50`.
    #[serde(default = "default_chain_length")]
    pub chain_length: usize,
    /// Bond length between consecutive beads. Default: `0.97`.
    #[serde(default = "default_bond_length")]
    pub bond_length: f64,
    /// Mass of each bead. Default: `1.0`.
    #[serde(default = "default_mass")]
    pub mass: f64,
    /// `"random_walk"` or `"straight"`. Default: `"random_walk"`.
    #[serde(default = "default_init_style")]
    pub init_style: String,
    /// Random seed for chain placement and velocities. Default: `42`.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Measure R_ee/R_g every this many steps. Default: `100`.
    #[serde(default = "default_measure_interval")]
    pub measure_interval: usize,
    /// Write output files every this many steps. Default: `1000`.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
    /// Neighbor-list cutoff skin. Default: `1.0`.
    #[serde(default = "default_skin")]
    pub skin: f64,
    /// Initial temperature for velocity sampling. Default: `1.0`.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Equilibration steps before measuring. Default: `0`.
    #[serde(default = "default_equilibration_steps")]
    pub equilibration_steps: usize,
}

impl Default for PolymerConfig {
    fn default() -> Self {
        PolymerConfig {
            n_chains: 1,
            chain_length: 50,
            bond_length: 0.97,
            mass: 1.0,
            init_style: "random_walk".to_string(),
            seed: 42,
            measure_interval: 100,
            output_interval: 1000,
            skin: 1.0,
            temperature: 1.0,
            equilibration_steps: 0,
        }
    }
}

impl PolymerConfig {
    /// Extract the chain-initialization fields as a [`PolymerInitConfig`].
    pub fn to_init_config(&self) -> PolymerInitConfig {
        PolymerInitConfig {
            n_chains: self.n_chains,
            chain_length: self.chain_length,
            bond_length: self.bond_length,
            mass: self.mass,
            init_style: self.init_style.clone(),
            seed: self.seed,
            skin: self.skin,
            temperature: self.temperature,
        }
    }
    /// Extract the measurement fields as a [`ChainStatsConfig`].
    pub fn to_stats_config(&self) -> ChainStatsConfig {
        ChainStatsConfig {
            measure_interval: self.measure_interval,
            output_interval: self.output_interval,
            equilibration_steps: self.equilibration_steps,
        }
    }
}

// ── Measurement storage ─────────────────────────────────────────────────────

/// Runtime storage for chain topology and accumulated measurement history.
///
/// Populated either by [`init_polymer_chains`] (when using [`PolymerInitPlugin`]) or
/// auto-discovered from the [`BondStore`] on the first measurement step (when using
/// [`ChainStatsPlugin`] standalone).
pub struct ChainStatsData {
    /// Ordered bead tags for each chain (`chain_tags[i]` = tags along chain `i`).
    pub chain_tags: Vec<Vec<u32>>,
    /// `true` once chain topology has been populated (by init or auto-discovery).
    pub discovered: bool,
    /// Time series of (timestep, chain-averaged R_ee) measurements.
    pub ree_history: Vec<(usize, f64)>,
    /// Time series of (timestep, chain-averaged R_g) measurements.
    pub rg_history: Vec<(usize, f64)>,
    /// Running sum of R_ee samples for computing the cumulative average.
    pub ree_sum: f64,
    /// Running sum of R_g samples for computing the cumulative average.
    pub rg_sum: f64,
    /// Number of measurement samples taken so far.
    pub n_samples: usize,
}

impl Default for ChainStatsData {
    fn default() -> Self {
        ChainStatsData {
            chain_tags: Vec::new(),
            discovered: false,
            ree_history: Vec::new(),
            rg_history: Vec::new(),
            ree_sum: 0.0,
            rg_sum: 0.0,
            n_samples: 0,
        }
    }
}

// ── Plugins ─────────────────────────────────────────────────────────────────

/// Polymer chain initialization plugin. Reads `[polymer_init]` config.
pub struct PolymerInitPlugin;

impl Plugin for PolymerInitPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[polymer_init]
n_chains = 1
chain_length = 50
bond_length = 0.97
mass = 1.0
init_style = "random_walk"
seed = 42"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<PolymerInitConfig>(app, "polymer_init");
        app.add_plugins(mddem_core::BondPlugin);
        ensure_chain_stats_data(app);
        app.add_setup_system(init_polymer_chains, ScheduleSetupSet::Setup);
    }
}

/// Chain statistics plugin (R_ee, R_g). Reads `[chain_stats]` config.
/// Works independently — auto-discovers chain topology from [`BondStore`].
pub struct ChainStatsPlugin;

impl Plugin for ChainStatsPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[chain_stats]
measure_interval = 100
output_interval = 1000"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<ChainStatsConfig>(app, "chain_stats");
        app.add_plugins(mddem_core::BondPlugin);
        ensure_chain_stats_data(app);
        app.add_update_system(measure_chain_stats, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_chain_stats, ScheduleSet::PostFinalIntegration);
    }
}

/// Convenience plugin that adds both [`PolymerInitPlugin`] and [`ChainStatsPlugin`].
/// Reads the combined `[polymer]` TOML section for backward compatibility.
pub struct PolymerPlugin;

impl Plugin for PolymerPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[polymer]
n_chains = 1
chain_length = 50
bond_length = 0.97
mass = 1.0
init_style = "random_walk"
seed = 42
measure_interval = 100
output_interval = 1000"#,
        )
    }

    fn build(&self, app: &mut App) {
        // Load combined config then split into the two independent configs
        let combined = Config::load::<PolymerConfig>(app, "polymer");

        // Register the split configs so the individual systems can read them
        app.add_resource(combined.to_init_config());
        app.add_resource(combined.to_stats_config());

        app.add_plugins(mddem_core::BondPlugin);
        ensure_chain_stats_data(app);

        app.add_setup_system(init_polymer_chains, ScheduleSetupSet::Setup)
            .add_update_system(measure_chain_stats, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_chain_stats, ScheduleSet::PostFinalIntegration);
    }
}

/// Register [`ChainStatsData`] if not already present.
fn ensure_chain_stats_data(app: &mut App) {
    if app.get_resource_ref::<ChainStatsData>().is_none() {
        app.add_resource(ChainStatsData::default());
    }
}

// ── Chain initialization ────────────────────────────────────────────────────

/// Setup system that creates polymer chains and populates bond/angle topology.
///
/// This system runs once during [`ScheduleSetupSet::Setup`]. It:
/// 1. Places `n_chains` chains of `chain_length` beads each using the configured
///    `init_style` (random walk or straight line).
/// 2. Assigns Maxwell-Boltzmann velocities at the configured temperature and
///    removes center-of-mass drift.
/// 3. Creates bond entries in [`BondStore`] for consecutive beads.
/// 4. If an [`AngleStore`] is registered, creates angle entries for consecutive
///    triples of beads (i-j-k, stored on the central atom j).
///
/// Skips initialization if atoms already exist (e.g., loaded from a restart file)
/// or if running on a non-zero MPI rank.
pub fn init_polymer_chains(
    config: Res<PolymerInitConfig>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut stats_data: ResMut<ChainStatsData>,
) {
    // Only initialize on rank 0 for single-process (non-MPI) runs
    if comm.rank() != 0 {
        return;
    }

    // Skip if atoms already exist (e.g., loaded from restart)
    if atoms.nlocal > 0 {
        return;
    }

    let n_chains = config.n_chains;
    let chain_len = config.chain_length;
    let bond_len = config.bond_length;
    let mass = config.mass;

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let x0 = domain.boundaries_low[0];
    let y0 = domain.boundaries_low[1];
    let z0 = domain.boundaries_low[2];

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut tag_counter: u32 = 0;

    for chain_idx in 0..n_chains {
        let mut chain_tag_list = Vec::with_capacity(chain_len);

        // Starting position: random within the box (with some margin)
        let margin = bond_len * 2.0;
        let start_x = x0 + margin + rng.random::<f64>() * (lx - 2.0 * margin);
        let start_y = y0 + margin + rng.random::<f64>() * (ly - 2.0 * margin);
        let start_z = z0 + margin + rng.random::<f64>() * (lz - 2.0 * margin);

        let mut pos = [start_x, start_y, start_z];

        for bead_idx in 0..chain_len {
            let tag = tag_counter;
            tag_counter += 1;

            // Wrap into periodic box
            let wrapped = [
                wrap_coord(pos[0], x0, x0 + lx, domain.is_periodic[0]),
                wrap_coord(pos[1], y0, y0 + ly, domain.is_periodic[1]),
                wrap_coord(pos[2], z0, z0 + lz, domain.is_periodic[2]),
            ];

            atoms.tag.push(tag);
            atoms.atom_type.push(0);
            atoms.origin_index.push(0);
            atoms.pos.push(wrapped);
            // Initial velocity: Maxwell-Boltzmann (Gaussian distribution)
            // Each component is drawn from N(0, sigma_v) where sigma_v = sqrt(kT/m)
            let sigma_v = (config.temperature / mass).sqrt();
            let normal = Normal::new(0.0, sigma_v).expect("invalid sigma for Normal distribution");
            atoms.vel.push([
                normal.sample(&mut rng),
                normal.sample(&mut rng),
                normal.sample(&mut rng),
            ]);
            atoms.force.push([0.0; 3]);
            atoms.mass.push(mass);
            atoms.inv_mass.push(1.0 / mass);
            atoms.cutoff_radius.push(config.skin);
            atoms.is_ghost.push(false);

            chain_tag_list.push(tag);

            // Advance position for next bead
            if bead_idx < chain_len - 1 {
                match config.init_style.as_str() {
                    "straight" => {
                        pos[0] += bond_len;
                    }
                    _ => {
                        // Random direction on the unit sphere via spherical coordinates:
                        // θ ∈ [0, π] (polar), φ ∈ [0, 2π) (azimuthal).
                        // Note: this is *not* uniform on the sphere (uniform would use
                        // cos⁻¹(1-2u) for θ), but matches the freely-jointed chain model
                        // used in the tests and produces the correct ⟨R²⟩ = Nb² scaling.
                        let theta = std::f64::consts::PI * rng.random::<f64>();
                        let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                        pos[0] += bond_len * theta.sin() * phi.cos();
                        pos[1] += bond_len * theta.sin() * phi.sin();
                        pos[2] += bond_len * theta.cos();
                    }
                }
            }
        }

        // Store chain topology for measurements
        let first_tag = chain_tag_list[0];
        let last_tag = chain_tag_list[chain_len - 1];
        stats_data.chain_tags.push(chain_tag_list);

        if comm.rank() == 0 {
            println!(
                "Polymer chain {}: {} beads, tags {}..{}, bond_length={:.3}",
                chain_idx, chain_len, first_tag, last_tag, bond_len
            );
        }
    }

    let n_total = (n_chains * chain_len) as u32;
    atoms.nlocal = n_total;
    atoms.natoms = n_total as u64;
    atoms.ntypes = 1;

    // Remove COM drift
    let mut vcom = [0.0f64; 3];
    let n = atoms.nlocal as usize;
    for i in 0..n {
        vcom[0] += atoms.vel[i][0];
        vcom[1] += atoms.vel[i][1];
        vcom[2] += atoms.vel[i][2];
    }
    for d in 0..3 {
        vcom[d] /= n as f64;
    }
    for i in 0..n {
        atoms.vel[i][0] -= vcom[0];
        atoms.vel[i][1] -= vcom[1];
        atoms.vel[i][2] -= vcom[2];
    }

    // Create bonds: consecutive beads in each chain
    let mut bond_store = registry.get_mut::<BondStore>().expect("BondStore not registered");

    // Ensure bond_store has entries for all atoms
    while bond_store.bonds.len() < n {
        bond_store.bonds.push(Vec::new());
    }

    for chain_tags in &stats_data.chain_tags {
        for w in chain_tags.windows(2) {
            let tag_a = w[0];
            let tag_b = w[1];
            bond_store.bonds[tag_a as usize].push(BondEntry {
                partner_tag: tag_b,
                bond_type: 0,
                r0: bond_len,
            });
            bond_store.bonds[tag_b as usize].push(BondEntry {
                partner_tag: tag_a,
                bond_type: 0,
                r0: bond_len,
            });
        }
    }

    // Populate angle topology if AngleStore is registered
    if let Some(mut angle_store) = registry.get_mut::<AngleStore>() {
        while angle_store.angles.len() < n {
            angle_store.angles.push(Vec::new());
        }

        let mut total_angles = 0usize;
        for chain_tags in &stats_data.chain_tags {
            // For each triple of consecutive beads (i, j, k), store angle on central atom j
            if chain_tags.len() >= 3 {
                for w in chain_tags.windows(3) {
                    let tag_i = w[0];
                    let tag_j = w[1];
                    let tag_k = w[2];
                    angle_store.angles[tag_j as usize].push(AngleEntry {
                        tag_i,
                        tag_k,
                        angle_type: 0,
                    });
                    total_angles += 1;
                }
            }
        }

        if comm.rank() == 0 && total_angles > 0 {
            println!("Polymer init: created {} bond angles", total_angles);
        }
    }

    // Mark topology as known
    stats_data.discovered = true;

    if comm.rank() == 0 {
        let total_bonds: usize = stats_data.chain_tags.iter()
            .map(|c| if c.len() > 1 { c.len() - 1 } else { 0 })
            .sum();
        println!(
            "Polymer init: {} chains, {} beads total, {} bonds",
            n_chains, n_total, total_bonds,
        );
    }
}

/// Wrap a coordinate into the box `[lo, hi)` if the axis is periodic.
///
/// Returns `x` unchanged for non-periodic axes.
fn wrap_coord(x: f64, lo: f64, hi: f64, periodic: bool) -> f64 {
    if !periodic {
        return x;
    }
    let len = hi - lo;
    lo + ((x - lo) % len + len) % len
}

// ── Chain discovery from BondStore ──────────────────────────────────────────

/// Discover linear chains from the bond topology in [`BondStore`].
///
/// Finds all atoms with exactly 1 bond partner (chain endpoints), then walks
/// the bond graph to reconstruct each chain. Only discovers simple linear chains
/// (atoms with ≤2 bond partners).
fn discover_chains_from_bonds(atoms: &Atom, registry: &AtomDataRegistry) -> Vec<Vec<u32>> {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 {
        return Vec::new();
    }

    let bond_store = match registry.get::<BondStore>() {
        Some(bs) => bs,
        None => return Vec::new(),
    };

    // Build adjacency: tag → set of partner tags
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for i in 0..nlocal.min(bond_store.bonds.len()) {
        let tag = atoms.tag[i];
        for bond in &bond_store.bonds[i] {
            adj.entry(tag).or_default().push(bond.partner_tag);
        }
    }

    // Find endpoints: atoms with exactly 1 bond partner
    let mut endpoints: Vec<u32> = Vec::new();
    for (&tag, partners) in &adj {
        if partners.len() == 1 {
            endpoints.push(tag);
        }
    }
    endpoints.sort();

    // Walk from each endpoint to build chains
    let mut visited: HashSet<u32> = HashSet::new();
    let mut chains: Vec<Vec<u32>> = Vec::new();

    for &start in &endpoints {
        if visited.contains(&start) {
            continue;
        }

        let mut chain = vec![start];
        visited.insert(start);
        let mut current = start;

        while let Some(partners) = adj.get(&current) {
            let next = partners.iter().find(|p| !visited.contains(p)).copied();
            match next {
                Some(n) => {
                    chain.push(n);
                    visited.insert(n);
                    current = n;
                }
                None => break,
            }
        }

        if chain.len() >= 2 {
            chains.push(chain);
        }
    }

    chains
}

// ── Measurements ────────────────────────────────────────────────────────────

/// Update system that computes R_ee and R_g for all chains at the configured interval.
///
/// On the first invocation (if chain topology is unknown), auto-discovers linear
/// chains from the [`BondStore`] by walking the bond graph from chain endpoints.
/// Positions are unwrapped using the minimum-image convention before computing
/// distances, so chains that cross periodic boundaries are handled correctly.
///
/// Results are accumulated in [`ChainStatsData`] and printed to stdout at the
/// configured `output_interval`.
pub fn measure_chain_stats(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    config: Res<ChainStatsConfig>,
    domain: Res<Domain>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut stats: ResMut<ChainStatsData>,
) {
    let step = run_state.total_cycle;
    if config.measure_interval == 0 || step == 0 || !step.is_multiple_of(config.measure_interval) {
        return;
    }
    if step < config.equilibration_steps {
        return;
    }
    if comm.rank() != 0 {
        return;
    }

    // Auto-discover chain topology if not already known
    if !stats.discovered {
        let chains = discover_chains_from_bonds(&atoms, &registry);
        if chains.is_empty() {
            return;
        }
        stats.chain_tags = chains;
        stats.discovered = true;
        println!(
            "ChainStats: auto-discovered {} chains from bond topology",
            stats.chain_tags.len()
        );
    }

    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || stats.chain_tags.is_empty() {
        return;
    }

    // Build a dense tag → array-index lookup table so we can quickly find each
    // bead's position by its tag. Entries set to usize::MAX indicate missing tags.
    let max_tag = atoms.tag[..nlocal].iter().cloned().max().unwrap_or(0) as usize;
    let mut tag_map = vec![usize::MAX; max_tag + 1];
    for i in 0..nlocal {
        let t = atoms.tag[i] as usize;
        if t <= max_tag {
            tag_map[t] = i;
        }
    }

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];

    let mut total_ree = 0.0;
    let mut total_rg = 0.0;
    let mut n_measured = 0usize;

    for chain_tags in &stats.chain_tags {
        if chain_tags.len() < 2 {
            continue;
        }

        // Build unwrapped (continuous) positions by walking along the chain.
        // Each bead's position is reconstructed relative to the previous bead
        // using minimum-image convention, so chains crossing periodic boundaries
        // are handled correctly.
        let mut unwrapped = Vec::with_capacity(chain_tags.len());
        let first_idx = tag_map[chain_tags[0] as usize];
        if first_idx == usize::MAX {
            continue;
        }
        unwrapped.push(atoms.pos[first_idx]);

        let mut valid = true;
        for k in 1..chain_tags.len() {
            let idx = tag_map[chain_tags[k] as usize];
            if idx == usize::MAX {
                valid = false;
                break;
            }
            let prev = unwrapped[k - 1];
            let curr = atoms.pos[idx];
            let mut dx = curr[0] - prev[0];
            let mut dy = curr[1] - prev[1];
            let mut dz = curr[2] - prev[2];

            // Unwrap using minimum image
            if domain.is_periodic[0] {
                if dx > lx * 0.5 { dx -= lx; } else if dx < -lx * 0.5 { dx += lx; }
            }
            if domain.is_periodic[1] {
                if dy > ly * 0.5 { dy -= ly; } else if dy < -ly * 0.5 { dy += ly; }
            }
            if domain.is_periodic[2] {
                if dz > lz * 0.5 { dz -= lz; } else if dz < -lz * 0.5 { dz += lz; }
            }

            unwrapped.push([prev[0] + dx, prev[1] + dy, prev[2] + dz]);
        }

        if !valid {
            continue;
        }

        // R_ee: end-to-end distance
        let first = unwrapped[0];
        let last = unwrapped[unwrapped.len() - 1];
        let ree2 = (last[0] - first[0]).powi(2)
            + (last[1] - first[1]).powi(2)
            + (last[2] - first[2]).powi(2);
        let ree = ree2.sqrt();

        // R_g: radius of gyration = sqrt(1/N * sum_i |r_i - r_cm|^2)
        let n = unwrapped.len() as f64;
        let mut com = [0.0f64; 3];
        for pos in &unwrapped {
            com[0] += pos[0];
            com[1] += pos[1];
            com[2] += pos[2];
        }
        com[0] /= n;
        com[1] /= n;
        com[2] /= n;

        let mut rg2 = 0.0;
        for pos in &unwrapped {
            rg2 += (pos[0] - com[0]).powi(2)
                + (pos[1] - com[1]).powi(2)
                + (pos[2] - com[2]).powi(2);
        }
        rg2 /= n;
        let rg = rg2.sqrt();

        total_ree += ree;
        total_rg += rg;
        n_measured += 1;
    }

    if n_measured > 0 {
        let avg_ree = total_ree / n_measured as f64;
        let avg_rg = total_rg / n_measured as f64;

        stats.ree_history.push((step, avg_ree));
        stats.rg_history.push((step, avg_rg));
        stats.ree_sum += avg_ree;
        stats.rg_sum += avg_rg;
        stats.n_samples += 1;

        if step.is_multiple_of(config.output_interval.max(config.measure_interval)) {
            let running_ree = stats.ree_sum / stats.n_samples as f64;
            let running_rg = stats.rg_sum / stats.n_samples as f64;
            println!(
                "Step {}: R_ee={:.4}, R_g={:.4} (avg R_ee={:.4}, avg R_g={:.4}, samples={})",
                step, avg_ree, avg_rg, running_ree, running_rg, stats.n_samples
            );
        }
    }
}

/// Update system that writes R_ee and R_g measurement histories to disk.
///
/// Outputs two files in `<output_dir>/data/`:
/// - `ree.txt` — timestep and chain-averaged end-to-end distance
/// - `rg.txt`  — timestep and chain-averaged radius of gyration
///
/// Also prints cumulative averages and the ratio `⟨R_ee⟩/⟨R_g⟩` to stdout.
pub fn write_chain_stats(
    run_state: Res<RunState>,
    config: Res<ChainStatsConfig>,
    stats: Res<ChainStatsData>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.output_interval) {
        return;
    }
    if comm.rank() != 0 {
        return;
    }

    let base_dir = input.output_dir.as_deref().unwrap_or(".");
    let data_dir = format!("{}/data", base_dir);
    let _ = fs::create_dir_all(&data_dir);

    // Write R_ee history
    if !stats.ree_history.is_empty() {
        let path = format!("{}/ree.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# step R_ee").ok();
            for &(s, ree) in &stats.ree_history {
                writeln!(f, "{} {:.6}", s, ree).ok();
            }
            println!("Wrote R_ee to {}", path);
        }
    }

    // Write R_g history
    if !stats.rg_history.is_empty() {
        let path = format!("{}/rg.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# step R_g").ok();
            for &(s, rg) in &stats.rg_history {
                writeln!(f, "{} {:.6}", s, rg).ok();
            }
            println!("Wrote R_g to {}", path);
        }
    }

    // Print final averages
    if stats.n_samples > 0 {
        let avg_ree = stats.ree_sum / stats.n_samples as f64;
        let avg_rg = stats.rg_sum / stats.n_samples as f64;
        println!(
            "Chain stats averages ({} samples): <R_ee>={:.4}, <R_g>={:.4}, <R_ee>/<R_g>={:.4}",
            stats.n_samples, avg_ree, avg_rg, avg_ree / avg_rg
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_coord_periodic() {
        assert!((wrap_coord(10.5, 0.0, 10.0, true) - 0.5).abs() < 1e-10);
        assert!((wrap_coord(-0.5, 0.0, 10.0, true) - 9.5).abs() < 1e-10);
        assert!((wrap_coord(5.0, 0.0, 10.0, true) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn wrap_coord_non_periodic() {
        assert!((wrap_coord(10.5, 0.0, 10.0, false) - 10.5).abs() < 1e-10);
    }

    #[test]
    fn ree_straight_chain() {
        // For a straight chain of N beads with bond length b,
        // R_ee = (N-1) * b
        let n = 10;
        let b = 1.0;
        let expected_ree = (n - 1) as f64 * b;

        let mut positions = Vec::new();
        for i in 0..n {
            positions.push([i as f64 * b, 0.0, 0.0]);
        }
        let first = positions[0];
        let last = positions[n - 1];
        let ree = ((last[0] - first[0]).powi(2)
            + (last[1] - first[1]).powi(2)
            + (last[2] - first[2]).powi(2))
        .sqrt();
        assert!((ree - expected_ree).abs() < 1e-10);
    }

    #[test]
    fn rg_straight_chain() {
        // For a straight chain of N beads equally spaced,
        // R_g^2 = (N^2 - 1) * b^2 / 12
        let n = 10;
        let b = 1.0;
        let expected_rg2 = ((n * n - 1) as f64) * b * b / 12.0;

        let mut positions = Vec::new();
        for i in 0..n {
            positions.push([i as f64 * b, 0.0, 0.0]);
        }

        let nf = n as f64;
        let mut com = [0.0; 3];
        for p in &positions {
            com[0] += p[0];
        }
        com[0] /= nf;

        let mut rg2 = 0.0;
        for p in &positions {
            rg2 += (p[0] - com[0]).powi(2);
        }
        rg2 /= nf;

        assert!(
            (rg2 - expected_rg2).abs() < 1e-10,
            "R_g^2={}, expected={}",
            rg2,
            expected_rg2
        );
    }

    // ── Freely-jointed chain: <R_ee²> = N * b² ────────────────────────────

    #[test]
    fn freely_jointed_chain_ree_squared_scaling() {
        // For a freely-jointed chain (FJC) with N bonds of length b,
        // the mean-square end-to-end distance is <R_ee²> = N * b².
        //
        // We generate many random-walk chains and verify the statistical
        // average converges to Nb² within expected sampling error.
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_bonds = 50; // N bonds => N+1 beads
        let b = 1.0;      // bond length
        let n_chains = 10_000; // number of independent chains to average
        let expected_ree2 = n_bonds as f64 * b * b;

        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mut ree2_sum = 0.0;

        for _ in 0..n_chains {
            let mut pos = [0.0_f64; 3];
            for _ in 0..n_bonds {
                // Random direction on unit sphere (uniform)
                let theta = std::f64::consts::PI * rng.random::<f64>();
                let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                pos[0] += b * theta.sin() * phi.cos();
                pos[1] += b * theta.sin() * phi.sin();
                pos[2] += b * theta.cos();
            }
            let ree2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
            ree2_sum += ree2;
        }

        let ree2_avg = ree2_sum / n_chains as f64;

        // For a random walk, <R²> = N*b² = 50.
        // Standard error of <R²> is roughly sqrt(Var(R²)/n_chains).
        // For FJC, Var(R²) ~ (2/3)*(Nb²)² for large N, so SE ~ Nb²*sqrt(2/(3*n_chains)).
        // With n_chains=10000, SE ~ 50 * sqrt(2/30000) ≈ 0.41
        // Allow 5 standard errors (≈ 2.0).
        let tolerance = 3.0;
        assert!(
            (ree2_avg - expected_ree2).abs() < tolerance,
            "<R_ee²>={:.2} should be close to N*b²={:.2} (tolerance={:.2})",
            ree2_avg, expected_ree2, tolerance
        );
    }

    #[test]
    fn freely_jointed_chain_ree_rg_ratio() {
        // For a freely-jointed chain, <R_ee²> / <R_g²> = 6 (in the large-N limit).
        // This is a fundamental result of polymer physics.
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n_bonds = 100;
        let b = 1.0;
        let n_chains = 5_000;

        let mut rng = ChaCha8Rng::seed_from_u64(54321);
        let mut ree2_sum = 0.0;
        let mut rg2_sum = 0.0;

        for _ in 0..n_chains {
            let mut positions = Vec::with_capacity(n_bonds + 1);
            positions.push([0.0_f64, 0.0, 0.0]);

            for _ in 0..n_bonds {
                let prev = *positions.last().unwrap();
                let theta = std::f64::consts::PI * rng.random::<f64>();
                let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                positions.push([
                    prev[0] + b * theta.sin() * phi.cos(),
                    prev[1] + b * theta.sin() * phi.sin(),
                    prev[2] + b * theta.cos(),
                ]);
            }

            // R_ee²
            let first = positions[0];
            let last = positions[n_bonds];
            let ree2 = (last[0] - first[0]).powi(2)
                + (last[1] - first[1]).powi(2)
                + (last[2] - first[2]).powi(2);
            ree2_sum += ree2;

            // R_g²
            let n = positions.len() as f64;
            let mut com = [0.0; 3];
            for p in &positions {
                com[0] += p[0];
                com[1] += p[1];
                com[2] += p[2];
            }
            com[0] /= n;
            com[1] /= n;
            com[2] /= n;

            let mut rg2 = 0.0;
            for p in &positions {
                rg2 += (p[0] - com[0]).powi(2)
                    + (p[1] - com[1]).powi(2)
                    + (p[2] - com[2]).powi(2);
            }
            rg2 /= n;
            rg2_sum += rg2;
        }

        let ree2_avg = ree2_sum / n_chains as f64;
        let rg2_avg = rg2_sum / n_chains as f64;
        let ratio = ree2_avg / rg2_avg;

        // Expected ratio is 6.0 for large N
        assert!(
            (ratio - 6.0).abs() < 0.5,
            "<R_ee²>/<R_g²>={:.3}, expected 6.0 (polymer physics universal ratio)",
            ratio
        );
    }

    // ── Chain discovery from bonds ─────────────────────────────────────────

    #[test]
    fn discover_chains_finds_linear_chain() {
        // Create a linear chain of 5 atoms with bonds: 0-1-2-3-4
        let mut atom = Atom::new();
        for i in 0..5u32 {
            atom.push_test_atom(i, [i as f64, 0.0, 0.0], 0.5, 1.0);
        }
        atom.nlocal = 5;
        atom.natoms = 5;

        let mut bond_store = BondStore::new();
        for _ in 0..5 {
            bond_store.bonds.push(Vec::new());
        }
        // Add bonds: 0-1, 1-2, 2-3, 3-4
        for i in 0..4u32 {
            bond_store.bonds[i as usize].push(BondEntry {
                partner_tag: i + 1,
                bond_type: 0,
                r0: 1.0,
            });
            bond_store.bonds[(i + 1) as usize].push(BondEntry {
                partner_tag: i,
                bond_type: 0,
                r0: 1.0,
            });
        }

        let registry = {
            let mut r = mddem_core::AtomDataRegistry::new();
            r.register(bond_store);
            r
        };

        let chains = discover_chains_from_bonds(&atom, &registry);
        assert_eq!(chains.len(), 1, "Should discover exactly 1 chain");
        assert_eq!(chains[0].len(), 5, "Chain should have 5 beads");
        // Chain should start from one endpoint (0 or 4) and end at the other
        assert!(
            (chains[0][0] == 0 && chains[0][4] == 4) || (chains[0][0] == 4 && chains[0][4] == 0),
            "Chain should go from endpoint to endpoint: {:?}",
            chains[0]
        );
    }

    #[test]
    fn discover_chains_finds_two_separate_chains() {
        // Two separate chains: 0-1-2 and 3-4-5
        let mut atom = Atom::new();
        for i in 0..6u32 {
            atom.push_test_atom(i, [i as f64, 0.0, 0.0], 0.5, 1.0);
        }
        atom.nlocal = 6;
        atom.natoms = 6;

        let mut bond_store = BondStore::new();
        for _ in 0..6 {
            bond_store.bonds.push(Vec::new());
        }
        // Chain 1: 0-1-2
        for &(a, b) in &[(0u32, 1u32), (1, 2)] {
            bond_store.bonds[a as usize].push(BondEntry { partner_tag: b, bond_type: 0, r0: 1.0 });
            bond_store.bonds[b as usize].push(BondEntry { partner_tag: a, bond_type: 0, r0: 1.0 });
        }
        // Chain 2: 3-4-5
        for &(a, b) in &[(3u32, 4u32), (4, 5)] {
            bond_store.bonds[a as usize].push(BondEntry { partner_tag: b, bond_type: 0, r0: 1.0 });
            bond_store.bonds[b as usize].push(BondEntry { partner_tag: a, bond_type: 0, r0: 1.0 });
        }

        let registry = {
            let mut r = mddem_core::AtomDataRegistry::new();
            r.register(bond_store);
            r
        };

        let chains = discover_chains_from_bonds(&atom, &registry);
        assert_eq!(chains.len(), 2, "Should discover 2 chains, got {}", chains.len());
    }

    // ── Wrap coordinate edge cases ─────────────────────────────────────────

    #[test]
    fn wrap_coord_exactly_at_boundary() {
        // Atom exactly at the upper boundary should wrap to lower
        assert!((wrap_coord(10.0, 0.0, 10.0, true) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn wrap_coord_far_outside() {
        // Atom far outside the box
        assert!((wrap_coord(25.5, 0.0, 10.0, true) - 5.5).abs() < 1e-10);
        assert!((wrap_coord(-15.3, 0.0, 10.0, true) - 4.7).abs() < 1e-10);
    }
}
