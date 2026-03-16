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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_core::{Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, Config, Domain, Input, RunState};
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
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PolymerInitConfig {
    /// Number of polymer chains.
    #[serde(default = "default_n_chains")]
    pub n_chains: usize,
    /// Number of beads per chain.
    #[serde(default = "default_chain_length")]
    pub chain_length: usize,
    /// Bond length between consecutive beads.
    #[serde(default = "default_bond_length")]
    pub bond_length: f64,
    /// Mass of each bead.
    #[serde(default = "default_mass")]
    pub mass: f64,
    /// Initialization style: "straight" or "random_walk".
    #[serde(default = "default_init_style")]
    pub init_style: String,
    /// Random seed for random walk initialization and initial velocities.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Cutoff skin for neighbor list.
    #[serde(default = "default_skin")]
    pub skin: f64,
    /// Initial temperature for Maxwell-Boltzmann velocity sampling.
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
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ChainStatsConfig {
    /// Measure R_ee and R_g every N steps.
    #[serde(default = "default_measure_interval")]
    pub measure_interval: usize,
    /// Write measurement output files every N steps.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
    /// Number of equilibration steps before starting measurements.
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
    #[serde(default = "default_n_chains")]
    pub n_chains: usize,
    #[serde(default = "default_chain_length")]
    pub chain_length: usize,
    #[serde(default = "default_bond_length")]
    pub bond_length: f64,
    #[serde(default = "default_mass")]
    pub mass: f64,
    #[serde(default = "default_init_style")]
    pub init_style: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_measure_interval")]
    pub measure_interval: usize,
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
    #[serde(default = "default_skin")]
    pub skin: f64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
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
    /// Extract the init portion.
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
    /// Extract the measurement portion.
    pub fn to_stats_config(&self) -> ChainStatsConfig {
        ChainStatsConfig {
            measure_interval: self.measure_interval,
            output_interval: self.output_interval,
            equilibration_steps: self.equilibration_steps,
        }
    }
}

// ── Measurement storage ─────────────────────────────────────────────────────

/// Stores chain topology (first/last bead tags) and measurement history.
pub struct ChainStatsData {
    /// All bead tags per chain, ordered along the chain.
    pub chain_tags: Vec<Vec<u32>>,
    /// Whether chain topology has been discovered from bonds.
    pub discovered: bool,
    /// (step, R_ee) values over time.
    pub ree_history: Vec<(usize, f64)>,
    /// (step, R_g) values over time.
    pub rg_history: Vec<(usize, f64)>,
    /// Running sum for averages.
    pub ree_sum: f64,
    pub rg_sum: f64,
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
                        // Random direction on unit sphere
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

    // Build tag-to-index map
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

        // Compute unwrapped positions along the chain (walk from first bead)
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
}
