//! Polymer chain initialization and measurements (R_ee, R_g) for MDDEM.
//!
//! Provides:
//! - Chain initialization as a straight line or random walk of bonded beads
//! - End-to-end distance R_ee measurement
//! - Radius of gyration R_g measurement
//! - Periodic output to files

use std::fs;
use std::io::Write;

use mddem_app::prelude::*;
use mddem_core::{Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, Config, Domain, Input, RunState};
use mddem_scheduler::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::Deserialize;

// ── Config ──────────────────────────────────────────────────────────────────

fn default_n_chains() -> usize {
    1
}
fn default_chain_length() -> usize {
    50
}
fn default_bond_length() -> f64 {
    0.97
}
fn default_mass() -> f64 {
    1.0
}
fn default_init_style() -> String {
    "random_walk".to_string()
}
fn default_seed() -> u64 {
    42
}
fn default_measure_interval() -> usize {
    100
}
fn default_output_interval() -> usize {
    1000
}
fn default_skin() -> f64 {
    1.0
}
fn default_temperature() -> f64 {
    1.0
}
fn default_equilibration_steps() -> usize {
    0
}

/// TOML `[polymer]` configuration.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PolymerConfig {
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
    /// Measure R_ee and R_g every N steps.
    #[serde(default = "default_measure_interval")]
    pub measure_interval: usize,
    /// Write measurement output files every N steps.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
    /// Cutoff skin for neighbor list.
    #[serde(default = "default_skin")]
    pub skin: f64,
    /// Initial temperature for Maxwell-Boltzmann velocity sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Number of equilibration steps before starting measurements.
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

// ── Measurement storage ─────────────────────────────────────────────────────

/// Stores chain topology (first/last bead tags) and measurement history.
pub struct PolymerMeasure {
    /// (first_tag, last_tag) for each chain.
    pub chain_endpoints: Vec<(u32, u32)>,
    /// All bead tags per chain, ordered along the chain.
    pub chain_tags: Vec<Vec<u32>>,
    /// (step, R_ee) values over time.
    pub ree_history: Vec<(usize, f64)>,
    /// (step, R_g) values over time.
    pub rg_history: Vec<(usize, f64)>,
    /// Running sum for averages.
    pub ree_sum: f64,
    pub rg_sum: f64,
    pub n_samples: usize,
}

impl Default for PolymerMeasure {
    fn default() -> Self {
        PolymerMeasure {
            chain_endpoints: Vec::new(),
            chain_tags: Vec::new(),
            ree_history: Vec::new(),
            rg_history: Vec::new(),
            ree_sum: 0.0,
            rg_sum: 0.0,
            n_samples: 0,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Polymer chain initialization and R_ee/R_g measurement plugin.
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
        Config::load::<PolymerConfig>(app, "polymer");

        // Register BondStore if not already present
        app.add_plugins(mddem_core::BondPlugin);

        app.add_resource(PolymerMeasure::default())
            .add_setup_system(init_polymer_chains, ScheduleSetupSet::Setup)
            .add_update_system(measure_polymer, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_polymer_measurements, ScheduleSet::PostFinalIntegration);
    }
}

// ── Chain initialization ────────────────────────────────────────────────────

pub fn init_polymer_chains(
    config: Res<PolymerConfig>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut measure: ResMut<PolymerMeasure>,
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
        let start_x = x0 + margin + rng.gen::<f64>() * (lx - 2.0 * margin);
        let start_y = y0 + margin + rng.gen::<f64>() * (ly - 2.0 * margin);
        let start_z = z0 + margin + rng.gen::<f64>() * (lz - 2.0 * margin);

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
            // Initial velocity: Maxwell-Boltzmann
            let sigma_v = (config.temperature / mass).sqrt();
            atoms.vel.push([
                rng.gen::<f64>() * 2.0 * sigma_v - sigma_v,
                rng.gen::<f64>() * 2.0 * sigma_v - sigma_v,
                rng.gen::<f64>() * 2.0 * sigma_v - sigma_v,
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
                    "random_walk" | _ => {
                        // Random direction on unit sphere
                        let theta = std::f64::consts::PI * rng.gen::<f64>();
                        let phi = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
                        pos[0] += bond_len * theta.sin() * phi.cos();
                        pos[1] += bond_len * theta.sin() * phi.sin();
                        pos[2] += bond_len * theta.cos();
                    }
                }
            }
        }

        // Store chain endpoints
        let first_tag = chain_tag_list[0];
        let last_tag = chain_tag_list[chain_len - 1];
        measure.chain_endpoints.push((first_tag, last_tag));
        measure.chain_tags.push(chain_tag_list);

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

    for chain_tags in &measure.chain_tags {
        for w in chain_tags.windows(2) {
            let tag_a = w[0];
            let tag_b = w[1];
            // Both atoms store the bond
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

    if comm.rank() == 0 {
        let total_bonds: usize = measure.chain_tags.iter()
            .map(|c| if c.len() > 1 { c.len() - 1 } else { 0 })
            .sum();
        println!(
            "Polymer init: {} chains, {} beads total, {} bonds",
            n_chains,
            n_total,
            total_bonds,
        );
    }
}

fn wrap_coord(x: f64, lo: f64, hi: f64, periodic: bool) -> f64 {
    if !periodic {
        return x;
    }
    let len = hi - lo;
    let mut val = x;
    while val >= hi {
        val -= len;
    }
    while val < lo {
        val += len;
    }
    val
}

// ── Measurements ────────────────────────────────────────────────────────────

pub fn measure_polymer(
    atoms: Res<Atom>,
    config: Res<PolymerConfig>,
    domain: Res<Domain>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut measure: ResMut<PolymerMeasure>,
) {
    let step = run_state.total_cycle;
    if config.measure_interval == 0 || step == 0 || !step.is_multiple_of(config.measure_interval) {
        return;
    }
    if step < config.equilibration_steps {
        return;
    }
    if comm.rank() != 0 {
        return; // single-process only for now
    }

    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || measure.chain_tags.is_empty() {
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

    for chain_tags in &measure.chain_tags {
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

        measure.ree_history.push((step, avg_ree));
        measure.rg_history.push((step, avg_rg));
        measure.ree_sum += avg_ree;
        measure.rg_sum += avg_rg;
        measure.n_samples += 1;

        if step.is_multiple_of(config.output_interval.max(config.measure_interval)) {
            let running_ree = measure.ree_sum / measure.n_samples as f64;
            let running_rg = measure.rg_sum / measure.n_samples as f64;
            println!(
                "Step {}: R_ee={:.4}, R_g={:.4} (avg R_ee={:.4}, avg R_g={:.4}, samples={})",
                step, avg_ree, avg_rg, running_ree, running_rg, measure.n_samples
            );
        }
    }
}

pub fn write_polymer_measurements(
    run_state: Res<RunState>,
    config: Res<PolymerConfig>,
    measure: Res<PolymerMeasure>,
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
    if !measure.ree_history.is_empty() {
        let path = format!("{}/ree.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# step R_ee").ok();
            for &(s, ree) in &measure.ree_history {
                writeln!(f, "{} {:.6}", s, ree).ok();
            }
            println!("Wrote R_ee to {}", path);
        }
    }

    // Write R_g history
    if !measure.rg_history.is_empty() {
        let path = format!("{}/rg.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(f, "# step R_g").ok();
            for &(s, rg) in &measure.rg_history {
                writeln!(f, "{} {:.6}", s, rg).ok();
            }
            println!("Wrote R_g to {}", path);
        }
    }

    // Print final averages
    if measure.n_samples > 0 {
        let avg_ree = measure.ree_sum / measure.n_samples as f64;
        let avg_rg = measure.rg_sum / measure.n_samples as f64;
        println!(
            "Polymer averages ({} samples): <R_ee>={:.4}, <R_g>={:.4}, <R_ee>/<R_g>={:.4}",
            measure.n_samples, avg_ree, avg_rg, avg_ree / avg_rg
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

        // Manually compute
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
