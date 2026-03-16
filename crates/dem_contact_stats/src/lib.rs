//! Per-atom coordination number and contact statistics for DEM particles.
//!
//! Computes the number of contacts (coordination number) for each particle
//! by checking neighbor pair overlaps after forces are computed. Optionally
//! identifies rattler particles (< 4 contacts in 3D).
//!
//! Enable via `[contact_stats]` in TOML config. Without the section, the
//! AtomData is registered but no systems run (zero overhead).

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_print::{DumpRegistry, Thermo};
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, Config};
use mddem_neighbor::Neighbor;

// ── Config ──────────────────────────────────────────────────────────────────

/// TOML `[contact_stats]` — contact statistics configuration.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ContactStatsConfig {
    /// Enable rattler detection (particles with < 4 contacts in 3D).
    #[serde(default)]
    pub rattlers: bool,
}

impl Default for ContactStatsConfig {
    fn default() -> Self {
        ContactStatsConfig { rattlers: false }
    }
}

// ── Per-atom contact data ───────────────────────────────────────────────────

/// Per-atom contact statistics: coordination number (contact count).
#[derive(AtomData)]
pub struct ContactStats {
    /// Per-atom coordination number (number of contacting neighbors).
    #[forward]
    pub coordination: Vec<f64>,
}

impl Default for ContactStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ContactStats {
    pub fn new() -> Self {
        ContactStats {
            coordination: Vec::new(),
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers per-atom contact statistics and coordination number computation.
pub struct ContactStatsPlugin;

impl Plugin for ContactStatsPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [contact_stats]
# rattlers = false  # Identify rattler particles (< 4 contacts in 3D)"#,
        )
    }

    fn build(&self, app: &mut App) {
        register_atom_data!(app, ContactStats::new());

        let has_config = {
            let config = app
                .get_resource_ref::<Config>()
                .expect("Config must exist");
            config.table.get("contact_stats").is_some()
        };

        if !has_config {
            app.add_resource(ContactStatsConfig::default());
            return;
        }

        let stats_config = Config::load::<ContactStatsConfig>(app, "contact_stats");
        app.add_resource(stats_config);

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.add_update_system(push_coordination_thermo, ScheduleSet::PostFinalIntegration);
        app.add_setup_system(register_dump_columns, ScheduleSetupSet::PostSetup);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Register coordination as a per-atom dump column for ParaView visualization.
fn register_dump_columns(mut dump_registry: ResMut<DumpRegistry>) {
    dump_registry.register_scalar("coordination", |atoms, registry| {
        let stats = registry.expect::<ContactStats>("dump coordination");
        let nlocal = atoms.nlocal as usize;
        stats.coordination[..nlocal].to_vec()
    });
}

/// Compute per-atom coordination number by checking neighbor pair overlaps.
fn compute_coordination(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
) {
    let dem = registry.expect::<DemAtom>("compute_coordination");
    let mut stats = registry.expect_mut::<ContactStats>("compute_coordination");

    let n = atoms.len();

    // Ensure coordination vector covers all atoms (local + ghost).
    while stats.coordination.len() < n {
        stats.coordination.push(0.0);
    }

    // Zero coordination for all atoms.
    let nlocal = atoms.nlocal as usize;
    for c in stats.coordination[..n].iter_mut() {
        *c = 0.0;
    }

    // Loop neighbor pairs, check for overlap.
    for (i, j) in neighbor.pairs(nlocal) {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];
        let sum_r = r1 + r2;

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq >= sum_r * sum_r {
            continue;
        }

        let distance = dist_sq.sqrt();
        let delta = sum_r - distance;
        if delta > 0.0 {
            stats.coordination[i] += 1.0;
            stats.coordination[j] += 1.0;
        }
    }
}

/// Push coordination statistics to thermo output.
fn push_coordination_thermo(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    config: Res<ContactStatsConfig>,
    mut thermo: ResMut<Thermo>,
) {
    let stats = registry.expect::<ContactStats>("push_coordination_thermo");
    let nlocal = atoms.nlocal as usize;

    if nlocal == 0 {
        thermo.set("coord_avg", 0.0);
        thermo.set("coord_max", 0.0);
        thermo.set("coord_min", 0.0);
        if config.rattlers {
            thermo.set("n_rattlers", 0.0);
            thermo.set("rattler_fraction", 0.0);
        }
        return;
    }

    let mut sum = 0.0;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    let mut n_rattlers = 0usize;

    for i in 0..nlocal {
        let c = stats.coordination[i];
        sum += c;
        if c > max {
            max = c;
        }
        if c < min {
            min = c;
        }
        // Rattler: particle with < 4 contacts (in 3D, needs d+1 = 4 contacts for stability).
        if c < 4.0 {
            n_rattlers += 1;
        }
    }

    let avg = sum / nlocal as f64;

    thermo.set("coord_avg", avg);
    thermo.set("coord_max", max);
    thermo.set("coord_min", min);

    if config.rattlers {
        thermo.set("n_rattlers", n_rattlers as f64);
        thermo.set("rattler_fraction", n_rattlers as f64 / nlocal as f64);
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

    /// Helper: create atoms and neighbor list for testing coordination.
    fn setup_atoms(
        positions: &[[f64; 3]],
        radius: f64,
        neighbor_pairs: &[(u32, u32)],
    ) -> App {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        atom.dt = 1e-7;

        for (idx, &pos) in positions.iter().enumerate() {
            push_dem_test_atom(&mut atom, &mut dem, idx as u32, pos, radius);
        }
        let n = positions.len();
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        let mut stats = ContactStats::new();
        for _ in 0..n {
            stats.coordination.push(0.0);
        }

        // Build CSR neighbor list from pairs.
        let mut neighbor = Neighbor::new();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(a, b) in neighbor_pairs {
            adj[a as usize].push(b);
        }
        let mut offsets = vec![0u32; n + 1];
        let mut indices = Vec::new();
        for i in 0..n {
            for &j in &adj[i] {
                indices.push(j);
            }
            offsets[i + 1] = indices.len() as u32;
        }
        neighbor.neighbor_offsets = offsets;
        neighbor.neighbor_indices = indices;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(stats);

        let config = ContactStatsConfig { rattlers: true };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(config);
        app.add_resource(Thermo::new());

        app
    }

    #[test]
    fn two_touching_particles_coord_1() {
        let radius = 0.001;
        let sep = 0.0019; // overlap: 2*0.001 - 0.0019 = 0.0001 > 0
        let app = &mut setup_atoms(
            &[[0.0, 0.0, 0.0], [sep, 0.0, 0.0]],
            radius,
            &[(0, 1)],
        );

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let stats = registry.expect::<ContactStats>("test");
        assert_eq!(stats.coordination[0], 1.0, "atom 0 should have coord=1");
        assert_eq!(stats.coordination[1], 1.0, "atom 1 should have coord=1");
    }

    #[test]
    fn isolated_particle_coord_0() {
        let radius = 0.001;
        let sep = 0.003; // no overlap: 2*0.001 - 0.003 = -0.001 < 0
        let app = &mut setup_atoms(
            &[[0.0, 0.0, 0.0], [sep, 0.0, 0.0]],
            radius,
            &[(0, 1)],
        );

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let stats = registry.expect::<ContactStats>("test");
        assert_eq!(stats.coordination[0], 0.0, "atom 0 should have coord=0");
        assert_eq!(stats.coordination[1], 0.0, "atom 1 should have coord=0");
    }

    #[test]
    fn particle_touching_four_neighbors_coord_4() {
        let radius = 0.001;
        let sep = 0.0019; // overlap
        // Central particle at origin, 4 neighbors along +x, -x, +y, -y
        let positions = vec![
            [0.0, 0.0, 0.0],        // central
            [sep, 0.0, 0.0],         // +x
            [-sep, 0.0, 0.0],        // -x
            [0.0, sep, 0.0],         // +y
            [0.0, -sep, 0.0],        // -y
        ];
        // Neighbor pairs: central (0) has neighbors 1,2,3,4
        let pairs = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let app = &mut setup_atoms(&positions, radius, &pairs);

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let stats = registry.expect::<ContactStats>("test");
        assert_eq!(stats.coordination[0], 4.0, "central atom should have coord=4");
        // Each outer atom touches only the central one
        for i in 1..5 {
            assert_eq!(stats.coordination[i], 1.0, "outer atom {} should have coord=1", i);
        }
    }

    #[test]
    fn no_overlap_means_coord_zero() {
        let radius = 0.001;
        let sep = 0.002; // exactly touching: delta = 2*0.001 - 0.002 = 0.0, not > 0
        let app = &mut setup_atoms(
            &[[0.0, 0.0, 0.0], [sep, 0.0, 0.0]],
            radius,
            &[(0, 1)],
        );

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let stats = registry.expect::<ContactStats>("test");
        assert_eq!(stats.coordination[0], 0.0, "exactly touching = no overlap = coord 0");
        assert_eq!(stats.coordination[1], 0.0, "exactly touching = no overlap = coord 0");
    }

    #[test]
    fn thermo_values_correct() {
        let radius = 0.001;
        let sep = 0.0019;
        // 3 particles: 0 touches 1 and 2, 1 touches 0, 2 touches 0
        let positions = vec![
            [0.0, 0.0, 0.0],
            [sep, 0.0, 0.0],
            [0.0, sep, 0.0],
        ];
        let pairs = vec![(0, 1), (0, 2)];
        let app = &mut setup_atoms(&positions, radius, &pairs);

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.add_update_system(push_coordination_thermo, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let thermo = app.get_resource_ref::<Thermo>().unwrap();
        let avg = *thermo.values.get("coord_avg").unwrap();
        let max = *thermo.values.get("coord_max").unwrap();
        let min = *thermo.values.get("coord_min").unwrap();

        // Coords: [2, 1, 1] → avg = 4/3, max = 2, min = 1
        assert!((avg - 4.0 / 3.0).abs() < 1e-10, "avg should be 4/3, got {}", avg);
        assert_eq!(max, 2.0, "max should be 2");
        assert_eq!(min, 1.0, "min should be 1");

        // All 3 particles have < 4 contacts → all rattlers
        let n_rattlers = *thermo.values.get("n_rattlers").unwrap();
        let rattler_frac = *thermo.values.get("rattler_fraction").unwrap();
        assert_eq!(n_rattlers, 3.0, "all 3 are rattlers");
        assert!((rattler_frac - 1.0).abs() < 1e-10, "rattler fraction should be 1.0");
    }

    #[test]
    fn non_rattler_with_4_contacts() {
        let radius = 0.001;
        let sep = 0.0019;
        // Central particle with 4 contacts → not a rattler
        let positions = vec![
            [0.0, 0.0, 0.0],
            [sep, 0.0, 0.0],
            [-sep, 0.0, 0.0],
            [0.0, sep, 0.0],
            [0.0, -sep, 0.0],
        ];
        let pairs = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let app = &mut setup_atoms(&positions, radius, &pairs);

        app.add_update_system(compute_coordination, ScheduleSet::PostForce);
        app.add_update_system(push_coordination_thermo, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let thermo = app.get_resource_ref::<Thermo>().unwrap();
        let n_rattlers = *thermo.values.get("n_rattlers").unwrap();
        // Central atom has 4 contacts → not a rattler. Outer atoms have 1 → rattlers.
        assert_eq!(n_rattlers, 4.0, "4 outer atoms are rattlers, central is not");
    }
}
