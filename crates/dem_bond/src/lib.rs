//! Bond force models for MDDEM.
//!
//! Provides [`DemBondPlugin`] which registers the
//! [`BondStore`](mddem_core::BondStore), auto-bonds initially touching
//! particles, and computes normal bond forces (spring + damping).

use std::collections::HashMap;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, Config, VirialStress, VirialStressPlugin};
use mddem_print::Thermo;

/// TOML `[bonds]` configuration section.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct BondConfig {
    /// Auto-bond initially touching particles (default: false).
    #[serde(default)]
    pub auto_bond: bool,
    /// Multiplier on sum of radii for auto-bond distance check (default: 1.001).
    #[serde(default = "default_bond_tolerance")]
    pub bond_tolerance: f64,
    /// Normal spring stiffness k_n (N/m). Default: 0.
    #[serde(default)]
    pub normal_stiffness: f64,
    /// Normal damping coefficient gamma_n. Default: 0.
    #[serde(default)]
    pub normal_damping: f64,
}

fn default_bond_tolerance() -> f64 {
    1.001
}

impl Default for BondConfig {
    fn default() -> Self {
        BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 0.0,
            normal_damping: 0.0,
        }
    }
}

/// Accumulated bond metrics for strain output.
#[derive(Default)]
pub struct BondMetrics {
    pub strain_sum: f64,
    pub bond_count: usize,
}

/// Plugin that enables bond support in an MDDEM simulation.
///
/// Registers the bond topology data structure ([`BondStore`](mddem_core::BondStore))
/// via [`BondPlugin`](mddem_core::BondPlugin), auto-bonds touching particles at
/// setup, and computes normal bond forces each timestep.
pub struct DemBondPlugin;

impl Plugin for DemBondPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[bonds]
# auto_bond = false
# bond_tolerance = 1.001
# normal_stiffness = 0.0
# normal_damping = 0.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_plugins(mddem_core::BondPlugin);
        app.add_plugins(VirialStressPlugin);
        Config::load::<BondConfig>(app, "bonds");
        app.add_resource(BondMetrics::default());
        app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
        app.add_update_system(zero_bond_metrics, ScheduleSet::PreForce);
        app.add_update_system(bond_normal_force.label("dem_bond_force"), ScheduleSet::Force);
        app.add_update_system(output_bond_metrics, ScheduleSet::PostForce);
    }
}

/// Auto-bond initially touching particles at setup time.
///
/// Runs at `PostSetup` (after all atoms are inserted). For each pair of local
/// atoms within `(r_i + r_j) * tolerance`, creates symmetric bond entries.
pub fn auto_bond_touching(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    comm: Res<CommResource>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }
    if !bond_config.auto_bond {
        return;
    }

    let dem = registry.expect::<DemAtom>("auto_bond_touching");
    let mut bond_store = registry.expect_mut::<BondStore>("auto_bond_touching");

    let nlocal = atoms.nlocal as usize;

    // Ensure bond storage covers all local atoms
    while bond_store.bonds.len() < nlocal {
        bond_store.bonds.push(Vec::new());
    }

    let tolerance = bond_config.bond_tolerance;
    let mut bond_count = 0u64;

    for i in 0..nlocal {
        for j in (i + 1)..nlocal {
            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            let sum_radii = dem.radius[i] + dem.radius[j];
            if dist <= sum_radii * tolerance {
                // Create symmetric bond entries
                bond_store.bonds[i].push(BondEntry {
                    partner_tag: atoms.tag[j],
                    bond_type: 0,
                    r0: dist,
                });
                bond_store.bonds[j].push(BondEntry {
                    partner_tag: atoms.tag[i],
                    bond_type: 0,
                    r0: dist,
                });
                bond_count += 1;
            }
        }
    }

    if comm.rank() == 0 {
        println!("DemBond: auto-bonded {} pairs", bond_count);
    }
}

/// Normal bond force: spring + damping along the bond axis.
///
/// For each local atom's bond list, looks up the partner by tag, computes the
/// stretch/compression relative to `r0`, and applies a spring-damper force.
/// Forces are applied with Newton's 3rd law (each bond processed once).
pub fn bond_normal_force(
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    mut metrics: ResMut<BondMetrics>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let bond_store = registry.get::<BondStore>();
    let bonds = match bond_store {
        Some(ref b) => b,
        None => return,
    };

    let k_n = bond_config.normal_stiffness;
    let gamma_n = bond_config.normal_damping;

    if k_n == 0.0 && gamma_n == 0.0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    if bonds.bonds.len() < nlocal {
        return;
    }

    // Build tag → index lookup for all atoms (local + ghost)
    let mut tag_to_index: HashMap<u32, usize> = HashMap::with_capacity(atoms.len());
    for idx in 0..atoms.len() {
        tag_to_index.insert(atoms.tag[idx], idx);
    }

    for i in 0..nlocal {
        for bond in &bonds.bonds[i] {
            let j = match tag_to_index.get(&bond.partner_tag) {
                Some(&idx) => idx,
                None => continue, // partner not present (ghost not available)
            };

            // Process each bond once: only when i's tag < partner's tag
            if atoms.tag[i] >= bond.partner_tag {
                continue;
            }

            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < 1e-20 {
                continue;
            }

            // Unit normal from i to j
            let nx = dx / dist;
            let ny = dy / dist;
            let nz = dz / dist;

            // Stretch: positive = stretched, negative = compressed
            let delta = dist - bond.r0;

            // Normal spring force magnitude (positive = attractive when stretched)
            let f_spring = k_n * delta;

            // Relative velocity projected onto bond axis (j relative to i)
            let dvx = atoms.vel[j][0] - atoms.vel[i][0];
            let dvy = atoms.vel[j][1] - atoms.vel[i][1];
            let dvz = atoms.vel[j][2] - atoms.vel[i][2];
            let v_n = dvx * nx + dvy * ny + dvz * nz;

            // Damping force (opposes relative motion along bond)
            let f_damp = gamma_n * v_n;

            // Total force magnitude along bond axis
            // Positive total → force pulls i toward j (attractive)
            let f_total = f_spring + f_damp;

            // Force vector from i toward j
            let fx = f_total * nx;
            let fy = f_total * ny;
            let fz = f_total * nz;

            // Newton's 3rd law: i pulled toward j, j pulled toward i
            atoms.force[i][0] += fx;
            atoms.force[i][1] += fy;
            atoms.force[i][2] += fz;
            atoms.force[j][0] -= fx;
            atoms.force[j][1] -= fy;
            atoms.force[j][2] -= fz;

            // Accumulate virial stress tensor
            if let Some(ref mut v) = virial {
                if v.active {
                    v.add_pair(dx, dy, dz, fx, fy, fz);
                }
            }

            // Accumulate bond metrics
            metrics.strain_sum += delta / bond.r0;
            metrics.bond_count += 1;
        }
    }
}

/// Reset bond metrics to zero before force computation.
pub fn zero_bond_metrics(mut metrics: ResMut<BondMetrics>) {
    metrics.strain_sum = 0.0;
    metrics.bond_count = 0;
}

/// Output bond metrics to thermo after force computation.
pub fn output_bond_metrics(
    metrics: Res<BondMetrics>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
) {
    let strain_sum = comm.all_reduce_sum_f64(metrics.strain_sum);
    let bond_count = comm.all_reduce_sum_f64(metrics.bond_count as f64);

    if bond_count > 0.0 {
        let avg_strain = strain_sum / bond_count;
        thermo.set("bond_strain", avg_strain);
    } else {
        thermo.set("bond_strain", 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use mddem_core::{Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, SingleProcessComm};
    use mddem_print::Thermo;

    fn push_test_atom(
        atom: &mut Atom,
        dem: &mut DemAtom,
        tag: u32,
        pos: [f64; 3],
        radius: f64,
    ) {
        let mass = 2500.0 * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        atom.push_test_atom(tag, pos, radius, mass);
        dem.radius.push(radius);
        dem.density.push(2500.0);
        dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
    }

    #[test]
    fn auto_bond_creates_symmetric_bonds() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Two touching particles along x-axis (distance = 2*radius = 0.002)
        push_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            2,
            [0.002, 0.0, 0.0],
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(BondStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: true,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
        });
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(SchedulerManager::default());
        app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
        app.organize_systems();
        app.setup();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        assert_eq!(bonds.bonds.len(), 2);
        assert_eq!(bonds.bonds[0].len(), 1, "atom 0 should have 1 bond");
        assert_eq!(bonds.bonds[1].len(), 1, "atom 1 should have 1 bond");
        assert_eq!(bonds.bonds[0][0].partner_tag, 2);
        assert_eq!(bonds.bonds[1][0].partner_tag, 1);
        assert!((bonds.bonds[0][0].r0 - 0.002).abs() < 1e-10);
    }

    #[test]
    fn auto_bond_skips_separated_atoms() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Two far-apart particles
        push_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            2,
            [0.01, 0.0, 0.0],
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(BondStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: true,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
        });
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(SchedulerManager::default());
        app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
        app.organize_systems();
        app.setup();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        assert_eq!(bonds.bonds[0].len(), 0, "no bonds for separated atoms");
        assert_eq!(bonds.bonds[1].len(), 0, "no bonds for separated atoms");
    }

    #[test]
    fn bond_force_attracts_stretched_pair() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Two particles at distance 0.0025, bonded with r0 = 0.002 (stretched)
        push_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            2,
            [0.0025, 0.0, 0.0],
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 2,
            bond_type: 0,
            r0: 0.002,
        }]);
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 1,
            bond_type: 0,
            r0: 0.002,
        }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_normal_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Atom 0 should be pulled in +x (toward atom 1)
        assert!(
            atom.force[0][0] > 0.0,
            "stretched bond should attract atom 0 toward atom 1, got {}",
            atom.force[0][0]
        );
        // Atom 1 should be pulled in -x (toward atom 0)
        assert!(
            atom.force[1][0] < 0.0,
            "stretched bond should attract atom 1 toward atom 0, got {}",
            atom.force[1][0]
        );
        // Newton's 3rd law
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "forces should be equal and opposite"
        );
    }

    #[test]
    fn bond_force_repels_compressed_pair() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Two particles at distance 0.0015, bonded with r0 = 0.002 (compressed)
        push_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            2,
            [0.0015, 0.0, 0.0],
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 2,
            bond_type: 0,
            r0: 0.002,
        }]);
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 1,
            bond_type: 0,
            r0: 0.002,
        }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_normal_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Atom 0 should be pushed in -x (away from atom 1)
        assert!(
            atom.force[0][0] < 0.0,
            "compressed bond should repel atom 0 away from atom 1, got {}",
            atom.force[0][0]
        );
        // Atom 1 should be pushed in +x (away from atom 0)
        assert!(
            atom.force[1][0] > 0.0,
            "compressed bond should repel atom 1 away from atom 0, got {}",
            atom.force[1][0]
        );
    }

    #[test]
    fn bond_force_zero_at_equilibrium() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Two particles exactly at r0
        push_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            2,
            [0.002, 0.0, 0.0],
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 2,
            bond_type: 0,
            r0: 0.002,
        }]);
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 1,
            bond_type: 0,
            r0: 0.002,
        }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_normal_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force[0][0].abs() < 1e-10,
            "no force at equilibrium, got {}",
            atom.force[0][0]
        );
        assert!(
            atom.force[1][0].abs() < 1e-10,
            "no force at equilibrium, got {}",
            atom.force[1][0]
        );
    }
}
