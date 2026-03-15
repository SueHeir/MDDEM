//! Shared test utilities for MDDEM crates.

use mddem_core::group::{Group, GroupDef};
use mddem_core::{Atom, CommResource, GroupRegistry, SingleProcessComm};

/// Create an `Atom` with `n` test atoms at positions (i, 0, 0), mass 1.0, radius 0.5.
pub fn make_atoms(n: usize) -> Atom {
    let mut atom = Atom::new();
    for i in 0..n {
        atom.push_test_atom(i as u32, [i as f64, 0.0, 0.0], 0.5, 1.0);
    }
    atom.nlocal = n as u32;
    atom.natoms = n as u64;
    atom.dt = 0.001;
    atom
}

/// Create a `GroupRegistry` with a single named group and the given mask.
pub fn make_group_registry(name: &str, mask: Vec<bool>) -> GroupRegistry {
    let count = mask.iter().filter(|&&m| m).count();
    let mut registry = GroupRegistry::new();
    registry.groups.push(Group {
        name: name.to_string(),
        def: GroupDef {
            name: name.to_string(),
            atom_types: None,
            region: None,
            dynamic: None,
        },
        mask,
        count,
    });
    registry
}

/// Create a single-process `CommResource` for testing.
pub fn make_single_comm() -> CommResource {
    CommResource(Box::new(SingleProcessComm::new()))
}

/// Push a DEM test atom with all `DemAtom` fields populated.
///
/// Creates a solid sphere with `density = 2500`, computes mass from radius,
/// and fills all rotational fields with defaults.
pub fn push_dem_test_atom(
    atom: &mut Atom,
    dem: &mut dem_atom::DemAtom,
    tag: u32,
    pos: [f64; 3],
    radius: f64,
) {
    let density = 2500.0;
    let mass = density * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    atom.push_test_atom(tag, pos, radius, mass);
    dem.radius.push(radius);
    dem.density.push(density);
    dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
    dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
    dem.omega.push([0.0; 3]);
    dem.ang_mom.push([0.0; 3]);
    dem.torque.push([0.0; 3]);
}

/// Create a single-material "glass" [`MaterialTable`] for testing.
pub fn make_material_table() -> dem_atom::MaterialTable {
    let mut mt = dem_atom::MaterialTable::new();
    mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
    mt.build_pair_tables();
    mt
}
