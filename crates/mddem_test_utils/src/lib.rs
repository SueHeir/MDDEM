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
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: None,
            region_z_high: None,
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

/// Create a single-material "glass" [`MaterialTable`] for testing.
pub fn make_material_table() -> dem_atom::MaterialTable {
    let mut mt = dem_atom::MaterialTable::new();
    mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
    mt.build_pair_tables();
    mt
}
