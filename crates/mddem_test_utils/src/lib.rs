//! Shared test utilities for all MDDEM crates.
//!
//! This crate provides helper functions to quickly set up [`Atom`], [`GroupRegistry`],
//! [`CommResource`], [`DemAtom`](dem_atom::DemAtom), and [`MaterialTable`](dem_atom::MaterialTable)
//! instances for use in unit tests. Each helper produces a minimal, valid object so tests can
//! focus on the logic under test rather than boilerplate setup.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use mddem_test_utils::{make_atoms, make_group_registry, make_single_comm};
//!
//! let atom = make_atoms(3); // 3 atoms at (0,0,0), (1,0,0), (2,0,0)
//! let groups = make_group_registry("all", vec![true, true, true]);
//! let comm = make_single_comm();
//! ```
//!
//! For DEM-specific tests, use [`push_dem_test_atom`] and [`make_material_table`]:
//!
//! ```rust,ignore
//! use mddem_test_utils::{push_dem_test_atom, make_material_table};
//!
//! let mut atom = mddem_core::Atom::new();
//! let mut dem = dem_atom::DemAtom::default();
//! push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], 0.5);
//!
//! let materials = make_material_table(); // single "glass" material
//! ```

use mddem_core::group::{Group, GroupDef};
use mddem_core::{Atom, CommResource, GroupRegistry, SingleProcessComm};

/// Create an [`Atom`] with `n` test atoms arranged along the x-axis.
///
/// Each atom `i` is placed at position `(i, 0, 0)` with radius `0.5`, mass `1.0`,
/// and a timestep of `0.001`. Atom tags are `0..n`.
///
/// # Examples
///
/// ```rust,ignore
/// use mddem_test_utils::make_atoms;
///
/// let atom = make_atoms(5);
/// assert_eq!(atom.nlocal, 5);
/// assert_eq!(atom.natoms, 5);
/// assert_eq!(atom.position[2], [2.0, 0.0, 0.0]);
/// ```
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

/// Create a [`GroupRegistry`] containing a single named group with the given membership mask.
///
/// The `mask` vector should have one entry per atom: `true` means the atom belongs to the
/// group, `false` means it does not. The group count is computed automatically from the mask.
///
/// # Examples
///
/// ```rust,ignore
/// use mddem_test_utils::make_group_registry;
///
/// // Group "mobile" includes atoms 0 and 2, but not atom 1
/// let registry = make_group_registry("mobile", vec![true, false, true]);
/// assert_eq!(registry.groups[0].count, 2);
/// ```
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

/// Create a single-process [`CommResource`] for testing.
///
/// Returns a communication resource backed by [`SingleProcessComm`], which requires no
/// MPI initialization. Suitable for all single-process unit tests.
///
/// # Examples
///
/// ```rust,ignore
/// use mddem_test_utils::make_single_comm;
///
/// let comm = make_single_comm();
/// ```
pub fn make_single_comm() -> CommResource {
    CommResource(Box::new(SingleProcessComm::new()))
}

/// Push a DEM test atom with all [`DemAtom`](dem_atom::DemAtom) fields populated.
///
/// Creates a solid sphere using a density of `2500 kg/m³`. Mass is computed from the
/// given `radius` via `m = ρ · (4/3)πr³`, and the moment of inertia assumes a uniform
/// solid sphere (`I = 0.4 · m · r²`). Rotational fields (quaternion, omega, angular
/// momentum, torque) are initialized to zero/identity defaults.
///
/// Both `atom` and `dem` are extended in parallel — callers must ensure they stay in sync.
///
/// # Examples
///
/// ```rust,ignore
/// use mddem_test_utils::push_dem_test_atom;
///
/// let mut atom = mddem_core::Atom::new();
/// let mut dem = dem_atom::DemAtom::default();
/// push_dem_test_atom(&mut atom, &mut dem, 0, [1.0, 2.0, 3.0], 0.5);
/// assert_eq!(dem.radius[0], 0.5);
/// assert_eq!(dem.density[0], 2500.0);
/// ```
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
    dem.body_id.push(0.0);
}

/// Create a single-material "glass" [`MaterialTable`](dem_atom::MaterialTable) for testing.
///
/// Returns a material table with one material named `"glass"` that has the following
/// properties:
///
/// | Property          | Value   |
/// |-------------------|---------|
/// | Young's modulus   | 8.7 GPa |
/// | Poisson ratio     | 0.3     |
/// | Restitution       | 0.95    |
/// | Friction          | 0.4     |
/// | Rolling friction  | 0.0     |
/// | Cohesion energy   | 0.0     |
///
/// Pair tables are pre-built, so the table is ready for contact computations immediately.
///
/// # Examples
///
/// ```rust,ignore
/// use mddem_test_utils::make_material_table;
///
/// let mt = make_material_table();
/// assert_eq!(mt.material_names().len(), 1);
/// ```
pub fn make_material_table() -> dem_atom::MaterialTable {
    let mut mt = dem_atom::MaterialTable::new();
    mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.0, 0.0);
    mt.build_pair_tables();
    mt
}
