# mddem_test_utils

Shared test utilities for MDDEM crates, providing quick setup helpers for unit tests.

## Purpose

This crate eliminates boilerplate when setting up test fixtures. Each helper function creates a minimal, valid object so tests can focus on logic rather than initialization.

## Key Functions

- **`make_atoms(n)`** — Create `n` test atoms arranged along the x-axis at positions `(0,0,0)`, `(1,0,0)`, etc., each with radius 0.5 and mass 1.0.

- **`make_group_registry(name, mask)`** — Create a group with the given name; `mask` is a `Vec<bool>` indicating atom membership.

- **`make_single_comm()`** — Create a single-process communication resource (no MPI required).

- **`push_dem_test_atom(atom, dem, tag, pos, radius)`** — Add a DEM particle with full fields (mass from `ρ=2500 kg/m³`, inertia, quaternion, omega, etc.) to parallel `Atom` and `DemAtom` structures.

- **`make_material_table()`** — Create a single "glass" material with realistic properties (E=8.7 GPa, ν=0.3, e=0.95, μ=0.4). Pair tables are pre-built.

## Usage Example

```rust
use mddem_test_utils::{make_atoms, make_group_registry, make_single_comm};
use mddem_test_utils::{push_dem_test_atom, make_material_table};

// MD/shared tests
let atoms = make_atoms(3);
let groups = make_group_registry("all", vec![true, true, true]);
let comm = make_single_comm();

// DEM tests
let mut atom = mddem_core::Atom::new();
let mut dem = dem_atom::DemAtom::default();
push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], 0.5);

let materials = make_material_table();
```
