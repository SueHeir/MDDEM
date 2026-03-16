# mddem_core

Core simulation infrastructure for [MDDEM](https://github.com/SueHeir/MDDEM). Provides the foundational resource types, plugins, and systems that all MDDEM simulations depend on.

## Modules

- **`atom`** — `Atom` struct with SoA (struct-of-arrays) layout: flat `Vec<f64>` for positions, velocities, forces, torques, angular velocities, angular momenta, plus per-atom mass, tag, type, `ntypes` (global type count), and flags. `AtomDataRegistry` enables physics crates to register additional per-atom data (e.g., `DemAtom` for radius/density) with automatic MPI pack/unpack, bin-sort reordering (`apply_permutation`), and generic restart serialization. Use `#[derive(AtomData)]` from `mddem_derive` to generate trait implementations.
- **`comm`** — `CommBackend` trait abstracting MPI communication. `CommunicationPlugin` provides full 3D domain decomposition with ghost atom exchange, border forwarding, and reverse force accumulation. `SingleProcessCommPlugin` is a drop-in replacement for non-MPI builds.
- **`domain`** — `Domain` resource for simulation box boundaries and periodicity. `DomainDecomposition` trait with `CartesianDecomposition` default. `DomainPlugin` reads `[domain]` config and decomposes the box across MPI ranks.
- **`input`** — TOML config loading via `Config` resource. `Config::load::<T>(app, key)` deserializes a TOML section and adds it as an `App` resource. All config structs use `#[serde(deny_unknown_fields)]` to reject typos at startup. `InputPlugin` handles CLI parsing and banner printing.
- **`pair_coeff`** — `PairCoeffTable<T>` generic NxN symmetric storage for per-type-pair coefficients. `MixingRule` enum (`Geometric`, `Arithmetic`) for combining per-type parameters. Used by `md_lj` for multi-type LJ simulations and available for any pair potential.
- **`region`** — `Region` enum with serde-tagged deserialization for spatial primitives: `Block`, `Sphere`, `Cylinder`, `Plane`, `Union`, `Intersect`. Methods: `contains()` for point-in-region tests, `random_point_inside()` for uniform sampling. `Union` matches if any child region contains the point; `Intersect` matches if all child regions contain the point. Used by groups and particle insertion.
- **`run`** — `RunPlugin` for run/cycle management. Supports single-stage `[run]` and multi-stage `[[run]]` configs with per-stage step counts, output intervals, and per-stage config overrides (e.g., `gravity.gz = -981.0`). Named stages are validated against `StageEnum` variants when `StageAdvancePlugin` is active.
- **`virial`** — `VirialStress` full symmetric stress tensor (xx, yy, zz, xy, xz, yz) shared across all force types. `VirialStressPlugin` guards against double-registration so multiple force plugins (LJ, bond, contact) can each add it safely.

## Features

- `mpi_backend` (default) — enables MPI communication via [rsmpi](https://github.com/rsmpi/rsmpi). Disable with `--no-default-features` for single-process builds.

## Usage

`mddem_core` is typically not used directly. Instead, use the [`mddem`](https://crates.io/crates/mddem) umbrella crate with `CorePlugins`:

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
