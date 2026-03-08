# mddem_core

Core simulation infrastructure for [MDDEM](https://github.com/SueHeir/MDDEM). Provides the foundational resource types, plugins, and systems that all MDDEM simulations depend on.

## Modules

- **`atom`** — `Atom` struct with SoA (struct-of-arrays) layout: flat `Vec<f64>` for positions, velocities, forces, torques, angular velocities, angular momenta, plus per-atom mass, tag, type, and flags. `AtomDataRegistry` enables physics crates to register additional per-atom data (e.g., `DemAtom` for radius/density) with automatic MPI pack/unpack, bin-sort reordering (`apply_permutation`), and generic restart serialization. Use `#[derive(AtomData)]` from `mddem_derive` to generate trait implementations.
- **`comm`** — `CommBackend` trait abstracting MPI communication. `CommunicationPlugin` provides full 3D domain decomposition with ghost atom exchange, border forwarding, and reverse force accumulation. `SingleProcessCommPlugin` is a drop-in replacement for non-MPI builds.
- **`domain`** — `Domain` resource for simulation box boundaries and periodicity. `DomainDecomposition` trait with `CartesianDecomposition` default. `DomainPlugin` reads `[domain]` config and decomposes the box across MPI ranks.
- **`input`** — TOML config loading via `Config` resource. `Config::load::<T>(app, key)` deserializes a TOML section and adds it as an `App` resource. All config structs use `#[serde(deny_unknown_fields)]` to reject typos at startup. `InputPlugin` handles CLI parsing and banner printing.
- **`run`** — `RunPlugin` for run/cycle management. Supports single-stage `[run]` and multi-stage `[[run]]` configs with per-stage step counts and output intervals.

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
