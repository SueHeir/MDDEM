# mddem_core

Core simulation infrastructure for MDDEM: particle storage, MPI domain decomposition, TOML config, and run control.

## What It Does

Provides foundational systems for particle-based simulations:

- **Per-atom storage** (`Atom`, `AtomData` trait): struct-of-arrays with extensible fields (DEM radius, angular velocity, etc.)
- **Domain decomposition & boundaries**: box geometry, MPI splitting, periodic/fixed/shrink-wrap conditions
- **Communication**: MPI ghost/exchange and single-process fallback
- **Configuration**: TOML parsing with multi-stage runs and per-stage overrides
- **Spatial regions**: Box, Sphere, Cylinder, Plane, Union, Intersect with point/sample tests

## Key Types

| Type | Purpose |
|------|---------|
| `Atom` | Core per-atom fields in struct-of-arrays layout |
| `AtomData` | Trait to register plugin-specific data (e.g., `DemAtom`) |
| `AtomDataRegistry` | Manages extensions with MPI pack/unpack |
| `Domain` | Box geometry, bounds, periodicity |
| `CommBackend` | Abstraction over MPI or serial communication |
| `Config` | TOML table with typed deserialization |
| `Region` | Spatial primitives for groups and insertion |
| `StageConfig` | Per-stage settings and config overrides |

## Quick Start

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
   .add_plugins(GranularDefaultPlugins);
app.run();
```

DEM/MD-specific types register via plugins. Use `#[derive(AtomData)]` to extend `Atom` with custom fields.

## Features

- `mpi_backend` (default): Enable MPI; disable for serial-only builds
