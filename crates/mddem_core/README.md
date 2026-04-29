# mddem_core

Core simulation infrastructure for MDDEM: particle storage, MPI domain decomposition, and spatial regions. TOML loading, the multi-stage run loop, the App/Plugin framework, and the MPI abstraction live in the [grass](https://github.com/elizabeth-suehr/grass) workspace; this crate re-exports them so existing `use mddem_core::{Config, RunPlugin, ...}` keeps working.

## What It Does

Provides foundational systems for particle-based simulations:

- **Per-atom storage** (`Atom`, `AtomData` trait): struct-of-arrays with extensible fields (DEM radius, angular velocity, etc.)
- **Domain decomposition & boundaries**: box geometry, MPI splitting, periodic/fixed/shrink-wrap conditions
- **Communication wiring**: MPI ghost/exchange systems on top of `grass_mpi::CommBackend` (single-process fallback included)
- **Spatial regions**: Box, Sphere, Cylinder, Plane, Union, Intersect with point/sample tests
- **Re-exports from grass:** `Config`, `InputPlugin`, `RunPlugin`, `RunConfig`, `StageConfig`, `StageOverrides`, `ScheduleSetupSet`, etc.

## Key Types

| Type | Purpose | Defined in |
|------|---------|-----------|
| `Atom` | Core per-atom fields in struct-of-arrays layout | `mddem_core` |
| `AtomData` | Trait to register plugin-specific data (e.g., `DemAtom`) | `mddem_core` |
| `AtomDataRegistry` | Manages extensions with MPI pack/unpack | `mddem_core` |
| `Domain` | Box geometry, bounds, periodicity | `mddem_core` |
| `Region` | Spatial primitives for groups and insertion | `mddem_core` |
| `CommBackend` | Abstraction over MPI or serial communication | `grass_mpi` (re-exported) |
| `Config` | TOML table with typed deserialization | `grass_io` (re-exported) |
| `RunConfig` / `StageConfig` | Multi-stage run + per-stage overrides | `grass_io` (re-exported) |

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
