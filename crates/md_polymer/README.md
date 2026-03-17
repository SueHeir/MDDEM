# md_polymer

Polymer chain initialization and chain statistics (R_ee, R_g) for MDDEM.

## Overview

This crate provides two independent plugins for MD simulations of bead-spring polymer chains:

- **`PolymerInitPlugin`** — Creates polymer chains with configurable topology (random walk or straight line) and initializes bonds and velocities.
- **`ChainStatsPlugin`** — Measures end-to-end distance (R_ee) and radius of gyration (R_g) for any bonded linear chains. Auto-discovers chain topology from the bond store if not populated by init.
- **`PolymerPlugin`** — Convenience plugin that adds both.

## Key Types

- **`PolymerInitConfig`** — TOML `[polymer_init]` configuration for chain creation.
- **`ChainStatsConfig`** — TOML `[chain_stats]` configuration for statistics measurement.
- **`ChainStatsData`** — Per-chain measurements: time series of R_ee and R_g with cumulative averages.

## Configuration

### Unified config (backward compatible):

```toml
[polymer]
n_chains = 5
chain_length = 100
bond_length = 0.97
mass = 1.0
init_style = "random_walk"  # or "straight"
seed = 42
temperature = 1.0
measure_interval = 100
output_interval = 1000
```

### Separate configs (recommended):

```toml
[polymer_init]
n_chains = 5
chain_length = 100
bond_length = 0.97
mass = 1.0
init_style = "random_walk"
seed = 42
temperature = 1.0

[chain_stats]
measure_interval = 100
output_interval = 1000
equilibration_steps = 0
```

## Usage Example

```rust
use mddem_app::prelude::*;
use md_polymer::PolymerPlugin;

let mut app = App::new();
app.add_plugins(PolymerPlugin);
app.run();
```

## Measurements

- **R_ee**: Euclidean distance between first and last bead (minimum-image wrapped).
- **R_g**: Root-mean-square distance of beads from chain center of mass.

For ideal freely-jointed chains: ⟨R_ee²⟩ = N·b² and ⟨R_ee²⟩ / ⟨R_g²⟩ = 6.

Output files: `ree.txt` and `rg.txt` in `data/` subdirectory.
