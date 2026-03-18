# md_lattice

FCC lattice initialization with Maxwell-Boltzmann velocities for [MDDEM](https://github.com/SueHeir/MDDEM).

## What it does

Populates the simulation domain with atoms on a face-centered cubic (FCC) lattice and assigns each atom a velocity drawn from the Maxwell-Boltzmann distribution at a specified temperature. Velocities are automatically corrected to remove center-of-mass drift and rescaled to match the target temperature exactly.

## Key features

- **FCC lattice**: 4 atoms per unit cell at positions (0,0,0), (½,½,0), (½,0,½), (0,½,½)
- **Automatic fitting**: Lattice constant adjusted per-dimension so cells tile the domain exactly
- **Proper velocity initialization**: Samples from `N(0, σ_v)` where `σ_v = √(T/m)`, removes COM drift, rescales to exact T
- **Multi-type support**: Optional per-type masses and cumulative type fractions for mixed systems
- **Standard timestep**: Automatically sets `dt = 0.005` (LJ reduced units)

## Key types

- **`LatticeConfig`**: TOML configuration struct with density, temperature, mass, skin distance
- **`LatticePlugin`**: ECS plugin that registers `fcc_insert` and `lattice_set_dt` systems

## TOML configuration

```toml
[lattice]
style = "fcc"           # Lattice type (only "fcc" supported)
density = 0.85          # Number density ρ* (atoms per unit volume)
temperature = 0.85      # Initial temperature T* for velocity sampling
mass = 1.0              # Default atom mass (LJ reduced units)
skin = 1.25             # Neighbor-list skin distance

# Optional: multi-type systems
# type_fractions = [0.8, 1.0]   # Cumulative fractions → 80% type 0, 20% type 1
# type_masses = [1.0, 2.0]      # Per-type masses (overrides `mass`)
```

## Usage example

```rust
use sim_app::prelude::*;

let mut app = App::new();
app.add_plugins(md_lattice::LatticePlugin);
app.setup();
```

The plugin automatically reads `[lattice]` config and runs on the first stage (rank 0 only). Atoms are then redistributed to other ranks during communication setup.
