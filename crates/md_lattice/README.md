# md_lattice

FCC lattice initialization for [MDDEM](https://github.com/SueHeir/MDDEM): places atoms on a face-centered cubic lattice with Maxwell-Boltzmann velocities.

## Physics

### FCC Lattice Insertion (`LatticePlugin`)
- Lattice constant `a = (4/rho)^(1/3)` (4 atoms per FCC unit cell)
- Adjusted per-dimension to fit the simulation box exactly
- 4 FCC basis positions per cell: (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
- Maxwell-Boltzmann velocity initialization with COM drift removal
- Velocities rescaled to exact target temperature

### Timestep
Sets `dt = 0.005` in LJ reduced units (standard for LJ simulations).

## Config

```toml
[lattice]
style = "fcc"
density = 0.85       # number density rho*
temperature = 0.85   # initial T* for Maxwell-Boltzmann velocities
mass = 1.0           # particle mass
skin = 1.25          # neighbor skin distance
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();
```

This plugin replaces `DemAtomInsertPlugin` for LJ simulations. It adds `AtomPlugin` automatically if not already present.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
