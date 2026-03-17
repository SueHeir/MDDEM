# dem_clump

Multisphere/clump rigid body composites for non-spherical DEM particles.

## Overview

A **clump** is a rigid body composed of multiple overlapping spheres. Each sphere participates in contact detection independently, but all contact forces are aggregated to a single parent atom (the clump center-of-mass). The parent integrates both translational and rotational equations of motion, while sub-sphere positions and velocities are derived from the parent's state and orientation.

## Key Types

- **`ClumpDef`**: Configuration struct defining a named clump type with multiple spheres
- **`ClumpSphereConfig`**: Specifies a sphere's offset and radius within the clump body frame
- **`ClumpAtom`**: Per-atom data tracking clump membership, parent/sub-sphere status, and body-frame offsets
- **`ClumpRegistry`**: Runtime registry of loaded clump definitions from config
- **`ClumpPlugin`**: Main plugin that registers systems for force aggregation and position updates

## Key Functions

- **`compute_clump_inertia()`**: Calculates total mass and moment of inertia via parallel axis theorem
- **`insert_clump()`**: Inserts a clump as one parent + N sub-spheres into the simulation
- **`same_clump()`**: Checks if two atoms belong to the same clump (for contact exclusion)
- **`quat_rotate()`**: Rotates vectors by quaternion for position/velocity updates

## TOML Configuration

Define clump types in `[[clump.definitions]]`:

```toml
[[clump.definitions]]
name = "dimer"
spheres = [
    { offset = [-0.0003, 0.0, 0.0], radius = 0.001 },
    { offset = [0.0003, 0.0, 0.0], radius = 0.001 },
]
```

Reference clumps during particle insertion in `[[particles.insert]]`:

```toml
[[particles.insert]]
clump = "dimer"
material = "glass"
count = 100
density = 2500.0
region = { type = "block", min = [0, 0, 0.01], max = [0.01, 0.01, 0.02] }
```

## Usage Example

```rust
use dem_clump::{ClumpPlugin, ClumpRegistry};
use mddem_app::App;

let mut app = App::new();
app.add_plugin(ClumpPlugin);
// ... configure materials, contact models, etc.
app.run();
```

The plugin automatically:
- Loads clump definitions from `clump.definitions` config
- Registers per-atom clump data
- Aggregates forces from sub-spheres to parents
- Updates sub-sphere positions/velocities after parent integration
