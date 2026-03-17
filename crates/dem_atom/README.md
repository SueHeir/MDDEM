# dem_atom — Per-Atom DEM Data & Material Properties

Core per-atom DEM extension data and material property system for MDDEM simulations.

## What It Does

**dem_atom** provides:
- **`DemAtom`** — Per-atom extension data (radius, density, angular velocity, torque) with MPI pack/unpack attributes
- **`MaterialTable`** — Named materials with elastic properties and precomputed per-pair mixing tables for efficient contact evaluation
- **`DemAtomInsertPlugin`** — Particle insertion (random, rate-based, file-based)
- **`RadiusSpec`** — Fixed or statistical particle radius distributions (uniform, gaussian, lognormal, discrete)

## Key Types

**`MaterialTable`** — Material registry with per-pair precomputed properties:
- `add_material()`, `add_material_with_sds()` — Add materials with properties
- `find_material()` — Look up material by name
- `build_pair_tables()` — Compute per-pair mixing tables (geometric/harmonic means)

**`DemAtom`** — Per-atom DEM data (7 fields):
- `radius: Vec<f64>` — Particle radius (m)
- `density: Vec<f64>` — Material density (kg/m³)
- `inv_inertia: Vec<f64>` — Inverse moment of inertia (1/(kg·m²))
- `quaternion: Vec<[f64; 4]>` — Orientation [w, x, y, z]
- `omega: Vec<[f64; 3]>` — Angular velocity (rad/s) `#[forward]`
- `ang_mom: Vec<[f64; 3]>` — Angular momentum (kg·m²/s)
- `torque: Vec<[f64; 3]>` — Torque (N·m) `#[reverse]` `#[zero]`

**`RadiusSpec`** — Fixed or distribution-based radius.

## TOML Configuration

```toml
[dem]
contact_model = "hertz"        # "hertz" or "hooke"
adhesion_model = "jkr"         # "jkr" or "dmt"

[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4                 # default: 0.4
rolling_friction = 0.1         # default: 0.0
cohesion_energy = 0.0          # default: 0.0
surface_energy = 0.0           # default: 0.0
twisting_friction = 0.0        # default: 0.0
kn = 0.0                       # Hooke normal stiffness (default: 0.0)
kt = 0.0                       # Hooke tangential stiffness (default: 0.0)
```

## Usage Example

```rust
use dem_atom::MaterialTable;

let mut table = MaterialTable::new();
let glass = table.add_material("glass", 8.7e9, 0.3, 0.95, 0.4, 0.1, 0.0);
let steel = table.add_material("steel", 200e9, 0.28, 0.8, 0.3, 0.05, 0.0);
table.build_pair_tables();

// Access per-pair properties:
let friction_ij = table.friction_ij[glass as usize][steel as usize];
let e_eff = table.e_eff_ij[glass as usize][steel as usize];
```

## Mixing Rules

Per-pair properties computed via:
- **Geometric mean** — friction, restitution, rolling/twisting friction, cohesion/surface energy
- **Harmonic mean** — Hooke stiffnesses and SDS spring stiffnesses
- **Effective moduli** — Hertz (E*) and Mindlin (G*) contact models
