# dem_atom

Per-atom DEM data, material property management, and particle insertion for [MDDEM](https://github.com/SueHeir/MDDEM).

## Module Structure

- `lib.rs` — Core types: `DemAtom`, `MaterialTable`, `DemAtomPlugin`, material config
- `radius.rs` — `RadiusSpec` and `RadiusDistribution` for size distributions
- `insert.rs` — Particle insertion: random, rate-based, and file-based

All public items are re-exported from the crate root via `pub use insert::*` and `pub use radius::*`.

## DemAtom

`DemAtom` stores per-atom DEM-specific fields as flat arrays:
- `radius: Vec<f64>` — particle radius
- `density: Vec<f64>` — particle density
- `inv_inertia: Vec<f64>` — inverse moment of inertia
- `quaternion: Vec<[f64; 4]>` — orientation quaternion
- `omega: Vec<[f64; 3]>` — angular velocity (`#[forward]`)
- `ang_mom: Vec<[f64; 3]>` — angular momentum
- `torque: Vec<[f64; 3]>` — torque (`#[reverse]`, `#[zero]`)

Implements the `AtomData` trait (via `#[derive(AtomData)]`) for automatic MPI pack/unpack, bin-sort reordering, and restart serialization.

## MaterialTable

`MaterialTable` manages named material types with per-material and per-pair precomputed properties:

**Per-material:**
- `youngs_mod`, `poisson_ratio`, `restitution`, `friction`
- `rolling_friction` — rolling resistance coefficient (default 0.0 = disabled)
- `cohesion_energy` — SJKR cohesion energy density in J/m² (default 0.0 = disabled)
- `surface_energy` — JKR surface energy in J/m² (default 0.0 = disabled)
- `twisting_friction` — twisting resistance coefficient (default 0.0 = disabled)
- `kn` — Hooke normal stiffness (default 0.0, used when `contact_model = "hooke"`)
- `kt` — Hooke tangential stiffness (default 0.0, used when `contact_model = "hooke"`)

**Per-pair (geometric-mean mixing):**
- `beta_ij` — damping coefficient from restitution
- `friction_ij` — Coulomb friction coefficient
- `rolling_friction_ij` — rolling resistance coefficient
- `cohesion_energy_ij` — SJKR cohesion
- `surface_energy_ij` — JKR adhesion
- `twisting_friction_ij` — twisting resistance coefficient
- `kn_ij` — Hooke normal stiffness (harmonic mean: `2*ki*kj/(ki+kj)`)
- `kt_ij` — Hooke tangential stiffness (harmonic mean)
- `e_eff_ij` — effective Young's modulus (Hertz)
- `g_eff_ij` — effective shear modulus (Mindlin)

Pair properties use geometric-mean mixing: `mu_ij = sqrt(mu_i * mu_j)`. Hooke stiffnesses use harmonic-mean mixing.

**DemConfig:**
- `contact_model` — `"hertz"` (default) or `"hooke"` — selects the contact model used by `HertzMindlinContactPlugin`.

`cohesion_energy` and `surface_energy` are mutually exclusive on the same material — the plugin exits with an error if both are nonzero.

## Radius Distributions

`RadiusSpec` supports fixed or distribution-based particle radii:

```toml
radius = 0.001                                                            # fixed
radius = { distribution = "uniform", min = 0.0008, max = 0.0012 }        # uniform
radius = { distribution = "gaussian", mean = 0.001, std = 0.0001 }       # Gaussian (clamped positive)
radius = { distribution = "lognormal", mean = 0.001, std = 0.0001 }      # log-normal
radius = { distribution = "discrete", values = [0.001, 0.0015], weights = [0.7, 0.3] }  # weighted discrete
```

## Particle Insertion

`DemAtomInsertPlugin` provides three insertion modes from `[[particles.insert]]` TOML config:

**Random** (default): overlap-free placement within a region or full domain.

**Rate-based**: periodic insertion during simulation runtime, with configurable interval, start/end timesteps, and total particle limit.

**File-based**: read particle data from external files.
- `format = "csv"` — CSV with configurable column mapping
- `format = "lammps_dump"` — LAMMPS dump files with auto-detected column names
- `format = "lammps_data"` — LAMMPS data files with `atom_style` support (`atomic`, `sphere`, `bpm/sphere`)

**Type mapping** (`type_map`): maps integer atom types in files to named materials:
```toml
type_map = { 1 = "glass", 2 = "steel" }
```

## Configuration

```toml
[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4
# rolling_friction = 0.1       # rolling resistance (default 0.0)
# twisting_friction = 0.01     # twisting resistance (default 0.0)
# cohesion_energy = 0.05        # SJKR cohesion J/m² (default 0.0)
# surface_energy = 0.05         # JKR adhesion J/m² (default 0.0)
# kn = 1e5                      # Hooke normal stiffness (for contact_model = "hooke")
# kt = 1e4                      # Hooke tangential stiffness (for contact_model = "hooke")

# Random insertion
[[particles.insert]]
material = "glass"
count = 100
radius = 0.001
density = 2500.0
# velocity = 0.1
# velocity_x = 0.0
# velocity_y = 0.0
# velocity_z = -1.0
# region = { type = "block", min = [0.0, 0.0, 0.0], max = [1.0, 1.0, 1.0] }

# Rate-based insertion
# [[particles.insert]]
# material = "glass"
# radius = { distribution = "uniform", min = 0.0008, max = 0.0012 }
# density = 2500.0
# rate = 10
# rate_interval = 100
# rate_start = 0
# rate_end = 500000
# rate_limit = 5000

# File-based insertion (CSV)
# [[particles.insert]]
# source = "file"
# file = "particles.csv"
# format = "csv"
# material = "glass"
# density = 2500.0
# columns = { x = 0, y = 1, z = 2, radius = 3, atom_type = 4 }
# type_map = { 1 = "glass", 2 = "steel" }

# File-based insertion (LAMMPS data)
# [[particles.insert]]
# source = "file"
# file = "data.lammps"
# format = "lammps_data"
# material = "glass"
# atom_style = "sphere"
# type_map = { 1 = "glass", 2 = "steel" }
```

## Usage

`DemAtomPlugin` is included in `GranularDefaultPlugins`. It registers `DemAtom` with the `AtomDataRegistry`, builds `MaterialTable` from config at plugin build time, and sets `atom.ntypes` from the number of defined materials.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
