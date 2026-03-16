# MDDEM Configuration Reference

Complete reference for all TOML configuration sections, fields, types, defaults, and units.

> **Tip:** Run any example with `--generate-config` to print default values for all registered plugins.

---

## Table of Contents

- [Core Infrastructure](#core-infrastructure)
  - [`[comm]`](#comm)
  - [`[domain]`](#domain)
  - [`[neighbor]`](#neighbor)
  - [`[run]` / `[[run]]`](#run)
  - [`[[group]]`](#group)
- [DEM Granular](#dem-granular)
  - [`[dem]`](#dem)
  - [`[[dem.materials]]`](#demmaterials)
  - [`[[particles.insert]]`](#particlesinsert)
  - [`[[wall]]`](#wall)
  - [`[gravity]`](#gravity)
  - [`[thermal]`](#thermal)
- [Fixes](#fixes)
  - [`[[addforce]]`](#addforce)
  - [`[[setforce]]`](#setforce)
  - [`[[freeze]]`](#freeze)
  - [`[[move_linear]]`](#move_linear)
  - [`[[viscous]]`](#viscous)
- [Molecular Dynamics](#molecular-dynamics)
  - [`[lj]`](#lj)
  - [`[lattice]`](#lattice)
  - [`[thermostat]`](#thermostat)
  - [`[langevin]`](#langevin)
  - [`[md_bond]`](#md_bond)
  - [`[polymer_init]`](#polymer_init)
  - [`[polymer]`](#polymer)
  - [`[chain_stats]`](#chain_stats)
- [Measurement & Output](#measurement--output)
  - [`[measure]`](#measure)
  - [`[[type_rdf]]`](#type_rdf)
  - [`[[measure_plane]]`](#measure_plane)
  - [`[dump]`](#dump)
  - [`[vtp]`](#vtp)
  - [`[restart]`](#restart)
  - [`[velocity_distribution]`](#velocity_distribution)
- [Energy Minimization](#energy-minimization)
  - [`[fire]`](#fire)
- [DEM Bonds](#dem-bonds)
  - [`[bonds]`](#bonds)

---

## Core Infrastructure

### `[comm]`

MPI processor layout. Product of processors must equal total MPI ranks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `processors_x` | integer | `1` | MPI grid size in x |
| `processors_y` | integer | `1` | MPI grid size in y |
| `processors_z` | integer | `1` | MPI grid size in z |

### `[domain]`

Simulation box geometry and boundary conditions.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `x_low` | float | `0.0` | m (DEM) / σ (MD) | Box lower x bound |
| `x_high` | float | `1.0` | m / σ | Box upper x bound |
| `y_low` | float | `0.0` | m / σ | Box lower y bound |
| `y_high` | float | `1.0` | m / σ | Box upper y bound |
| `z_low` | float | `0.0` | m / σ | Box lower z bound |
| `z_high` | float | `1.0` | m / σ | Box upper z bound |
| `periodic_x` | bool | `false` | — | Periodic boundary in x |
| `periodic_y` | bool | `false` | — | Periodic boundary in y |
| `periodic_z` | bool | `false` | — | Periodic boundary in z |

### `[neighbor]`

Neighbor list rebuild and spatial binning settings.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `skin_fraction` | float | `1.0` | — | Multiplier on pairwise cutoff for neighbor skin. Larger = fewer rebuilds, more pairs per step. |
| `bin_size` | float | `1.0` | m / σ | Spatial bin width. Should be ≥ largest particle diameter. |
| `every` | integer | `0` | steps | Rebuild every N steps (0 = displacement-based only) |
| `check` | bool | `true` | — | Also check displacement threshold when `every > 0` |
| `sort_every` | integer | `1000` | steps | Sort atoms by spatial bin for cache locality (0 = disabled) |
| `rebuild_on_pbc_wrap` | bool | `false` | — | Force rebuild on PBC crossing (required for DEM) |

### `[run]` / `[[run]]`

Simulation run settings. Use `[run]` for single-stage or `[[run]]` array for multi-stage.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `name` | string | — | — | Stage name (required for multi-stage) |
| `steps` | integer | `1000` | steps | Number of timesteps |
| `thermo` | integer | `100` | steps | Thermo output interval |
| `dt` | float | auto | s (DEM) / reduced (MD) | Timestep (auto-calculated for DEM if omitted) |
| `dump_interval` | integer | — | steps | Dump file write interval |
| `restart_interval` | integer | — | steps | Restart file write interval |

Multi-stage example with per-stage overrides:
```toml
[[run]]
name = "settling"
steps = 10000
thermo = 100

[[run]]
name = "production"
steps = 50000
thermo = 500
thermostat.temperature = 1.2  # override thermostat for this stage
```

### `[[group]]`

Named atom groups for selective force application and measurements.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Group name (referenced by fixes) |
| `type` | integer | — | Select atoms by type index |
| `region` | string | — | Select atoms by region name |

---

## DEM Granular

### `[dem]`

Top-level DEM settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `contact_model` | string | `"hertz"` | Contact model: `"hertz"` or `"hooke"` |
| `materials` | array | — | Array of material definitions (see below) |

### `[[dem.materials]]`

Material properties for DEM particles and walls. Define one block per material.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `name` | string | required | — | Unique material name |
| `youngs_mod` | float | required | Pa | Young's modulus |
| `poisson_ratio` | float | required | — | Poisson's ratio (0–0.5) |
| `restitution` | float | required | — | Coefficient of restitution (0–1) |
| `friction` | float | `0.4` | — | Sliding (Coulomb) friction coefficient |
| `rolling_friction` | float | `0.0` | — | Rolling friction coefficient (0 = disabled) |
| `cohesion_energy` | float | `0.0` | J/m² | SJKR cohesion energy density (0 = disabled) |
| `surface_energy` | float | `0.0` | J/m² | JKR surface energy (0 = disabled) |
| `twisting_friction` | float | `0.0` | — | Twisting friction coefficient (0 = disabled) |
| `kn` | float | `0.0` | N/m | Linear normal stiffness for Hooke model |
| `kt` | float | `0.0` | N/m | Linear tangential stiffness for Hooke model |

### `[[particles.insert]]`

Particle insertion configuration. Supports random, rate-based, and file-based insertion.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `material` | string | required | — | Material name from `[[dem.materials]]` |
| `count` | integer | — | — | Number of particles to insert immediately |
| `radius` | float or table | — | m | Particle radius or size distribution (see below) |
| `density` | float | — | kg/m³ | Particle density |
| `velocity` | float | `0.0` | m/s | Random initial velocity magnitude |
| `rate` | integer | — | particles | Insertion rate (enables rate-based mode) |
| `rate_interval` | integer | `1` | steps | Insert `rate` particles every N steps |
| `source` | string | — | — | `"file"` for file-based insertion |
| `format` | string | — | — | `"csv"`, `"lammps_dump"`, or `"lammps_data"` |
| `file` | string | — | — | Path to particle data file |

**Size distributions** (use instead of scalar `radius`):
```toml
# Uniform distribution
radius = { distribution = "uniform", min = 0.0005, max = 0.0015 }

# Gaussian distribution
radius = { distribution = "gaussian", mean = 0.001, std = 0.0002 }

# Log-normal distribution
radius = { distribution = "lognormal", mean = 0.001, std = 0.0002 }

# Discrete sizes with weights
radius = { distribution = "discrete", values = [0.0005, 0.001, 0.002], weights = [0.2, 0.5, 0.3] }
```

### `[[wall]]`

Wall boundaries for DEM simulations. Supports plane, cylinder, and sphere geometries.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `type` | string | `"plane"` | — | `"plane"`, `"cylinder"`, or `"sphere"` |
| `material` | string | required | — | Material name from `[[dem.materials]]` |
| `name` | string | — | — | Optional name for runtime wall control |
| **Plane fields:** | | | | |
| `normal_x/y/z` | float | `0.0` | — | Wall normal direction |
| `point_x/y/z` | float | `0.0` | m | A point on the wall plane |
| **Cylinder fields:** | | | | |
| `center` | [float, float] | required | m | Center coordinates (2D) |
| `radius` | float | required | m | Cylinder radius |
| `axis` | string | `"z"` | — | Cylinder axis (`"x"`, `"y"`, or `"z"`) |
| **Motion fields:** | | | | |
| `velocity` | float | `0.0` | m/s | Constant wall velocity |
| `oscillate` | table | — | — | `{amplitude, frequency}` for oscillation |
| `servo` | table | — | — | `{target_force, max_velocity, gain}` for servo control |

### `[gravity]`

Constant gravitational acceleration (requires `GravityPlugin`).

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `gx` | float | `0.0` | m/s² | Gravity x-component |
| `gy` | float | `0.0` | m/s² | Gravity y-component |
| `gz` | float | `0.0` | m/s² | Gravity z-component |

### `[thermal]`

DEM heat conduction between contacting particles (requires `ThermalPlugin`).

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `conductivity` | float | `1.0` | W/(m·K) | Thermal conductivity |
| `specific_heat` | float | `500.0` | J/(kg·K) | Specific heat capacity |
| `initial_temperature` | float | `300.0` | K | Initial particle temperature |

---

## Fixes

All fix sections are arrays (`[[...]]`). Each entry requires a `group` field matching a `[[group]]` name.

### `[[addforce]]`
Add constant force to group atoms every step.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `group` | string | required | — | Group name |
| `fx/fy/fz` | float | `0.0` | N (DEM) / reduced (MD) | Force components |

### `[[setforce]]`
Override force on group atoms every step.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `group` | string | required | — | Group name |
| `fx/fy/fz` | float | `0.0` | N / reduced | Force components |

### `[[freeze]]`
Zero velocity and force on group atoms (frozen particles).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `group` | string | required | Group name |

### `[[move_linear]]`
Set constant velocity on group atoms.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `group` | string | required | — | Group name |
| `vx/vy/vz` | float | `0.0` | m/s / reduced | Velocity components |

### `[[viscous]]`
Apply viscous damping force (F = -γv) to group atoms.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `group` | string | required | — | Group name |
| `gamma` | float | required | kg/s | Damping coefficient |

---

## Molecular Dynamics

### `[lj]`

Lennard-Jones 12-6 pair potential (requires `LJForcePlugin`).

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `epsilon` | float | `1.0` | ε | Well depth |
| `sigma` | float | `1.0` | σ | Particle diameter |
| `cutoff` | float | `2.5` | σ | Cutoff distance |
| `mixing` | string | — | — | `"geometric"` or `"arithmetic"` for multi-type |
| `types` | array | — | — | Per-type LJ parameters (see below) |
| `pair_coeffs` | array | — | — | Explicit pair coefficient overrides |

Multi-type example:
```toml
[lj]
cutoff = 2.5
mixing = "geometric"

[[lj.types]]
epsilon = 1.0
sigma = 1.0

[[lj.types]]
epsilon = 0.5
sigma = 0.8
```

### `[lattice]`

FCC lattice initialization (requires `LatticePlugin`).

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `style` | string | `"fcc"` | — | Lattice style |
| `scale` | float | `1.0` | σ | Lattice constant |
| `mass` | float | `1.0` | m | Atom mass |
| `temperature` | float | `0.0` | T* | Initial temperature for Maxwell-Boltzmann velocities |

### `[thermostat]`

Nosé-Hoover NVT thermostat with fused Velocity Verlet integration.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `temperature` | float | `0.85` | T* (reduced) | Target temperature |
| `damping` | float | `100.0` | dt | Damping parameter (in timestep units) |
| `group` | string | `"all"` | — | Group to thermostat |

### `[langevin]`

Langevin thermostat with stochastic dynamics.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `temperature` | float | `1.0` | T* | Target temperature |
| `damping` | float | `1.0` | τ | Damping time |
| `seed` | integer | `12345` | — | Random number seed |

### `[md_bond]`

Harmonic bond potential for MD simulations.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `k` | float | required | ε/σ² | Spring constant |
| `r0` | float | required | σ | Equilibrium bond length |

### `[polymer_init]`

Polymer chain initialization.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `num_chains` | integer | required | — | Number of polymer chains |
| `chain_length` | integer | required | — | Monomers per chain |
| `bond_length` | float | required | σ | Initial bond length |
| `mass` | float | `1.0` | m | Monomer mass |
| `temperature` | float | `1.0` | T* | Initial temperature |

### `[polymer]`

FENE bond potential for polymer simulations.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `k_fene` | float | `30.0` | ε/σ² | FENE spring constant |
| `r0_fene` | float | `1.5` | σ | Maximum FENE bond extension |

### `[chain_stats]`

End-to-end distance and radius of gyration measurements.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `interval` | integer | `1000` | steps | Measurement interval |
| `file` | string | `"chain_stats.csv"` | — | Output file path |

---

## Measurement & Output

### `[measure]`

RDF and MSD measurements for MD simulations.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `rdf_bins` | integer | `200` | — | Number of RDF histogram bins |
| `rdf_max` | float | — | σ | Maximum RDF distance (default: half box) |
| `rdf_interval` | integer | `100` | steps | RDF accumulation interval |
| `msd_interval` | integer | `10` | steps | MSD accumulation interval |

### `[[type_rdf]]`

Type-filtered radial distribution function.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type_i` | integer | required | First atom type index |
| `type_j` | integer | required | Second atom type index |
| `bins` | integer | `200` | Number of histogram bins |
| `max` | float | — | Maximum distance |

### `[[measure_plane]]`

Measurement planes for DEM mass/momentum flux.

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `name` | string | required | — | Plane identifier |
| `axis` | string | required | — | Normal axis: `"x"`, `"y"`, or `"z"` |
| `position` | float | required | m | Plane position along axis |
| `interval` | integer | `100` | steps | Measurement interval |

### `[dump]`

Atom dump file output settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `style` | string | `"custom"` | Dump style |
| `fields` | array | `["id","type","x","y","z"]` | Fields to output |

### `[vtp]`

VTK/VTP output for ParaView visualization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable VTP output |

### `[restart]`

Restart file settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | string | `"restart"` | Restart file prefix |

### `[velocity_distribution]`

Velocity distribution analysis output.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bins` | integer | `100` | Number of histogram bins |
| `interval` | integer | `1000` | Sampling interval (steps) |

---

## Energy Minimization

### `[fire]`

FIRE (Fast Inertial Relaxation Engine) energy minimizer.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dt_max` | float | `0.01` | Maximum timestep |
| `f_tol` | float | `1e-8` | Force convergence tolerance |
| `e_tol` | float | `1e-10` | Energy convergence tolerance |
| `max_steps` | integer | `10000` | Maximum minimization steps |

---

## DEM Bonds

### `[bonds]`

DEM bond model (parallel bond / BPM style).

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `style` | string | `"bpm"` | — | Bond style |
| `file` | string | — | — | Bond definition file path |
| `radius_ratio` | float | `0.5` | — | Bond radius as fraction of particle radius |

---

## Plugin Dependencies

Some plugins require other plugins to be registered first. MDDEM will print a clear error if dependencies are missing:

| Plugin | Requires |
|--------|----------|
| `HertzMindlinContactPlugin` | `DemAtomPlugin` |
| `WallPlugin` | `DemAtomPlugin` |
| `LJForcePlugin` | `NeighborPlugin` |
| `ThermalPlugin` | `DemAtomPlugin`, `NeighborPlugin` |

The `CorePlugins` group includes `NeighborPlugin`, and `GranularDefaultPlugins` includes `DemAtomPlugin`, so most users won't encounter these errors when using plugin groups.
