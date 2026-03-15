# dem_atom

Per-atom DEM data, material property management, and particle insertion for [MDDEM](https://github.com/SueHeir/MDDEM).

## DemAtom

`DemAtom` stores per-atom DEM-specific fields as flat arrays:
- `radius: Vec<f64>` — particle radius
- `density: Vec<f64>` — particle density
- `inv_inertia: Vec<f64>` — inverse moment of inertia
- `quaternion: Vec<[f64; 4]>` — orientation quaternion
- `omega: Vec<[f64; 3]>` — angular velocity
- `ang_mom: Vec<[f64; 3]>` — angular momentum
- `torque: Vec<[f64; 3]>` — torque

Implements the `AtomData` trait (via `#[derive(AtomData)]`) for automatic MPI pack/unpack, bin-sort reordering, and restart serialization.

## MaterialTable

`MaterialTable` manages named material types with per-material and per-pair precomputed properties:
- Per-material: `youngs_mod`, `poisson_ratio`, `restitution`, `friction`
- Per-pair: `beta_ij` (damping coefficient from restitution), `friction_ij` (Coulomb friction coefficient), `e_eff_ij` (effective Young's modulus), `g_eff_ij` (effective shear modulus)

Pair properties use geometric-mean mixing (LAMMPS convention): `e_ij = sqrt(e_i * e_j)`, `mu_ij = sqrt(mu_i * mu_j)`.

## Particle Insertion

`DemAtomInsertPlugin` inserts particles at setup time from `[[particles.insert]]` TOML config blocks. Features:
- Random overlap-free placement within a region (or the full domain)
- Per-insert random velocity (Gaussian) and/or directional velocity
- Automatic Rayleigh timestep estimation from material properties

## Configuration

```toml
[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4

[[particles.insert]]
material = "glass"
count = 100
radius = 0.001
density = 2500.0
# velocity = 0.1
# region = { type = "block", min = [0.0, 0.0, 0.0], max = [1.0, 1.0, 1.0] }
```

## Usage

`DemAtomPlugin` is included in `GranularDefaultPlugins`. It registers `DemAtom` with the `AtomDataRegistry`, builds `MaterialTable` from config at plugin build time, and sets `atom.ntypes` from the number of defined materials.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
