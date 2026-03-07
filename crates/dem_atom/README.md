# dem_atom

Per-atom DEM data and material property management for [MDDEM](https://github.com/SueHeir/MDDEM).

## DemAtom

`DemAtom` stores per-atom DEM-specific fields as flat arrays:
- `radius: Vec<f64>` — particle radius
- `density: Vec<f64>` — particle density

Implements the `AtomData` trait for automatic MPI pack/unpack during ghost atom communication.

## MaterialTable

`MaterialTable` manages named material types with per-material and per-pair precomputed properties:
- Per-material: `youngs_mod`, `poisson_ratio`, `restitution`, `friction`
- Per-pair: `beta_ij` (damping coefficient from restitution), `friction_ij` (Coulomb friction coefficient)

Pair properties use geometric-mean mixing (LAMMPS convention): `e_ij = sqrt(e_i * e_j)`, `mu_ij = sqrt(mu_i * mu_j)`.

## Configuration

```toml
[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4

[[dem.materials]]
name = "steel"
youngs_mod = 200e9
poisson_ratio = 0.28
restitution = 0.8
friction = 0.3
```

## Usage

`DemAtomPlugin` is included in `GranularDefaultPlugins`. It registers `DemAtom` with the `AtomDataRegistry` and builds `MaterialTable` from config at plugin build time.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
