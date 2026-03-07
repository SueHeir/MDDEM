# dem_atom_insert

DEM particle insertion for [MDDEM](https://github.com/SueHeir/MDDEM). Randomly places particles in the simulation domain with overlap checking, and automatically computes a stable timestep from the Rayleigh wave period.

## Features

- Random placement within the domain or a specified sub-region
- Overlap checking (1.1x radius buffer) to prevent initial interpenetration
- Optional random velocity (Gaussian per component) and/or directional velocity
- Multiple insert blocks with different materials, counts, radii, densities, and velocities
- Automatic timestep calculation: `dt = 0.15 * min(Rayleigh period)` across all particles

## Configuration

```toml
[[particles.insert]]
material = "glass"
count = 500
radius = 0.001
density = 2500.0
velocity = 0.5              # Random RMS velocity (Gaussian, optional)
velocity_x = 0.0            # Directional velocity (additive with random, optional)
velocity_y = 0.0
velocity_z = -1.0
region_x_low = 0.005        # Insertion sub-region (optional, defaults to domain)
region_x_high = 0.020
```

## Usage

`DemAtomInsertPlugin` is included in `GranularDefaultPlugins`. It runs at setup time on rank 0, populating both `Atom` and `DemAtom` fields.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
