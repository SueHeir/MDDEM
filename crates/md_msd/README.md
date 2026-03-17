# md_msd

Per-type mean-squared displacement (MSD) and diffusion coefficient plugin for MDDEM.

## Overview

This plugin measures how far particles travel from their initial positions over time, computing:
- **MSD(t)** = average squared displacement per atom type
- **Diffusion coefficient** D = MSD(t) / (6t) via the Einstein relation

Displacement is computed using **unwrapped coordinates**, which automatically account for periodic boundary crossings.

## Key Types

- **`MsdConfig`** — TOML configuration with `interval` and `output_interval` settings
- **`TypeMsdTracker`** — Per-atom resource that maintains:
  - Reference (initial) and unwrapped positions indexed by atom tag
  - Per-type counts and accumulated MSD values
  - Detects periodic boundary crossings by comparing wrapped displacements

## Configuration

```toml
[msd]
interval = 10           # sample MSD every N steps (default: 10)
output_interval = 1000  # write output files every N steps (default: 1000)
```

Set `interval = 0` to disable MSD tracking entirely.

## Output Files

Written to `<output_dir>/data/`:

- **`msd_all.txt`** — combined time-series with columns:
  ```
  dt_steps  dt_time  MSD_all  D_all  MSD_type0  D_type0  ...
  ```

- **`msd_type_<i>.txt`** — per-type file with columns:
  ```
  dt_steps  dt_time  MSD  D  (type i, N=count)
  ```

## Usage Example

```toml
[msd]
interval = 100
output_interval = 5000

[integrator]
type = "VelocityVerlet"
timestep = 0.001
```

The plugin automatically:
- Detects all atom types in the simulation
- Corrects for periodic boundary wrapping on each step
- Outputs MSD and diffusion coefficients (meaningful in the long-time/linear diffusive regime)
