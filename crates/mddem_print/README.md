# mddem_print

Output systems for [MDDEM](https://github.com/SueHeir/MDDEM) simulations: thermo logging, VTP visualization, granular temperature, dump files, and restart files.

## Output Types

- **Thermo** — LAMMPS-style periodic console output: step, atom count, kinetic energy, neighbor count, wall time, steps/s.
- **VTP** — ParaView-compatible `.vtp` files with per-atom positions, radii, velocity magnitudes, ghost/collision flags. Visualize with ParaView's Point Gaussian representation.
- **Granular Temperature** — Time series of granular temperature `T_g = (1/3N) * sum(m * v^2)` written to `data/GranularTemp.txt`.
- **Dump** — Per-atom data in CSV (`text`) or binary format. Fields: tag, type, x, y, z, vx, vy, vz, fx, fy, fz, radius.
- **Restart** — Full simulation state in bincode or JSON format. Supports reading restart files to resume simulations.

## Configuration

```toml
[vtp]
interval = 1000        # Write VTP every N steps (0 = disabled)

[dump]
interval = 5000        # Write dump every N steps (0 = disabled)
format = "text"        # "text" (CSV) or "binary"

[restart]
interval = 10000       # Write restart every N steps (0 = disabled)
format = "bincode"     # "bincode" or "json"
read = false           # Read restart file on startup
```

Per-stage overrides are supported in multi-stage `[[run]]` configs via `dump_interval`, `restart_interval`, and `vtp_interval`.

## Usage

`PrintPlugin` is included in `CorePlugins` and registers all output systems automatically.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
