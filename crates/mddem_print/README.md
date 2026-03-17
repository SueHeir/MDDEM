# mddem_print

Output systems for [MDDEM](https://github.com/SueHeir/MDDEM) simulations: periodic console logging, dump files, restart checkpoints, and ParaView visualization.

## Output Systems

- **Thermo** — Periodic console metrics with configurable columns (step, atoms, kinetic energy, temperature, performance).
- **Dump** — Per-atom snapshots in CSV (`text`) or binary format. Fields: tag, type, position, velocity, force, radius, plus custom registered data.
- **VTP** — ParaView-compatible visualization with positions, velocities, and custom per-atom data arrays.
- **Restart** — Full simulation state serialization in bincode or JSON. Automatically saves/restores all `AtomData` extensions.

## Key Types

- `Thermo` — Runtime state for console output. Push custom values via `.set(name, value)`.
- `ThermoConfig`, `DumpConfig`, `RestartConfig`, `VtpConfig` — TOML configuration structs.
- `DumpRegistry` — Plugin API to register scalar/vector per-atom data callbacks for dump and VTP output.

## Configuration

```toml
[thermo]
columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]

[dump]
interval = 5000
format = "text"    # or "binary"

[restart]
interval = 10000
format = "bincode" # or "json"
read = false       # auto-read latest on startup

[vtp]
interval = 500
```

## Usage

`PrintPlugin` is included in `CorePlugins` and registers all output systems automatically. In multi-stage runs, set `save_at_end = true` on a `[[run]]` stage to write dump/restart files when that stage ends.
