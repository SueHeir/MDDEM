# mddem_print

Output systems for [MDDEM](https://github.com/SueHeir/MDDEM) simulations: thermo logging, VTP visualization, dump files, and restart files.

## Output Types

- **Thermo** — LAMMPS-style periodic console output with configurable columns. Default columns: `step`, `atoms`, `ke`, `neighbors`, `walltime`, `stepps` (steps/s).
- **VTP** — ParaView-compatible `.vtp` files with per-atom positions, velocity magnitudes, and ghost/collision flags. Supports registered custom data arrays via `DumpRegistry`.
- **Dump** — Per-atom data in CSV (`text`) or binary format. Fields: tag, type, x, y, z, vx, vy, vz, fx, fy, fz, radius, plus any registered custom columns from `DumpRegistry`.
- **Restart** — Full simulation state in bincode or JSON format. Generic serialization — any registered `AtomData` extension (e.g., DemAtom, ContactHistoryStore) is automatically saved and restored. On read, the latest restart file is auto-detected by scanning step numbers in filenames.

Note: Granular temperature output (`GranularTempPlugin`) has moved to the `dem_granular` crate and is included in `GranularDefaultPlugins`.

## Thermo Columns

### Built-in columns
- `step` — current timestep
- `atoms` — total atom count
- `ke` — total kinetic energy
- `temp` — temperature: `T = 2*KE / (3*N - 3)`
- `neighbors` — neighbor list size
- `walltime` — elapsed wall-clock time
- `stepps` — steps per second (performance)

### Virial columns
When `VirialStress` is present (added by LJ, contact, or bond force plugins): `virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`, `virial_yz`. These are MPI-reduced and auto-populated at each thermo interval.

### Group-filtered columns
Append `/groupname` to filter by a named group:
- `ke/mobile` — kinetic energy of atoms in group "mobile"
- `temp/mobile` — temperature of group "mobile"
- `atoms/mobile` — atom count in group "mobile"

### User-defined columns
Any plugin can push custom values via `thermo.set("name", value)`, then add `"name"` to the TOML column list. Columns not set print `N/A`.

## DumpRegistry

`DumpRegistry` allows plugins to register per-atom data callbacks for dump and VTP output. Callbacks are only invoked on output steps — zero overhead otherwise.

```rust
// In a plugin's build():
let dump_reg = app.get_resource_mut::<DumpRegistry>().unwrap();
dump_reg.register_scalar("pressure", |atoms, registry| {
    // Return Vec<f64> of length atoms.nlocal
    (0..atoms.nlocal as usize).map(|_| 0.0).collect()
});
dump_reg.register_vector("angular_vel", |atoms, registry| {
    // Return Vec<[f64; 3]> of length atoms.nlocal
    (0..atoms.nlocal as usize).map(|_| [0.0; 3]).collect()
});
```

Scalar callbacks add one CSV column and one VTP `<DataArray>`. Vector callbacks add three CSV columns (`name_x`, `name_y`, `name_z`) and one 3-component VTP `<DataArray>`.

## Stage End Saves

In multi-stage simulations, setting `save_at_end = true` on a `[[run]]` stage writes both a dump and restart file when that stage finishes (or when stage advancement is triggered).

## Configuration

```toml
[thermo]
columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]

[vtp]
interval = 1000        # Write VTP every N steps (0 = disabled)

[dump]
interval = 5000        # Write dump every N steps (0 = disabled)
format = "text"        # "text" (CSV) or "binary"

[restart]
interval = 10000       # Write restart every N steps (0 = disabled)
format = "bincode"     # "bincode" or "json"
read = false           # Read latest restart file on startup
```

Per-stage overrides are supported in multi-stage `[[run]]` configs via `dump_interval`, `restart_interval`, and `vtp_interval`.

## Usage

`PrintPlugin` is included in `CorePlugins` and registers all output systems automatically.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
