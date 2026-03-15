# mddem_print

Output systems for [MDDEM](https://github.com/SueHeir/MDDEM) simulations: thermo logging, VTP visualization, granular temperature, dump files, and restart files.

## Output Types

- **Thermo** — LAMMPS-style periodic console output: step, atom count, kinetic energy, neighbor count, wall time, steps/s. When `VirialStress` is present, virial tensor components (`virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`, `virial_yz`) are available as thermo columns. User-defined values via `thermo.set(name, value)`.
- **VTP** — ParaView-compatible `.vtp` files with per-atom positions, velocity magnitudes, and ghost/collision flags. Visualize with ParaView's Point Gaussian representation. Supports registered custom data arrays via `DumpRegistry`.
- **Dump** — Per-atom data in CSV (`text`) or binary format. Fields: tag, type, x, y, z, vx, vy, vz, fx, fy, fz, radius, plus any registered custom columns from `DumpRegistry`.
- **Restart** — Full simulation state in bincode or JSON format. Generic serialization — any registered `AtomData` extension (e.g., DemAtom, ContactHistoryStore) is automatically saved and restored.

Note: Granular temperature output (`GranularTempPlugin`) has moved to the `dem_granular` crate and is included in `GranularDefaultPlugins`.

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
