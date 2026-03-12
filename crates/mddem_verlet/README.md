# mddem_verlet

Velocity Verlet translational integration for [MDDEM](https://github.com/SueHeir/MDDEM) simulations.

## Algorithm

Standard two-half-step Velocity Verlet:

1. **Initial integration** (`ScheduleSet::InitialIntegration`):
   - `v += 0.5 * dt * F / m`
   - `x += v * dt`

2. *(Forces are computed between these steps)*

3. **Final integration** (`ScheduleSet::FinalIntegration`):
   - `v += 0.5 * dt * F / m`

This is time-reversible, symplectic, and second-order accurate.

## Usage

`VelocityVerletPlugin` registers two systems — `initial_integration` and `final_integration` — at the appropriate schedule sets. Add it explicitly when your simulation does not use a thermostat that provides fused integration (e.g., `NoseHooverPlugin` fuses velocity rescaling with Verlet integration internally).

`VelocityVerletPlugin` is included in `GranularDefaultPlugins`. For MD simulations without a thermostat, add it manually:

```rust
app.add_plugins(CorePlugins)
    .add_plugins(VelocityVerletPlugin);
```

Rotational integration (quaternion-based angular Velocity Verlet) is provided separately by `dem_granular::RotationalDynamicsPlugin`.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
