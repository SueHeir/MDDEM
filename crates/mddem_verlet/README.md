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

`VelocityVerletPlugin` is included in `CorePlugins`. It registers two systems — `initial_integration` and `final_integration` — at the appropriate schedule sets.

Rotational integration (quaternion-based angular Velocity Verlet) is provided separately by `dem_granular::RotationalDynamicsPlugin`.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
