# md_thermostat

Nose-Hoover NVT thermostat for [MDDEM](https://github.com/SueHeir/MDDEM): temperature control via extended Lagrangian dynamics.

## Physics

### Nose-Hoover Thermostat (`NoseHooverPlugin`)
Symmetric Liouville splitting with Velocity Verlet integration fused into the thermostat systems to reduce array passes per timestep:

**Pre-initial integration** (`PreInitialIntegration`):
1. Compute kinetic energy
2. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`
3. Fused loop: rescale velocities `v *= exp(-dt/2 * p_xi/Q)`, then half-kick `v += 0.5*dt*F/m`, then drift `x += v*dt`

**Post-final integration** (`PostFinalIntegration`):
1. Fused loop: half-kick `v += 0.5*dt*F/m`, then rescale velocities `v *= exp(-dt/2 * p_xi/Q)`
2. Recompute kinetic energy
3. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`

The thermal mass `Q = ndof * T_target * tau^2` where `ndof = 3N - 3` (subtracting center-of-mass degrees of freedom).

Because `NoseHooverPlugin` includes Velocity Verlet integration, `VelocityVerletPlugin` should **not** be added separately when using this thermostat.

## Config

```toml
[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time tau_T
```

## Resources

- `ThermostatConfig` — deserialized config
- `NoseHooverState` — thermostat state: `p_xi`, `q_mass`, `target_temp`, `ndof`, `group_name`

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
