# md_thermostat

Nose-Hoover NVT thermostat for [MDDEM](https://github.com/SueHeir/MDDEM): temperature control via extended Lagrangian dynamics.

## Physics

### Nose-Hoover Thermostat (`NoseHooverPlugin`)
Symmetric Liouville splitting that wraps around Velocity Verlet without modifying it:

**Pre-initial integration** (`PreInitialIntegration`):
1. Compute kinetic energy
2. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`
3. Rescale velocities: `v *= exp(-dt/2 * p_xi/Q)`

**Post-final integration** (`PostFinalIntegration`):
1. Rescale velocities: `v *= exp(-dt/2 * p_xi/Q)`
2. Recompute kinetic energy
3. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`

The thermal mass `Q = ndof * T_target * tau^2` where `ndof = 3N - 3` (subtracting center-of-mass degrees of freedom).

## Config

```toml
[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time tau_T
```

## Resources

- `ThermostatConfig` — deserialized config
- `NoseHooverState` — thermostat state: `p_xi`, `Q`, `target_temp`, `ndof`

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
