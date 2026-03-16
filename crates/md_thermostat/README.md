# md_thermostat

Thermostats for [MDDEM](https://github.com/SueHeir/MDDEM): Nose-Hoover NVT (extended Lagrangian) and Langevin (stochastic friction + random force).

## Nose-Hoover Thermostat (`NoseHooverPlugin`)

Symmetric Liouville splitting with Velocity Verlet integration fused into the thermostat systems to reduce array passes per timestep.

**Pre-initial integration** (`PreInitialIntegration`):
1. Compute kinetic energy
2. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`
3. Fused loop: rescale velocities `v *= exp(-dt/2 * p_xi/Q)`, then half-kick `v += 0.5*dt*F/m`, then drift `x += v*dt`

**Post-final integration** (`PostFinalIntegration`):
1. Fused loop: half-kick `v += 0.5*dt*F/m`, then rescale velocities `v *= exp(-dt/2 * p_xi/Q)`
2. Recompute kinetic energy
3. Update thermostat momentum: `p_xi += (dt/2) * (2*KE - ndof*T_target)`

The thermal mass `Q = ndof * T_target * tau^2` where `ndof = 3N - 3` (subtracting center-of-mass degrees of freedom).

Because `NoseHooverPlugin` includes Velocity Verlet integration, `VelocityVerletPlugin` should **not** be added separately.

### Config

```toml
[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time tau_T
# group = "mobile"   # optional: thermostat only this group
```

## Langevin Thermostat (`LangevinPlugin`)

Stochastic thermostat with friction drag and random forces. Unlike Nose-Hoover, Langevin does **not** include Velocity Verlet — pair it with `VelocityVerletPlugin`.

**Physics** (applied at `PostForce`):
- Drag force: `F_drag = -gamma * m * v`
- Random force: `F_rand = sqrt(2 * gamma * m * kT / dt) * N(0,1)`
- Satisfies fluctuation-dissipation theorem

Uses `ChaCha8Rng` seeded per MPI rank for reproducible parallel results.

### Config

```toml
[langevin]
temperature = 0.85   # target temperature
damping = 1.0        # friction coefficient gamma
seed = 12345         # RNG seed
# group = "mobile"   # optional: thermostat only this group
```

## Group Support

Both thermostats support selective thermostatting via the `group` config field. When set, only atoms in the named group are rescaled/forced. The group must be defined in a `[[group]]` block. KE and ndof are computed from the group subset.

## Per-Stage Temperature Changes

Both thermostats re-read config at each `[[run]]` stage boundary using `Config::load_stage_aware`. Place thermostat parameters in a stage's override block to change temperature between stages:

```toml
[[run]]
steps = 50000
overrides.thermostat = { temperature = 1.0, coupling = 1.0 }

[[run]]
steps = 50000
overrides.thermostat = { temperature = 0.5, coupling = 1.0 }
```

## Resources

- `ThermostatConfig` — deserialized Nose-Hoover config
- `NoseHooverState` — thermostat state: `p_xi`, `q_mass`, `target_temp`, `ndof`, `group_name`
- `LangevinConfig` — deserialized Langevin config
- `LangevinState` — RNG, temperature, damping, group

## Usage

```rust
use mddem::prelude::*;

// With Nose-Hoover (includes Velocity Verlet)
let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();

// With Langevin (needs explicit Velocity Verlet)
let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(LatticePlugin)
    .add_plugins(LJForcePlugin)
    .add_plugins(VelocityVerletPlugin::new())
    .add_plugins(LangevinPlugin)
    .add_plugins(MeasurePlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
