# mddem_fire

FIRE (Fast Inertial Relaxation Engine) energy minimization for [MDDEM](https://github.com/SueHeir/MDDEM).

## Algorithm

Standard FIRE minimizer with adaptive timestep and velocity mixing:
- Velocity Verlet integration (half-kick + position update + half-kick)
- Power `P = F . v` computed each step
- When `P > 0` for `n_delay` steps: increase dt, decay mixing parameter alpha
- When `P < 0`: zero velocities, decrease dt, reset alpha
- Converged when max per-atom force magnitude < `ftol`

## Stage Support

`FireMinPlugin` can run in all stages or be restricted to a named stage:

```rust
// Run FIRE in all stages (replaces Velocity Verlet entirely)
app.add_plugins(FireMinPlugin::new());

// Run FIRE only during the "minimize" stage
// Can coexist with VelocityVerletPlugin for other stages
app.add_plugins(FireMinPlugin::for_stage("minimize"));
```

On convergence (when `stop_on_converge = true`), the plugin requests stage advancement to the next `[[run]]` block.

## Configuration

```toml
[fire]
ftol = 1e-6              # max per-atom force for convergence
dt_max_factor = 10.0     # max dt as multiple of base dt
f_inc = 1.1              # dt increase factor on positive power
f_dec = 0.5              # dt decrease factor on negative power
alpha_start = 0.1        # initial velocity mixing parameter
f_alpha = 0.99           # alpha decay factor
n_delay = 5              # positive-power steps before dt increases
stop_on_converge = true  # advance to next stage on convergence
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(FireMinPlugin::for_stage("minimize"));
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
