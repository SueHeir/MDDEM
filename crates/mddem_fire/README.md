# mddem_fire

FIRE (Fast Inertial Relaxation Engine) energy minimization for MDDEM simulations.

## Overview

This crate implements the FIRE algorithm ([Bitzek et al., 2006](https://doi.org/10.1103/PhysRevLett.97.170201)), a damped-dynamics minimizer that rapidly converges systems to local energy minima. FIRE steers velocities toward force directions and adaptively adjusts the timestep based on power feedback (`P = F · V`), achieving faster convergence than naive steepest descent.

## Key Features

- **Adaptive timestep**: Grows after sustained downhill motion, shrinks on uphill steps
- **Velocity mixing**: Steers particle velocities toward force directions via parameter `α`
- **Convergence criterion**: Stops when maximum per-atom force magnitude drops below `ftol`
- **Stage-aware**: Can run in a single stage or coexist with Velocity Verlet for multi-stage workflows

## Key Types

- **`FireConfig`** — TOML configuration struct with tunable parameters (force tolerance, timestep growth factors, mixing strength, etc.)
- **`FireState`** — Runtime state tracking current timestep `dt_fire`, mixing parameter `alpha`, positive-power step counter, and convergence status
- **`FireMinPlugin`** — Plugin that registers FIRE systems into the scheduler

## Configuration Example

```toml
[fire]
ftol = 1e-6           # force convergence tolerance
dt_max_factor = 10.0  # max timestep = base dt × this factor
f_inc = 1.1           # timestep growth multiplier
f_dec = 0.5           # timestep shrink multiplier
alpha_start = 0.1     # initial velocity-mixing strength
f_alpha = 0.99        # alpha decay factor
n_delay = 5           # steps required before timestep grows
stop_on_converge = true  # auto-advance to next stage at convergence
```

## Usage Example

**Single-stage minimization** (replaces Velocity Verlet entirely):
```rust
app.add_plugins(FireMinPlugin::new());
```

**Multi-stage workflow** (minimize → dynamics):
```rust
app.add_plugins(GranularDefaultPlugins)              // includes Verlet
    .add_plugins(FireMinPlugin::for_stage("minimize")); // FIRE only in "minimize" stage
```

After minimization, inspect `Res<FireState>` to check `converged` status or iteration count.

## References

Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Derlet, P. (2006).
*Structural Relaxation Made Simple.* Physical Review Letters, **97**(17), 170201.
https://doi.org/10.1103/PhysRevLett.97.170201
