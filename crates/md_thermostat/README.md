# md_thermostat

Temperature control for MD simulations in MDDEM: **Nosé-Hoover NVT** (deterministic) and **Langevin** (stochastic) thermostats.

## Quick Start

### Nosé-Hoover (includes Velocity Verlet)

```toml
[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time τ_T
# group = "mobile"   # optional: thermostat only this group
```

**Key struct:** `NoseHooverPlugin` + `NoseHooverState`

Time-reversible, preserves extended Hamiltonian. Use for equilibrium MD and correlation functions.

### Langevin (pair with VelocityVerletPlugin)

```toml
[langevin]
temperature = 0.85   # target T* (reduced units)
damping = 1.0        # friction coefficient γ
seed = 12345         # RNG seed
# group = "mobile"   # optional: thermostat only this group
```

**Key struct:** `LangevinPlugin` + `LangevinState`

Stochastic, applies drag `F_drag = -γmv` and random forces satisfying fluctuation-dissipation. Fast thermalization.

## Physics

**Nosé-Hoover:** Extended system with auxiliary heat bath variable ξ. Symmetric Liouville splitting ensures time-reversibility and second-order accuracy. Thermal mass: `Q = N_dof × T_target × τ²`.

**Langevin:** `F_drag + F_rand = -γmv + √(2γmkT/dt)·N(0,1)`. ChaCha8 RNG seeded per MPI rank for reproducibility.

## Per-Stage Temperature Ramps

Both thermostats support `StageOverrides`:

```toml
[[run]]
steps = 50000
overrides.thermostat = { temperature = 1.0, coupling = 1.0 }
```

## Group Support

Selective thermostatting: set `group = "groupname"` to thermostat only atoms in that group. KE and ndof computed from subset.

Part of [MDDEM](https://github.com/SueHeir/MDDEM).
