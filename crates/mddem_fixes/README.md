# mddem_fixes

Group-based atom manipulation fixes and gravity for [MDDEM](https://github.com/SueHeir/MDDEM).

Provides force and velocity controls during simulation: add/set forces, freeze atoms, prescribe motion, apply damping, and gravity.

## Key Types

- **`FixesPlugin`** — Registers group-based fixes (addforce, setforce, move_linear, freeze, viscous, nve_limit)
- **`GravityPlugin`** — Registers global gravity force
- **`FixesRegistry`** — Stores all fix definitions loaded from TOML config
- **`AddForceDef`**, **`SetForceDef`**, **`MoveLinearDef`**, **`FreezeDef`**, **`ViscousDef`**, **`NveLimitDef`** — Config structs
- **`GravityConfig`** — Gravity configuration

## TOML Configuration

### addforce — Add constant force to group atoms
```toml
[[addforce]]
group = "particles"
fx = 0.1
fy = 0.0
fz = 0.0
```

### setforce — Set force to constant value
```toml
[[setforce]]
group = "wall"
fx = 0.0
fy = 0.0
fz = 0.0
```

### move_linear — Prescribe constant velocity
```toml
[[move_linear]]
group = "piston"
vx = 0.0
vy = 0.0
vz = -0.001
```

### freeze — Immobilize atoms
```toml
[[freeze]]
group = "frozen"
```

### viscous — Velocity-proportional damping (F = −γv)
```toml
[[viscous]]
group = "all"
gamma = 0.1
```

### nve_limit — Cap max displacement per timestep
```toml
[[nve_limit]]
group = "all"
max_displacement = 0.0001
```

### gravity — Body force (F = m**g**)
```toml
[gravity]
gx = 0.0
gy = 0.0
gz = -9.81
```

## Schedule Phases

- **move_linear**: PreInitialIntegration (set velocity), PostForce (zero force)
- **addforce, setforce, freeze, viscous**: PostForce
- **nve_limit**: PostFinalIntegration
- **gravity**: Force

All fixes validate group names at setup. Ghost atoms (index ≥ nlocal) are skipped.
