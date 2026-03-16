# mddem_fixes

Group-based atom manipulation fixes and gravity for [MDDEM](https://github.com/SueHeir/MDDEM).

## Fixes (`FixesPlugin`)

All fixes operate on named atom groups defined in `[[group]]` blocks. Groups are validated at setup.

### AddForce
Adds a constant force to all atoms in a group at `PostForce`:
```toml
[[addforce]]
group = "mobile"
fx = 0.0
fy = 0.0
fz = -0.01
```

### SetForce
Overrides force on group atoms to a constant value at `PostForce`:
```toml
[[setforce]]
group = "wall_atoms"
fx = 0.0
fy = 0.0
fz = 0.0
```

### Freeze
Zeros velocity and force on group atoms at `PostForce`, effectively fixing them in place:
```toml
[[freeze]]
group = "frozen"
```

### MoveLinear
Prescribes constant velocity on group atoms. Sets velocity before integration (`PreInitialIntegration`) and zeros force after (`PostForce`) so Verlet doesn't change the prescribed velocity:
```toml
[[move_linear]]
group = "piston"
vx = 0.0
vy = 0.0
vz = -0.001
```

## Gravity (`GravityPlugin`)

Applies gravitational body force `F = m * g` to all local atoms at `Force`:
```toml
[gravity]
gx = 0.0
gy = 0.0
gz = -9.81     # default
```

Ghost atoms are skipped (only local atoms receive gravity).

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(FixesPlugin)
    .add_plugins(GravityPlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
