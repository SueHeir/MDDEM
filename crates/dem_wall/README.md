# dem_wall

Wall contact forces for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Supports plane, cylinder, and sphere wall geometries.

## Features

- Hertz elastic contact with viscoelastic damping (same model as particle-particle contacts)
- JKR adhesion and SJKR cohesion support (reads `surface_energy` / `cohesion_energy` from `MaterialTable`)
- Twisting friction support (reads `twisting_friction` from `MaterialTable`)
- **Plane walls** — general plane geometry, any orientation, optional bounding box
- **Cylinder walls** — axis-aligned cylindrical containers/pipes with inside/outside contact
- **Sphere walls** — spherical containers with inside/outside contact
- Named walls with runtime toggling via `walls.deactivate_by_name("blocker")`
- Multiple walls per simulation
- Wall motion: static, constant velocity, oscillating (sinusoidal), and servo-controlled (plane walls)

## Wall Motion

Walls can move during simulation. Motion type is set per-wall in TOML config:

**Static** (default): wall does not move.

**Constant velocity**: wall translates at a fixed velocity each step.
```toml
velocity = [0.0, 0.0, -0.01]
```

**Oscillating**: sinusoidal displacement along the wall normal.
```toml
oscillate = { amplitude = 0.001, frequency = 50.0 }
```
Position: `origin + amplitude * sin(2*pi*f*t)`. Velocity is analytically derived.

**Servo-controlled**: proportional feedback control targeting a net normal force.
```toml
servo = { target_force = 100.0, max_velocity = 0.1, gain = 0.001 }
```
The wall accumulates contact forces each step. The servo adjusts wall velocity: `vel = clamp(gain * (target - accumulated_force), -max_vel, max_vel)`. Useful for constant-stress compression.

Moving walls use corrected relative velocity for damping: `v_rel = v_atom - v_wall`.

## Configuration

```toml
# Axis-aligned floor wall
[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.0
normal_x = 0.0
normal_y = 0.0
normal_z = 1.0
material = "glass"

# Angled funnel wall with bounding box
[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.05
normal_x = 0.919
normal_y = 0.0
normal_z = 0.394
material = "glass"
bound_z_low = 0.015
bound_z_high = 0.05

# Named wall with constant velocity
[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.05
normal_x = 0.0
normal_y = 0.0
normal_z = -1.0
material = "glass"
name = "platen"
velocity = [0.0, 0.0, -0.001]

# Servo-controlled wall
[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.05
normal_x = 0.0
normal_y = 0.0
normal_z = -1.0
material = "glass"
name = "servo_top"
servo = { target_force = 50.0, max_velocity = 0.01, gain = 0.0001 }

# Cylindrical container (particles inside)
[[wall]]
type = "cylinder"
axis = "z"
center = [0.005, 0.005]
radius = 0.004
lo = 0.0
hi = 0.01
material = "glass"
inside = true

# Spherical container (particles inside)
[[wall]]
type = "sphere"
center = [0.005, 0.005, 0.005]
radius = 0.004
material = "glass"
inside = true
```

### Cylinder Walls

Axis-aligned cylindrical walls defined by an axis (`"x"`, `"y"`, or `"z"`), center (2D, perpendicular to axis), radius, and optional axial bounds (`lo`/`hi`). When `inside = true`, particles are contained inside the cylinder and the normal points inward. When `inside = false` (default), particles interact with the outside surface.

### Sphere Walls

Spherical walls defined by a 3D center and radius. When `inside = true`, particles are contained inside the sphere. When `inside = false` (default), particles interact with the outside surface.

## Runtime Wall Control

Walls can be activated/deactivated by name during simulation, useful for staged simulations like hoppers:

```rust
fn check_settled(mut walls: ResMut<Walls>, mut next_state: ResMut<NextState<Phase>>) {
    walls.deactivate_by_name("blocker");
    next_state.set(Phase::Flowing);
}
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(WallPlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
