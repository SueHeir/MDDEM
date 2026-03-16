# dem_wall

General plane wall contact forces for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Walls are defined by a point and normal vector, supporting both axis-aligned and angled walls.

## Features

- Hertz elastic contact with viscoelastic damping (same model as particle-particle contacts)
- JKR adhesion and SJKR cohesion support (reads `surface_energy` / `cohesion_energy` from `MaterialTable`)
- General plane geometry — any orientation, not just axis-aligned
- Optional bounding box to clip walls to finite regions
- Named walls with runtime toggling via `walls.deactivate_by_name("blocker")`
- Multiple walls per simulation
- Wall motion: static, constant velocity, oscillating (sinusoidal), and servo-controlled

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
```

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
