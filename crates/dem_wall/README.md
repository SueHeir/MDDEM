# dem_wall

General plane wall contact forces for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Walls are defined by a point and normal vector, supporting both axis-aligned and angled walls.

## Features

- Hertz elastic contact with viscoelastic damping (same model as particle-particle contacts)
- General plane geometry — any orientation, not just axis-aligned
- Optional bounding box to clip walls to finite regions
- Named walls with runtime toggling via `walls.deactivate_by_name("blocker")`
- Multiple walls per simulation

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

# Named wall (can be deactivated at runtime)
[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.015
normal_x = 0.0
normal_y = 0.0
normal_z = 1.0
material = "glass"
name = "blocker"
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
