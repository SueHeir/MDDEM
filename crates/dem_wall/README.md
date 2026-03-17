# dem_wall

Wall contact forces for DEM simulations using Hertz normal contact mechanics with viscous damping and optional adhesion (JKR, DMT, SJKR cohesion).

## Wall Types & Features

| Type | Description |
|------|-------------|
| **Plane** | Arbitrary-orientation infinite plane with optional bounding box |
| **Cylinder** | Axis-aligned cylindrical container with inside/outside modes |
| **Sphere** | Spherical container with inside/outside modes |
| **Region** | Any `Region` shape as a wall surface |

Wall motion modes (plane walls only): static, constant velocity, sinusoidal oscillation, servo-controlled.

## Key Types

- `WallDef`: TOML configuration
- `WallPlane`, `WallCylinder`, `WallSphere`, `WallRegion`: Runtime wall representations
- `WallMotion`: Static, ConstantVelocity, Oscillate, Servo
- `Walls`: Global wall manager with runtime enable/disable by name
- `WallPlugin`: Plugin registration

## TOML Configuration

```toml
# Plane floor
[[wall]]
type = "plane"
point_z = 0.0
normal_z = 1.0
material = "glass"
name = "floor"

# Cylindrical boundary
[[wall]]
type = "cylinder"
axis = "z"
center = [0.005, 0.005]
radius = 0.004
lo = 0.0
hi = 0.01
inside = true
material = "glass"

# Moving wall (constant velocity)
[[wall]]
type = "plane"
point_z = 0.1
normal_z = 1.0
material = "glass"
velocity = [0.0, 0.0, -0.01]

# Servo-controlled wall (proportional force feedback)
[[wall]]
type = "plane"
point_z = 0.1
normal_z = -1.0
material = "glass"
servo = { target_force = 100.0, max_velocity = 0.1, gain = 0.001 }
```

## Usage

```rust
app.add_plugins(WallPlugin);
```

Define walls as `[[wall]]` entries in TOML, each referencing a material from `[[dem.materials]]`. Systems compute Hertz contact forces, apply motion, and handle adhesion automatically.
