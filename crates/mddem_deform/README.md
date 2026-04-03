# mddem_deform

Box deformation fix for MDDEM — enables continuous modification of simulation domain boundaries during a run. Analogous to LAMMPS `fix deform`, ideal for triaxial compression, oedometer tests, and other geomechanics simulations.

## Features

- **Three deformation styles per axis (x, y, z)**:
  - `erate`: Engineering strain rate (e.g., `L(t) = L0 * (1 + rate * dt * step)`)
  - `vel`: Constant velocity on box faces (e.g., `L(t) = L0 + velocity * dt * step`)
  - `final`: Linear ramp to target bounds over the run duration
- **Independent per-axis control**: Leave axes unchanged or apply different styles simultaneously
- **Atom remapping**: Affine transformation maintains relative positions as domain deforms
- **Multi-stage support**: Per-stage `[deform]` config overrides via `[[run]]` blocks

## Configuration

Add `[deform]` section to your TOML config:

```toml
[deform]
# Uniaxial compression on z-axis (engineering strain rate)
z = { style = "erate", rate = -0.001 }

# Constant velocity expansion on x-axis (optional)
# x = { style = "vel", velocity = 0.01 }

# Linear ramp on y-axis (optional)
# y = { style = "final", lo = 0.0, hi = 0.02 }

# Remap atom positions affinely (default: true)
remap = true
```

## Key Types

- **`DeformConfig`**: Top-level TOML config (x, y, z axes + remap flag)
- **`AxisDeformDef`**: Per-axis definition with style and parameters
- **`DeformStyle`** enum: `Erate`, `Vel`, or `Final` variants
- **`DeformState`**: Runtime resource tracking deformation progress
- **`DeformPlugin`**: Registers systems at `ScheduleSetupSet::PostSetup` (setup) and `ParticleSimScheduleSet::PreInitialIntegration` (apply)

## Usage

Add the plugin to your simulation:

```rust
use mddem_deform::DeformPlugin;

app.add_plugin(DeformPlugin);
```

The plugin automatically reads `[deform]` config (including per-stage overrides) and applies deformation each timestep before the Verlet position update. Domain bounds, atom positions, and neighbor lists update automatically.
