# dem_measure_plane

General-purpose measurement plane plugin for counting particle crossings and mass flow rates in MDDEM simulations.

## Overview

A **measurement plane** is an infinite plane defined by a point and a normal vector. Each timestep, the plugin detects when particles cross the plane in the positive-normal direction and tracks:
- **Cumulative crossing count** — total particles crossing since simulation start
- **Mass flow rate** — mass per unit time (averaged over `report_interval`)
- **Crossing rate** — particles per unit time (averaged over `report_interval`)

This is useful for measuring throughput in hoppers, chutes, conveyors, and other granular flows.

## Crossing Detection Algorithm

For each particle (tracked by tag):
1. Compute signed distance: `d = (pos - point) · normal`
2. If previous distance `d_prev ≤ 0` and current `d > 0`, count a positive-direction crossing
3. Reverse crossings (positive → negative) are ignored

## Key Types

- **`MeasurePlaneDef`** — TOML configuration for a single plane
- **`MeasurePlaneState`** — Runtime state tracking signed distances and crossings per plane
- **`MeasurePlanes`** — ECS resource holding all plane states
- **`MeasurePlanePlugin`** — Plugin that registers crossing detection and reporting systems

## TOML Configuration

```toml
[[measure_plane]]
name = "outlet"           # Unique name; used in thermo output keys
point = [0.1, 0.0, 0.0]  # Any point on the plane [length units]
normal = [1.0, 0.0, 0.0] # Outward normal (auto-normalized)
report_interval = 1000   # Averaging window in timesteps (default: 1000)
```

Multiple `[[measure_plane]]` blocks can be defined for different cross-sections.

## Thermo Output Keys

For each plane named `<name>`:
- `crossings_<name>` — Total cumulative crossing count
- `flow_rate_<name>` — Mass flow rate (mass/time), averaged over `report_interval`
- `cross_rate_<name>` — Particle crossing rate (1/time), averaged over `report_interval`

## Usage Example

```rust
use dem_measure_plane::MeasurePlanePlugin;
use grass_app::prelude::*;

let mut app = App::new();
app.add_plugin(MeasurePlanePlugin);
// ... add other plugins and run simulation
```

Configure planes in your TOML file and results will be automatically reported to thermo output at each plane's `report_interval`.
