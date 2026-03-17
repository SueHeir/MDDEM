# dem_thermal

Contact-based heat conduction for DEM particles in MDDEM.

## Overview

Implements thermal energy transfer between overlapping particles in a Discrete Element Method simulation. Heat flows from hotter to cooler particles through the contact area, with temperatures updated each timestep. Fully conserves thermal energy.

## Physics Model

Heat flux between contacting particles i and j:
```
Q = k · 2a · (T_j − T_i)
```

where:
- `k` = thermal conductivity (W/(m·K))
- `a = √(r_eff · δ)` = Hertzian contact radius
- `r_eff = (r_i · r_j) / (r_i + r_j)` = effective radius
- `δ` = overlap depth
- Temperature integration: `T += dt · Q / (m · c_p)`

## Key Types

- **`ThermalConfig`** — Configuration struct with `conductivity`, `specific_heat`, and `initial_temperature` (defaults: 1.0 W/(m·K), 500.0 J/(kg·K), 300.0 K)
- **`ThermalAtom`** — Per-atom data with `temperature` (K, forward-communicated) and `heat_flux` (W, reverse-communicated and zeroed each step)
- **`ThermalPlugin`** — Plugin that registers thermal data and three systems: `initialize_temperatures`, `compute_heat_conduction`, `integrate_temperature`

Also handles wall heat conduction (plane, cylinder, sphere, region walls with optional temperatures).

## TOML Configuration

```toml
[thermal]
conductivity = 1.0          # W/(m·K)
specific_heat = 500.0       # J/(kg·K)
initial_temperature = 300.0 # K (optional, default: 300.0)
```

Omit the `[thermal]` section to register the plugin without running heat transfer.

## Usage Example

```rust
use mddem_app::prelude::*;
use dem_thermal::ThermalPlugin;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(ThermalPlugin);
app.start();
```

Access temperatures via the `AtomDataRegistry`:
```rust
let thermal = registry.expect::<ThermalAtom>("my_system");
let temp_i = thermal.temperature[i];
```

## Dependencies

Requires `DemAtomPlugin` (for particle radii) and `NeighborPlugin` (for contact pair iteration).
