# dem_granular

Granular physics for [MDDEM](https://github.com/SueHeir/MDDEM): Hertz and Hooke contact models with Mindlin friction, rolling/twisting resistance, adhesion, rotational dynamics, and granular temperature output.

## What it does

Core DEM physics for spherical particle simulations:

- **Normal contact**: Hertz (nonlinear `F_n ∝ δ^1.5`) or Hooke (linear `F_n ∝ δ`)
- **Tangential friction**: Mindlin spring-history with Coulomb cap and viscous damping
- **Rolling resistance**: Constant torque (default) or SDS spring-dashpot-slider
- **Twisting friction**: Constant torque (default) or SDS
- **Adhesion**: JKR (extended range) or DMT (contact-only), plus SJKR cohesion
- **Rotational dynamics**: Quaternion velocity Verlet (I = 2/5 m r² for solid spheres)
- **Output**: Granular temperature (velocity fluctuation) time series

## Key types

- `GranularDefaultPlugins` — Complete granular physics plugin group
- `HertzMindlinContactPlugin` — Fused normal + tangential (recommended)
- `HertzNormalForcePlugin` — Normal-only contact
- `MindlinTangentialForcePlugin` — Tangential friction alone
- `RotationalDynamicsPlugin` — Angular integration
- `GranularTempPlugin` — Granular temperature file output

## TOML configuration

```toml
[[materials]]
name = "glass"
youngs_modulus = 8.7e9      # Pa
poisson_ratio = 0.3
restitution = 0.95          # coeff. of restitution (0–1)
friction = 0.4              # sliding friction μ
rolling_friction = 0.1      # rolling friction μ_r
cohesion_energy = 0.0       # J/m² (SJKR, 0 = disabled)
surface_energy = 0.0        # J/m² (JKR/DMT, 0 = disabled)

[materials]
contact_model = "hertz"     # or "hooke"
adhesion_model = "jkr"      # or "dmt"
rolling_model = "constant"  # or "sds"
twisting_model = "constant" # or "sds"
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
app.start();
```

Loads all default physics: Hertz-Mindlin contact, rolling/twisting, rotational dynamics, and granular temperature. Use individual plugins for custom configurations.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
