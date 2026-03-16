# dem_thermal

Contact-based heat conduction for DEM particles in [MDDEM](https://github.com/SueHeir/MDDEM).

## Physics

Per-atom temperature with heat transfer through the contact area between overlapping particles:

- Contact radius: `a = sqrt(R* * delta)` where `R*` is the effective radius and `delta` is the overlap
- Heat transfer: `Q = conductivity * 2*a * (Tj - Ti)`
- Temperature integration: `T += dt * Q / (mass * cp)`

Energy is conserved — heat leaving one particle enters the other.

## Per-Atom Data (`ThermalAtom`)

- `temperature: Vec<f64>` — per-atom temperature in K (`#[forward]` communicated to ghost atoms)
- `heat_flux: Vec<f64>` — per-atom heat flux accumulator in W (`#[reverse]`, `#[zero]`)

## Configuration

```toml
[thermal]
conductivity = 1.0          # W/(m*K)
specific_heat = 500.0       # J/(kg*K)
initial_temperature = 300.0 # K (default)
```

The `[thermal]` section is optional. If omitted, `ThermalAtom` data is registered but no heat conduction systems run.

## Systems

- `initialize_temperatures` (Setup) — sets all atoms to `initial_temperature`
- `compute_heat_conduction` (Force) — loops neighbor pairs, computes contact heat transfer
- `integrate_temperature` (PostFinalIntegration) — updates temperatures from accumulated heat flux

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(ThermalPlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
