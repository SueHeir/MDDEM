# md_measure

Measurement tools for [MDDEM](https://github.com/SueHeir/MDDEM): radial distribution function, mean square displacement, and virial pressure.

## Measurements

### Radial Distribution Function (`accumulate_rdf`)
- Histograms pair distances from the neighbor list
- Normalizes by ideal gas shell volume: `g(r) = hist * V / (N_pairs * 4*pi*r^2*dr)`
- Accumulated over multiple samples and averaged at output time
- MPI-reduced across ranks

### Mean Square Displacement (`track_msd`)
- Tracks unwrapped coordinates by detecting periodic boundary crossings
- Computes `MSD = <|r_unwrap(t) - r_ref|^2>` averaged over all atoms
- Diffusion coefficient from Einstein relation: `D = MSD / (6*t)` in the linear regime

### Virial Pressure (`compute_pressure`)
- `P = rho*T - trace(virial)/(3*V) + P_tail`
- Uses `VirialStress` from `mddem_core` (shared across all force types) and `LJTailCorrections` for long-range correction
- Both resources are optional — pressure is computed when available

### File Output (`write_measurements`)
Writes to `{output_dir}/data/` at configurable intervals:
- `rdf.txt` — radial distribution function (r, g(r))
- `msd.txt` — mean square displacement (dt, MSD)
- `pressure.txt` — instantaneous pressure (step, P*)

## Config

```toml
[measure]
rdf_bins = 200
rdf_cutoff = 3.0
rdf_interval = 100
msd_interval = 10
output_interval = 1000
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
