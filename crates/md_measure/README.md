# md_measure

Measurement tools for molecular dynamics simulations: radial distribution function (RDF), mean square displacement (MSD), and virial pressure.

## Overview

`md_measure` provides three complementary measurement systems that run automatically during the `PostFinalIntegration` schedule:

- **RDF** — Computes g(r), measuring how particle density varies with distance from a reference particle. Normalized by ideal gas shell volume: `g(r) = hist × V / (N_pairs × V_shell)`.
- **MSD** — Tracks mean square displacement using unwrapped coordinates to correctly handle periodic boundary crossings. Enables diffusion coefficient estimation via Einstein relation: `D = MSD / (6t)` in 3D.
- **Virial Pressure** — Computes instantaneous pressure from kinetic and virial contributions: `P = ρT − Tr(W)/(3V) + P_tail`. Virial stress is shared across all force types; LJ tail corrections applied when available.

## Key Types

| Type | Purpose |
|------|---------|
| [`MeasureConfig`](src/lib.rs#L96) | TOML `[measure]` configuration with sensible defaults |
| [`RdfAccumulator`](src/lib.rs#L160) | Accumulates g(r) histogram samples over time |
| [`MsdTracker`](src/lib.rs#L206) | Tracks reference and unwrapped positions for MSD computation |
| [`PressureHistory`](src/lib.rs#L273) | Records timestep–pressure pairs |
| [`MeasurePlugin`](src/lib.rs#L285) | Plugin registering all measurement systems |

## TOML Configuration

All settings live under `[measure]` with sensible defaults:

```toml
[measure]
rdf_bins = 200          # Histogram bins for g(r) (default: 200)
rdf_cutoff = 3.0        # Max pair distance for RDF, in LJ units (default: 3.0)
rdf_interval = 100      # Accumulate RDF sample every N steps (default: 100)
msd_interval = 10       # Record MSD and pressure every N steps (default: 10)
output_interval = 1000  # Write files every N steps (default: 1000)
```

Set any interval to `0` to disable that measurement.

## Output Files

All files are written to `<output_dir>/data/` at `output_interval`:

- **`rdf.txt`** — Two columns: `r  g(r)`, time-averaged over all samples.
- **`msd.txt`** — Two columns: `dt  MSD`, where dt is steps since reference.
- **`pressure.txt`** — Two columns: `step  pressure`, instantaneous virial pressure.

## Usage

Add `MeasurePlugin` to your app to enable all measurements:

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.add_plugin(md_measure::MeasurePlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
