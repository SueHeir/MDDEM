# mddem_velocity_distribution

Velocity distribution analysis plugin for particle simulations (DEM, MD, granular gases).

## What It Does

Measures particle speed distributions and compares against the **Maxwell-Boltzmann distribution** using the measured granular temperature `T_g = <m v'²> / (3M)`. Outputs:

- Binned speed histogram with Maxwell-Boltzmann reference curve
- Per-component (vx, vy, vz) velocity distributions with Gaussian reference
- Quantitative deviation metrics: L2 norm and kurtosis excess
- Per-species granular temperature (polydisperse systems) and equipartition breakdown
- **Inelastic collapse detection**: warns when `T_g` falls below threshold or cooling rate diverges

## Key Types

- **`VelocityDistributionConfig`**: TOML configuration with `interval`, `num_bins`, `max_speed_factor`, `per_species`, `collapse_threshold`, `collapse_rate_window`
- **`VelocityDistributionResult`**: Speed and component histograms, PDFs, deviation metrics, granular temperature
- **`CollapseDetector`**: Tracks temperature history and detects inelastic collapse via threshold crossing and cooling-rate estimation
- **`analyze_velocity_distribution()`**: Core pure function (unit-testable) computing all metrics from velocity/mass arrays
- **`analyze_per_species()`**: Computes per-atom-type granular temperature for polydisperse systems

## TOML Configuration

```toml
[velocity_distribution]
interval = 1000            # output every N steps
num_bins = 50              # histogram bins
max_speed_factor = 3.0     # max speed = factor × v_rms
per_species = false        # enable per-species T_g
collapse_threshold = 1e-12 # T_g threshold for collapse warning
collapse_rate_window = 5   # samples for cooling-rate estimation
```

## Usage Example

```rust
use mddem_velocity_distribution::VelocityDistributionPlugin;
use sim_app::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugin(VelocityDistributionPlugin);
    app.run();
}
```

The plugin registers the ECS system `compute_velocity_distribution` on `ParticleSimScheduleSet::PostFinalIntegration`. Each output cycle writes CSV files to `data/`:
- `velocity_distribution_{step}.csv`: speed histogram
- `velocity_components_{step}.csv`: per-component distributions
- `velocity_distribution_summary.csv`: time series of T_g, cooling rate, collapse status

**Single-rank only**: Analysis is skipped on multi-rank MPI runs (incompatible with distributed particle arrays).
