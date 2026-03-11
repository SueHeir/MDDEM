# LJ Argon Example

Lennard-Jones fluid simulation in reduced units, validated against known liquid Argon properties at the triple point.

## Parameters
- T* = 0.85, rho* = 0.85 (near triple point)
- 864 atoms on FCC lattice (6x6x6 unit cells)
- LJ 12-6 potential with cutoff at 2.5 sigma
- Nose-Hoover NVT thermostat (tau = 1.0)
- dt = 0.005 (LJ reduced units)
- 100,000 steps

## Validated Results
- **g(r) first peak**: r = 1.09 sigma, height = 2.94 (expected ~1.0 sigma, ~2.4-3.5)
- **Diffusion coefficient**: D* = 0.041 (expected 0.035-0.042)
- **Virial pressure**: P* ~ 0.96 (with tail corrections)
- **g(r) -> 1** at large r: confirmed (mean = 1.015)

## Usage

```bash
# Build and run
cargo run --release --no-default-features --example lj_argon -- examples/lj_argon/config.toml

# Validate results and generate plots
python3 examples/lj_argon/validate.py
```

## Output

The simulation writes measurement data to `examples/lj_argon/data/`:
- `rdf.txt` — radial distribution function g(r)
- `msd.txt` — mean square displacement vs time
- `pressure.txt` — instantaneous virial pressure vs step

The validation script (`validate.py`) checks all measurements against expected values and generates a 4-panel plot saved as `validation.png`.

## Plugins Used

- `CorePlugins` — communication, domain, neighbor lists, Velocity Verlet, thermo output
- `LatticePlugin` — FCC lattice initialization with Maxwell-Boltzmann velocities
- `LJForcePlugin` — LJ 12-6 pair force with virial accumulator and tail corrections
- `NoseHooverPlugin` — Nose-Hoover NVT thermostat
- `MeasurePlugin` — RDF, MSD, and virial pressure measurements
