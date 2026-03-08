# Hopper

2D slot hopper with angled funnel walls, gravity, and simulation states.

```
Cross-section (x-z plane, periodic in y):

  |                     |    side walls (x=0, x=0.04)
  |   particles here    |
  |                     |
   \                   /     angled funnel walls
    \                 /
     \               /
      \             /
       \___     ___/         funnel exit (1 cm opening)
       |  blocker  |         removable blocker wall (z=0.015)
       |___________|         floor (z=0)
```

**Filling phase:** 200 particles are inserted in the upper region and settle under gravity onto the angled funnel walls and blocker.

**Flowing phase:** Once the total kinetic energy drops below 1e-8 J (particles nearly stationary), the blocker wall is automatically removed and particles flow through the funnel exit to the floor.

This example uses the Rust API (Tier 2) to add a custom update system that monitors KE every 100 steps and triggers the phase transition via `StatesPlugin` and `run_if(in_state(...))`.

## Run

```bash
# Single-process
cargo run --example hopper -- examples/hopper/config.toml

# With MPI
cargo build-examples
mpiexec -n 4 ./target/release/examples/hopper examples/hopper/config.toml
```

## Parameters

| Parameter | Value |
|-----------|-------|
| Particles | 200 |
| Radius | 0.001 m |
| Density | 2500 kg/m^3 |
| Young's modulus | 8.7 GPa |
| Poisson ratio | 0.3 |
| Restitution | 0.8 |
| Friction | 0.4 |
| Gravity | -9.81 m/s^2 (z) |
| Domain | 0.04 x 0.02 x 0.08 m |
| Boundaries | Non-periodic x/z, periodic y |
| Funnel angle | ~67 deg from horizontal |
| Funnel exit | 1 cm opening at z = 0.015 m |
| Blocker wall | z = 0.015 m (removed when KE < 1e-8 J) |
| Total steps | 150,000, thermo 500 |
| KE check | Every 100 steps after step 1000 |

## Validation

`validate.py` checks physics sanity: no NaN/Inf, non-negative temperature, bounded energy (no explosion). Run via `./validate.sh` or directly:

```bash
python3 examples/hopper/validate.py
```
