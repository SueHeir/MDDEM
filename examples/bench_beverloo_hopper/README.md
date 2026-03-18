# Beverloo Hopper Discharge Benchmark

Validates MDDEM hopper discharge mass flow rate against the **Beverloo correlation** for quasi-2D granular flow.

## Physics

The Beverloo equation predicts the steady-state mass flow rate through a hopper orifice:

**3D (circular):** W = C · rho_bulk · sqrt(g) · (D - k·d)^(5/2)

**2D (slot):** W = C · rho_bulk · sqrt(g) · (D - k·d)^(3/2) · depth

where:
- `D` = orifice width [m]
- `d` = particle diameter [m]
- `g` = gravitational acceleration [m/s^2]
- `rho_bulk` = bulk density = rho_particle x phi (packing fraction ~0.58)
- `C ~ 0.58` = empirical discharge coefficient
- `k ~ 1.4` = empty annulus correction (particles can't flow within ~k·d/2 of the edge)
- `depth` = slot length (periodic y extent for quasi-2D)

## Setup

- **Geometry**: Flat-bottom rectangular hopper with a central orifice, periodic in y (quasi-2D slab, 2d thick)
- **Particles**: 800 monodisperse glass spheres, d = 2 mm, rho = 2500 kg/m^3
- **Container**: 80 mm wide x 80 mm tall
- **Orifice widths**: 6d, 10d, 15d (12 mm, 20 mm, 30 mm)
- **Stages**: (1) Fill and settle under gravity, (2) Remove blocker wall and discharge
- **Settling criterion**: Per-particle KE < 1e-7 J (avoids scaling issues with total KE)

## Running

### Single orifice width (default D = 10d):
```bash
cargo run --release --example bench_beverloo_hopper --no-default-features \
    -- examples/bench_beverloo_hopper/config.toml
```

### Full sweep (3 orifice widths):
```bash
bash examples/bench_beverloo_hopper/run_sweep.sh
```

### Analysis:
```bash
python3 examples/bench_beverloo_hopper/validate.py   # PASS/FAIL checks
python3 examples/bench_beverloo_hopper/plot.py        # Generate plots
```

## Expected Results

The measured mass flow rate should agree with the Beverloo prediction within ~50% for each orifice width (small DEM systems with ~800 particles have significant statistical noise). The key physics test is the correct **scaling exponent of 3/2**: on a log-log plot of W vs (D - k·d), the data points should follow the theory line.

![Beverloo comparison](beverloo_comparison.png)

![Mass vs time](mass_vs_time.png)

## Assumptions and Simplifications

- **Quasi-2D**: Periodic boundary in y with 2d slab thickness. The 2D Beverloo formula is used with slab depth as the slot length.
- **Monodisperse**: All particles have the same diameter.
- **Flat bottom**: No converging hopper walls (funnel angle = 90 deg). Beverloo is valid for flat-bottom or steep hoppers.
- **Hertz-Mindlin contacts**: Standard DEM contact model with friction = 0.3, restitution = 0.3.
- **Reduced Young's modulus**: E = 5 MPa (vs. 70 GPa for real glass) for computational efficiency. This does not significantly affect steady-state flow rates.
- **Small system**: 800 particles provides adequate statistics for Beverloo validation but with ~50% noise.

## Files

| File | Description |
|------|-------------|
| `main.rs` | Simulation: fill, settle (per-particle KE), discharge with tracking |
| `config.toml` | Default config (D = 10d), fully documented |
| `run_sweep.sh` | Shell script to run 3 orifice widths |
| `validate.py` | Quantitative validation: PASS/FAIL per orifice |
| `plot.py` | Generates beverloo_comparison.png and mass_vs_time.png |
