# Beverloo Hopper Discharge Benchmark

Validates MDDEM hopper discharge mass flow rate against the **Beverloo correlation** for quasi-2D granular flow.

## Physics

The Beverloo equation predicts the steady-state mass flow rate through a hopper orifice:

**3D:** W = C · ρ_bulk · √g · (D − k·d)^(5/2)

**2D:** W = C · ρ_bulk · √g · (D − k·d)^(3/2) · depth

where:
- `D` = orifice width [m]
- `d` = particle diameter [m]
- `g` = gravitational acceleration [m/s²]
- `ρ_bulk` = bulk density ≈ ρ_particle × φ (packing fraction ~0.6)
- `C ≈ 0.58` = empirical discharge coefficient
- `k ≈ 1.4` = empty annulus correction (particles can't flow within ~k·d/2 of the wall)

## Setup

- **Geometry**: Flat-bottom rectangular hopper with a central orifice, periodic in y (quasi-2D slab, 2d thick)
- **Particles**: 3000 monodisperse glass spheres, d = 2 mm, ρ = 2500 kg/m³
- **Container**: 80 mm wide × 120 mm tall
- **Orifice widths**: 5d, 8d, 12d, 16d (10 mm, 16 mm, 24 mm, 32 mm)
- **Stages**: (1) Fill and settle under gravity, (2) Remove blocker wall and discharge

## Running

### Single orifice width (default D = 8d):
```bash
cargo run --release --example bench_beverloo_hopper --no-default-features \
    -- examples/bench_beverloo_hopper/config.toml
```

### Full sweep (4 orifice widths):
```bash
bash examples/bench_beverloo_hopper/run_sweep.sh
```

### Analysis:
```bash
python3 examples/bench_beverloo_hopper/validate.py   # PASS/FAIL checks
python3 examples/bench_beverloo_hopper/plot.py        # Generate plots
```

## Expected Results

The measured mass flow rate should agree with the Beverloo prediction within ±30% for each orifice width. The log-log plot of W vs (D − k·d) should show a slope of 3/2, consistent with the 2D Beverloo exponent.

![Beverloo comparison](beverloo_comparison.png)

![Mass vs time](mass_vs_time.png)

## Assumptions and Simplifications

- **Quasi-2D**: Periodic boundary in y with 2d slab thickness. True 2D Beverloo uses depth as a scaling factor.
- **Monodisperse**: All particles have the same diameter.
- **Flat bottom**: No converging hopper walls (funnel angle = 90°). Beverloo is strictly for flat-bottom or steep hoppers.
- **Hertz-Mindlin contacts**: Standard DEM contact model with friction = 0.3, restitution = 0.5.
- **Reduced Young's modulus**: E = 5 MPa (vs. 70 GPa for real glass) for computational efficiency. This does not significantly affect steady-state flow rates.

## Files

| File | Description |
|------|-------------|
| `main.rs` | Simulation: fill → settle → discharge with particle tracking |
| `config.toml` | Default config (D = 8d), fully documented |
| `run_sweep.sh` | Shell script to run all 4 orifice widths |
| `validate.py` | Quantitative validation: PASS/FAIL per orifice |
| `plot.py` | Generates beverloo_comparison.png and mass_vs_time.png |
