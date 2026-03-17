# Multisphere Segregation Benchmark

## Physics

This benchmark validates the **clump/multisphere** implementation by simulating
size-driven segregation of single spheres vs dimer clumps under vertical
vibration — the **Brazil nut effect** for particle shape.

A 50/50 mixture of:
- **75 single spheres** (type 0, radius 1.0 mm)
- **75 dimer clumps** (type 1, two overlapping spheres with 1.0 mm offset)

is placed in a box with:
- Periodic lateral boundaries (x, y)
- A vertically oscillating floor (sinusoidal, z-direction)
- A static ceiling

### Vibration Parameters

| Parameter | Value |
|-----------|-------|
| Amplitude A | 2.0 mm |
| Frequency f | 15 Hz |
| Angular frequency ω | 94.25 rad/s |
| Dimensionless acceleration Γ = Aω²/g | **3.6** |

### Expected Behavior

Under vibration at Γ ≈ 3.6, the dimers have a larger effective size than single
spheres. The **Brazil nut effect** (also called granular convection / geometric
segregation) causes larger effective particles to rise to the top of the bed.

The **segregation index** is defined as:

$$S = \frac{z_{\text{dimer}} - z_{\text{sphere}}}{z_{\text{dimer}} + z_{\text{sphere}}}$$

where $z$ denotes the mass-weighted center-of-mass height. **S > 0** indicates
dimers are preferentially at the top.

### Assumptions / Simplifications

- 3D simulation with periodic lateral boundaries (quasi-infinite horizontal extent)
- Monodisperse spheres and identical dimers (no size distribution)
- Softened Young's modulus (5×10⁷ Pa) for computational efficiency
- Hertz–Mindlin contact with rolling friction
- No cohesion or adhesion

## Running

```bash
# Build and run (release mode, ~2-3 min)
cargo run --release --example bench_multisphere_segregation --no-default-features \
    -- examples/bench_multisphere_segregation/config.toml

# Validate results (PASS/FAIL)
python3 examples/bench_multisphere_segregation/validate.py

# Generate plots
python3 examples/bench_multisphere_segregation/plot.py
```

## Output

- `data/segregation.csv` — step, time, z_sphere, z_dimer, segregation_index
- `data/Thermo.txt` — standard thermo output
- `segregation_index.png` — segregation index vs time
- `com_trajectories.png` — COM height trajectories

## Validation Criteria

| Check | Criterion |
|-------|-----------|
| No NaN/Inf | All data must be finite |
| Physical z | Both COM heights must be positive |
| S > 0 at steady state | Mean S in final 20% of run must be positive |
| Significant segregation | Mean final S > 0.005 |
| Positive trend | S increases (or is already high) in second half |
