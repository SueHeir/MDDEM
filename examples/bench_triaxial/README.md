# Triaxial Compression Benchmark

Validates DEM triaxial compression against **Mohr-Coulomb failure theory**.

## Physics

A sample of 100 randomly packed spheres is confined laterally by servo-controlled
walls at a specified confining pressure σ₃, then compressed axially at constant
velocity. The Mohr-Coulomb criterion predicts a linear failure envelope:

    σ₁ = σ₃ × (1 + sin φ) / (1 - sin φ) + 2c cos φ / (1 - sin φ)

For cohesionless particles (c = 0):

    sin φ = (σ₁ − σ₃) / (σ₁ + σ₃)

where σ₁ is peak axial stress, σ₃ is confining pressure, and φ is the internal
friction angle. For μ = 0.5 inter-particle friction, theory predicts φ ≈ 20–35°.

## Setup

| Parameter            | Value           | Units  |
|---------------------|-----------------|--------|
| Box dimensions      | 10 × 10 × 40   | mm     |
| Particle radius     | 1               | mm     |
| Particle count      | 100             | –      |
| Particle density    | 2500            | kg/m³  |
| Young's modulus     | 10 MPa          | Pa     |
| Poisson's ratio     | 0.3             | –      |
| Restitution         | 0.3             | –      |
| Friction (μ)        | 0.5             | –      |
| Confining pressures | 10, 50, 200     | kPa    |

### Simplifications

- **Monodisperse** spheres (no size distribution)
- **Soft particles** (E = 10 MPa) for fast timestep; contact overlaps remain < 2%
- Gravity set to zero during compression for **uniform stress distribution**
- Servo walls target constant **force** (not constant pressure), so σ₃ varies
  slightly with sample height changes during compression
- **100 particles** — small sample for fast runtime (~2 s per confining pressure)

## Running

```bash
# Run all three confining pressures (~4 s total)
bash examples/bench_triaxial/run_benchmark.sh

# Or run a single pressure
cargo run --release --example bench_triaxial --no-default-features \
    -- examples/bench_triaxial/config_10kPa.toml

# Validate results
python3 examples/bench_triaxial/validate.py

# Generate plots
python3 examples/bench_triaxial/plot.py
```

## Validation

`validate.py` checks:

1. **Valid data** — no NaN/Inf in stress measurements
2. **Stress ratio** — peak σ₁/σ₃ is in range [1.5, 8.0]
3. **Friction angle** — φ is in range [15°, 40°]
4. **Consistency** — φ is consistent across confining pressures (σ < 5°)
5. **MC linearity** — q–p failure line has R² > 0.9

## Plots

- **stress_strain.png** — Deviatoric stress q vs axial strain at all pressures
- **mohr_circles.png** — Mohr circles at failure with fitted envelope
- **q_p_plot.png** — q–p diagram with Mohr-Coulomb failure line

## References

- Cundall, P.A. & Strack, O.D.L. (1979). A discrete numerical model for
  granular assemblies. *Géotechnique*, 29(1), 47–65.
- Thornton, C. (2000). Numerical simulations of deviatoric shear deformation
  of granular media. *Géotechnique*, 50(1), 43–53.
