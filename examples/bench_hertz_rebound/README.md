# Hertz Contact Rebound Benchmark

Validates Hertzian contact mechanics by dropping a single sphere onto a rigid flat wall and measuring the coefficient of restitution (COR), contact duration, and peak overlap.

## Physics

A sphere of radius R, mass m, impacts a rigid flat wall at velocity v₀. The Hertz contact model predicts:

- **Contact duration** (elastic):
  ```
  t_c = 2.87 × (m²/(R·E*²·v₀))^(1/5)
  ```
- **Peak overlap** (elastic):
  ```
  δ_max = (15·m·v₀² / (16·√R·E*))^(2/5)
  ```
- **COR**: The viscoelastic damping model should reproduce the input COR parameter.

where E* = E/(2(1−ν²)) is the reduced modulus for sphere-on-flat contact.

## Material Properties

| Property | Value | Unit |
|----------|-------|------|
| Young's modulus E | 70 GPa | Pa |
| Poisson's ratio ν | 0.22 | — |
| Density ρ | 2500 | kg/m³ |
| Radius R | 5 | mm |

## Parameter Sweep

- **Impact velocities**: 0.1, 0.5, 1.0, 2.0 m/s
- **COR values**: 0.5, 0.7, 0.9, 0.95

## Validation Criteria

| Check | Tolerance | Notes |
|-------|-----------|-------|
| COR matches input (COR ≥ 0.7) | ≤ 3% relative error | |
| COR matches input (COR < 0.7) | ≤ 12% relative error | Known Hertz nonlinearity effect* |
| Contact duration vs Hertz | ≤ 10% relative error | |
| Peak overlap vs Hertz | ≤ 10% relative error | |
| All 16 cases complete | 16/16 | |

\* The β damping coefficient is derived from linear (Hooke) contact theory. When applied with nonlinear Hertz stiffness, the achieved COR deviates from the input value, especially at low COR. This is a well-known limitation shared by LAMMPS and other DEM codes using the same model.

## How to Run

### Single case (default config)

```bash
cargo run --release --example bench_hertz_rebound --no-default-features -- examples/bench_hertz_rebound/config.toml
```

### Full parameter sweep

```bash
python3 examples/bench_hertz_rebound/run_sweep.py
```

### Validate results

```bash
python3 examples/bench_hertz_rebound/validate.py
```

### Generate plots

```bash
python3 examples/bench_hertz_rebound/plot.py
```

## Expected Plots

### COR Validation
![COR validation](plots/cor_validation.png)

### Contact Duration
![Contact duration](plots/contact_duration.png)

### Peak Overlap
![Peak overlap](plots/peak_overlap.png)

## Assumptions

- **3D simulation** with a single spherical particle
- **No friction** (friction = 0) for clean normal-only rebound
- **No gravity** effect on contact (gravity is off; particle given direct velocity)
- **Monodisperse** — single particle size
- **Hertz–Mindlin** contact model with viscoelastic damping (MDDEM default)
- Wall is treated as **infinitely massive and rigid** (standard DEM wall)

## References

1. K.L. Johnson, *Contact Mechanics*, Cambridge University Press, 1985.
2. L. Vu-Quoc and X. Zhang, "An accurate and efficient tangential force-displacement model for elastic frictional contact in particle-flow simulations", *Mechanics of Materials*, 31(4):235–269, 1999.
