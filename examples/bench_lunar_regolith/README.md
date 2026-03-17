# Lunar Regolith Cohesive Angle of Repose Benchmark

Validates JKR adhesion in reduced gravity by simulating a funnel-pour of cohesive particles and measuring the resulting angle of repose.

## Physics

Cohesive particles form steeper piles than non-cohesive ones. The key dimensionless parameter is the **granular Bond number**:

$$\text{Bo} = \frac{F_{\text{adhesion}}}{mg} = \frac{\frac{3}{2}\pi\gamma R^*}{mg}$$

where $\gamma$ is JKR surface energy, $R^* = R/2$ for equal spheres, and $g$ is gravitational acceleration.

| Regime | Bond Number | Behavior |
|--------|-------------|----------|
| Gravity-dominated | Bo << 1 | Angle ≈ internal friction angle (~25-30°) |
| Transitional | Bo ~ 1 | Angle increases significantly |
| Cohesion-dominated | Bo >> 1 | Very steep piles, near-vertical cliffs |

Lunar gravity (1.62 m/s²) gives ~6× higher Bo than Earth gravity (9.81 m/s²) for the same particles, explaining why lunar regolith can maintain steeper slopes.

## Setup

- **Geometry**: Quasi-2D (periodic in y, 3 particle diameters thick)
- **Particles**: 350 spheres, R = 1 mm, ρ = 1500 kg/m³ (scaled up from real regolith for tractability)
- **Contact model**: Hertz-Mindlin with JKR adhesion
- **Material**: E = 5 MPa (soft for fast simulation), μ = 0.5, μ_r = 0.1
- **Insertion**: Rate-based pouring from narrow slot above center (10 particles every 500 steps)
- **Floor**: Flat wall at z = 0
- **Boundaries**: shrink-wrap in z, periodic in y, fixed in x (wide enough pile doesn't reach walls)

## Angle Measurement

The pile angle is measured by:
1. Binning particles by |x| (exploiting symmetry about x=0)
2. Finding the surface height in each bin (max z + radius)
3. Fitting a line to the surface profile
4. Angle = atan(|slope|)

## Running

### Single case (default: lunar gravity, medium adhesion)

```bash
cargo run --release --no-default-features --example bench_lunar_regolith -- examples/bench_lunar_regolith/config.toml
```

### Full parametric study (8 cases: {Earth, Moon} × {0, 5, 20, 50 mJ/m²})

```bash
python examples/bench_lunar_regolith/run_benchmark.py
```

### Validation

```bash
python examples/bench_lunar_regolith/validate.py
```

### Plots

```bash
python examples/bench_lunar_regolith/plot.py
```

## Validation Checks

1. **Non-cohesive angle**: 15-45° (consistent with μ = 0.5)
2. **Angle vs adhesion**: Monotonically increases with surface energy
3. **Lunar vs Earth**: Steeper piles on Moon for same adhesion (higher Bo)
4. **Significant cohesion effect**: High-adhesion angles > no-adhesion + 3°

## Expected Results

| Gravity | γ [mJ/m²] | Bo | Expected Angle |
|---------|-----------|-----|---------------|
| Earth | 0 | 0 | ~25-30° |
| Earth | 5 | 0.19 | ~27-32° |
| Earth | 20 | 0.76 | ~32-40° |
| Earth | 50 | 1.91 | ~40-55° |
| Moon | 0 | 0 | ~25-30° |
| Moon | 5 | 1.16 | ~35-45° |
| Moon | 20 | 4.63 | ~50-65° |
| Moon | 50 | 11.6 | ~65-80° |

## Output

- `results.csv` — Tabulated results from parametric study
- `angle_vs_surface_energy.png` — Angle vs γ for Earth and Moon
- `angle_vs_bond_number.png` — Universal scaling with Bond number

![Angle vs Surface Energy](angle_vs_surface_energy.png)
![Angle vs Bond Number](angle_vs_bond_number.png)

## References

- Castellanos, A. (2005). "The relationship between attractive interparticle forces and bulk behaviour in dry and uncharged fine powders." *Advances in Physics*, 54(4), 263-376.
- Johnson, K.L., Kendall, K., Roberts, A.D. (1971). "Surface energy and the contact of elastic solids." *Proc. R. Soc. Lond. A*, 324, 301-313.
- Carrier, W.D. (2003). "Particle size distribution of lunar soil." *Journal of Geotechnical and Geoenvironmental Engineering*, 129(10), 956-959.
