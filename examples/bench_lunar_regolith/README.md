# Lunar Regolith Cohesive Angle of Repose Benchmark

Validates JKR adhesion in reduced gravity using a **draining-box** method: particles settle in a walled box, the "gate" wall is removed, and the remaining pile's angle of repose is measured.

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

- **Method**: Draining box — 3-stage simulation (settle, drain, measure)
- **Geometry**: Quasi-2D (periodic in y, 10mm thick)
- **Particles**: 2500 spheres, R = 1 mm, ρ = 1500 kg/m³ (count-based instant insertion)
- **Box**: x = [-50mm, -10mm], z = [0, 100mm] — narrow column against the left wall
- **Gate**: Wall at x = -10mm, removed when KE drops below threshold
- **Drain area**: x = [-10mm, +120mm] — ample room for particles to spread
- **Contact model**: Hertz-Mindlin with JKR adhesion
- **Material**: E = 5 MPa, μ = 0.5, μ_r = 1.0, e = 0.1
- **Stages**: settle (100k steps) → drain (150k steps) → measure (50k steps) = 300k total

## Angle Measurement

After the gate is removed, the remaining pile rests against the left wall with a sloped free surface on the right. The angle is measured by:
1. Binning particles by x position (no symmetry — pile is one-sided)
2. Finding the surface height per bin (90th percentile of z + radius)
3. Fitting a line to the sloped region (20-80% of peak height)
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

1. **Non-cohesive angle**: 20-45° (consistent with μ = 0.5, μ_r = 1.0)
2. **Angle vs adhesion**: Monotonically increases with surface energy
3. **Lunar vs Earth**: Steeper piles on Moon for same adhesion (higher Bo)
4. **Significant cohesion effect**: High-adhesion angles > no-adhesion + 3°

## Expected Results

| Gravity | γ [mJ/m²] | Bo | Expected Angle |
|---------|-----------|-----|---------------|
| Earth | 0 | 0 | ~25-35° |
| Earth | 5 | 0.19 | ~27-37° |
| Earth | 20 | 0.76 | ~32-42° |
| Earth | 50 | 1.91 | ~40-55° |
| Moon | 0 | 0 | ~25-35° |
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
