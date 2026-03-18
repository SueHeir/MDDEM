# Packed Bed Thermal Conductivity Benchmark

## Physics

This benchmark validates DEM contact-based heat conduction through a packed bed of monodisperse spheres. Two flat walls at different temperatures (hot bottom at 400 K, cold top at 300 K) drive heat flow through particle-particle and particle-wall contacts in vacuum (no fluid phase).

The heat transfer model uses Hertzian contact mechanics:

```
Q = k · 2a · (T_j - T_i)
```

where `a = sqrt(R_eff · δ)` is the contact radius, `k` is the thermal conductivity, and `δ` is the overlap.

## Validation

The benchmark checks against the Zehner-Bauer-Schlunder (ZBS) framework for effective thermal conductivity in packed beds:

1. **Linear temperature profile** — At steady state with constant effective conductivity, Fourier's law predicts a linear temperature distribution (R² > 0.8).
2. **Steady-state convergence** — Wall heat flux variation < 10% in the last 20% of the simulation.
3. **Reasonable k_eff** — The effective thermal conductivity ratio k_eff/k_s falls in the physically expected range (0.001 to 1.0) for contact-only conduction in vacuum.
4. **Temperature bounds** — All particle temperatures stay between T_cold and T_hot.
5. **Average temperature** — The bed average temperature is near the midpoint (350 K).

## Setup

- **Particles**: 200 monodisperse steel spheres (R = 1 mm, ρ = 7800 kg/m³)
- **Domain**: 20 × 20 mm periodic in x,y; walls at z = 0 (bottom) and z = 22 mm (top)
- **Material**: k = 50 W/(m·K), cₚ = 500 J/(kg·K), E = 10 MPa (soft for fast packing)
- **Stage 1**: FIRE energy minimization to create a dense random packing under gravity
- **Stage 2**: Thermal conduction with wall temperatures T_hot = 400 K, T_cold = 300 K

## Running

```bash
# Run simulation (release mode recommended, ~2-3 minutes)
cargo run --release --example bench_thermal_bed -- examples/bench_thermal_bed/config.toml

# Validate results
cd examples/bench_thermal_bed
python3 validate.py

# Generate plots
python3 plot.py
```

## Expected Output

- `data/ThermalProfile.csv` — Per-atom z-position and temperature at steady state
- `data/WallHeatFlux.txt` — Time series of bottom wall heat flux
- `temperature_profile.png` — Steady-state T(z) with linear fit
- `wall_heat_flux.png` — Heat flux convergence to steady state
- `avg_temperature.png` — Average temperature evolution

## Plots

![Temperature Profile](temperature_profile.png)
![Wall Heat Flux](wall_heat_flux.png)
![Average Temperature](avg_temperature.png)

## References

- Zehner, P. & Schlunder, E.U. (1970). *Chemie Ingenieur Technik*, 42(14), 933-941.
- Bauer, R. & Schlunder, E.U. (1978). *Int. Chem. Eng.*, 18(2), 189-204.
- Batchelor, G.K. & O'Brien, R.W. (1977). *Proc. R. Soc. Lond. A*, 355, 313-333.
- Vargas, W.L. & McCarthy, J.J. (2001). *AIChE Journal*, 47(5), 1052-1059.
