# BPM fiber tensile test — overlap variant

Same tensile test as `bond_fiber_tensile/`, but the spheres are only **1
radius apart** (center-to-center), i.e. every adjacent pair **overlaps by 1
mm** — 50 % of a diameter. If Hertz contact forces were not correctly
suppressed between bonded pairs they would be on the order of hundreds of
newtons per pair and the fitted Young's modulus would be nonsense.

## Setup

- 11 glass-like spheres, radius `r = 1 mm`, spaced `r` apart → `L₀ = 10 mm`
- `bond_tolerance = 0.6` so only immediate neighbours (1 mm) auto-bond;
  `i ↔ i+2` pairs (2 mm) do not
- 10 direct bonds; 9 additional pairs sit within the neighbour cutoff but are
  skipped by the **1-3 exclusion** (shared bonded neighbour)
- Bond geometry `r_b = R = 1 mm`, `L = r₀ = 1 mm` →
  `K_n = E·A/L = π · 10⁶ N/m` (2× the touching case)
- Same material (`E = 1 GPa`, `G = 400 MPa`, ν = 0.25) and damping as the
  touching-fiber test

## Run

```bash
cargo run --release --example bond_fiber_tensile --no-default-features -- \
    examples/bond_fiber_tensile_overlap/config.toml

python3 examples/bond_fiber_tensile/validate.py \
    examples/bond_fiber_tensile_overlap/data/fiber_tensile.csv
```

## Result

| quantity                 | value                 |
|--------------------------|-----------------------|
| overlap per pair         | 1 mm (50 % of D)      |
| neighbour pairs          | 19  (10 bond + 9 1-3) |
| strain range             | 0 → 2.99 %            |
| E input                  | 1.00000 × 10⁹ Pa      |
| E fit (σ / ε slope)      | 1.00005 × 10⁹ Pa      |
| relative error           | 0.005 %               |
| ε_local / ε_global       | 1.000087              |

Identical to the touching-fiber fit (0.005 % error), confirming that
`BondStore::are_excluded` correctly skips contact forces on every overlapping
bonded and 1-3 pair.
