# BPM fiber tensile test

Pulls an 11-sphere fiber (10 bonds) along +x at constant velocity, records the
stress-strain curve, and fits the slope to recover the Young's modulus set in
`[bonds]`.

## Setup

- 11 glass-like spheres, radius `r = 1 mm`, spaced `2r` apart on the x-axis
- 10 bonds, auto-bonded at setup (touching pairs)
- Bond radius `r_b = r` (ratio 1.0) → area `A = π r²`, length `L = 2 r`
- `E = 1 GPa`, `G = 400 MPa` (ν = 0.25)
- Per-bond axial stiffness `K_n = E · A / L`
- Left end frozen, right end moved at `vx = 0.1 m/s` (strain rate ≈ 5 /s)
- Critical damping on all four channels to suppress transients

## Run

```bash
cargo run --release --example bond_fiber_tensile --no-default-features -- \
    examples/bond_fiber_tensile/config.toml

python3 examples/bond_fiber_tensile/validate.py
```

The first command produces `data/fiber_tensile.csv`; the second fits the linear
slope of σ(ε) and compares it to the input `E`.

## What's measured

Every 100 steps we record:

| column          | meaning                                               |
|-----------------|-------------------------------------------------------|
| `strain_global` | `(x_10 − x_0 − L₀) / L₀`, end-to-end kinematic strain |
| `strain_mid`    | stretch of the middle bond (tags 6↔7) divided by `r₀` |
| `force_mid`     | `K_n · δ_mid` — axial tension in the middle bond      |
| `stress_mid`    | `force_mid / A` — tensile stress                      |

Uniform strain distribution (`ε_local / ε_global ≈ 1`) and the σ = E · ε slope
together validate that the four-channel BPM propagates axial load correctly.
