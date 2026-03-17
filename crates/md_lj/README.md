# md_lj: Lennard-Jones 12-6 Pair Potential

Fast, production-ready Lennard-Jones 12-6 pair force plugin for MDDEM molecular dynamics.

## Overview

Implements the standard LJ 12-6 potential: `V(r) = 4ε[(σ/r)¹² − (σ/r)⁶]` with cutoff. Features include single/multi-type interactions with mixing rules, precomputed pair coefficients for speed, virial stress accumulation, and analytical long-range tail corrections.

## Key Types & Resources

- **`LJForcePlugin`**: Register this plugin with your app; depends on `NeighborPlugin`
- **`LJConfig`**: TOML config holder (epsilon, sigma, cutoff, mixing rule, per-type/pair overrides)
- **`LJPairCoeffs`**: Precomputed coefficients (lj1, lj2, cutoff2) for fast force evaluation
- **`LJPairTable`**: Symmetric NxN pair coefficient table
- **`LJTailCorrections`**: Energy and pressure corrections beyond cutoff

## TOML Configuration

**Single-type (default):**
```toml
[lj]
epsilon = 1.0
sigma = 1.0
cutoff = 2.5
```

**Multi-type with mixing:**
```toml
[lj]
cutoff = 2.5
mixing = "geometric"  # or "arithmetic"

[[lj.types]]
epsilon = 1.0
sigma = 1.0

[[lj.types]]
epsilon = 0.5
sigma = 1.2

[[lj.pair_coeffs]]  # optional overrides
types = [0, 1]
epsilon = 0.8
sigma = 1.1
```

## Usage

```rust
app.add_plugins(LJForcePlugin);
```

The plugin loads config from `[lj]`, builds the pair table at setup, computes tail corrections (first stage only), and evaluates forces with virial accumulation in the Force schedule. Automatically registers `VirialStressPlugin`.

## Performance

- Raw pointer arithmetic eliminates bounds checks in inner loop
- Precomputed coefficients avoid expensive power calculations
- Half neighbor list (Newton's third law) reduces pair evaluations by ~50%
- Virial accumulation integrated into force loop (no extra pair pass)
