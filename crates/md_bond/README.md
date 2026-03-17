# md_bond

MD bond potentials for bead-spring polymer models in MDDEM.

## Overview

This crate provides pairwise bond forces and optional three-body angle bending forces for molecular dynamics simulations. Bond topology is read from `BondStore` and angle topology from `AngleStore`.

## Bond Styles

### Harmonic
Simple spring with equilibrium length `r0`:
```
U(r) = k/2 (r - r0)²
```
Attractive when stretched, repulsive when compressed.

### FENE (Finitely Extensible Nonlinear Elastic)
Used in Kremer–Grest bead-spring models. Diverges at maximum extension `R0`, preventing chain crossing:
```
U(r) = -0.5 k R0² ln(1 - (r/R0)²)
```
Purely attractive (equilibrium at r=0). Diverges as r→R0. Often combined with LJ repulsion (WCA).

## Angle Bending (Cosine Potential)

For three consecutively bonded atoms i—j—k:
```
U(θ) = k_angle (1 - cos θ)
```
Minimum at θ=0 (straight). Set `k_angle=0.0` (default) for fully flexible chains.

## Configuration

```toml
[md_bond]
style = "fene"      # "harmonic" or "fene"
k = 30.0            # spring constant (energy/length²)
r0 = 1.5            # max extension R0 (FENE) or equilibrium length (harmonic)
k_angle = 0.0       # bond angle stiffness (0 = fully flexible)
```

### Example: Kremer–Grest Model
```toml
[md_bond]
style = "fene"
k = 30.0
r0 = 1.5
```

### Example: Stiff Chains
```toml
[md_bond]
style = "harmonic"
k = 100.0
r0 = 1.0
k_angle = 25.0
```

## Key Types

- **`MdBondConfig`**: TOML configuration (style, k, r0, k_angle)
- **`BondStyle`**: Enum for Harmonic or Fene dispatching
- **`MdBondPlugin`**: Registers force computation; auto-registers BondPlugin, VirialStressPlugin, and (if k_angle>0) AnglePlugin

## Features

- Energy-conserving force computation with Velocity Verlet
- Newton's 3rd law enforcement
- Virial stress accumulation for pressure calculation
- Minimum image convention for periodic domains
- Force capping beyond FENE maximum extension (with warning)
