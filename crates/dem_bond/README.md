# dem_bond

Elastic bond force models for DEM simulations in MDDEM.

## Overview

`dem_bond` adds bonded interactions between particle pairs. Each bond resists relative motion along three independent channels: normal (stretch/compression), tangential (sliding), and bending (rotation).

## Key Types

- **BondConfig**: Deserialized TOML configuration for bond stiffness, damping, and breakage thresholds
- **BondHistoryStore**: Tracks accumulated tangential displacement and relative rotation per bond (implements `AtomData` for MPI)
- **BondHistoryEntry**: Per-bond storage of `delta_t` (tangential displacement) and `delta_theta` (rotation angle)
- **BondMetrics**: Aggregates step-wise bond strain and breakage counts for output
- **DemBondPlugin**: Main plugin that registers resources and integrates force computation

## Force Model

| Channel | Force Equation | Notes |
|---------|---|---|
| **Normal** | `F_n = (k_n·δ + γ_n·v_n)·n̂` | δ = bond stretch, v_n = normal velocity |
| **Tangential** | `F_t = −(k_t·Δs + γ_t·v_t)` | Δs = accumulated displacement, history tracked |
| **Bending** | `M = −(k_bend·Δθ + γ_bend·ω_rel)` | Δθ = relative rotation, damped by angular velocity |

Bonds can break on normal strain (`|δ/r₀| > threshold`) or tangential displacement (`|Δs| > threshold`).

## TOML Configuration

```toml
[bonds]
auto_bond = true              # bond touching particles at setup
bond_tolerance = 1.001        # auto-bond tolerance multiplier
normal_stiffness = 1e7        # k_n (N/m)
normal_damping = 10.0         # γ_n (N·s/m)
tangential_stiffness = 5e6    # k_t (N/m)
tangential_damping = 5.0      # γ_t (N·s/m)
bending_stiffness = 1e4       # k_bend (N·m/rad)
bending_damping = 1.0         # γ_bend (N·m·s/rad)
break_normal_stretch = 0.1    # break on 10% strain (optional)
break_shear = 0.0005          # break on displacement (optional)
# file = "bonds.lammps"       # load from LAMMPS data file
# format = "lammps_data"      # file format
```

## Usage Example

Enable bonding in your TOML config:

```toml
[bonds]
auto_bond = true
normal_stiffness = 1e7
tangential_stiffness = 5e6
break_normal_stretch = 0.1
```

Add the plugin to your app:

```rust
use dem_bond::DemBondPlugin;

app.add_plugins(DemBondPlugin);
```

Bonds will be created at setup (auto-bonding or file-based), and forces computed each timestep with optional breakage tracking.
