# dem_bond

Bonded Particle Model (BPM) force plugin for MDDEM.

## Overview

`dem_bond` treats each bond between two particles as a **solid cylindrical
beam** and resists four independent deformation channels: axial stretch /
compression, transverse shear, twist (torsion about the bond axis), and
bending (rotation perpendicular to the bond axis). Bond stiffness can be
derived from material properties (Young's / shear modulus + bond radius) or
supplied directly; damping is specified as a critical-damping ratio; and
breakage uses beam-stress failure criteria.

The implementation follows the Fortran reference in
`fortran_bpm_info/fortran-codes-main/Bonded_Network/demcfd.f90` (subroutine
`bonds`). Plastic deformation is **not** currently implemented.

## Bond geometry

For each bond, the cross-section is a solid cylinder of radius

```
r_b = bond_radius_ratio · min(R_i, R_j)
```

giving

| quantity | expression         | meaning                        |
|----------|--------------------|--------------------------------|
| `A`      | `π r_b²`           | cross-sectional area           |
| `J`      | `½ π r_b⁴`         | polar second moment (torsion)  |
| `I`      | `¼ π r_b⁴ = ½ J`   | second moment for bending      |
| `L`      | `r₀`               | equilibrium bond length        |

## Four-channel force model

| Channel               | Stiffness (material mode) | Force / moment                                              |
|-----------------------|----------------------------|-------------------------------------------------------------|
| Normal                | `K_n   = E · A / L`        | `F_n = (K_n · δ + γ_n · v_n) n̂`                           |
| Shear                 | `K_t   = G · A / L`        | `F_t = −K_t · Δs − γ_t · v_t`                              |
| Twist (torsion)       | `K_tor = G · J / L`        | `M_tor = −K_tor · (Δθ · n̂) n̂ − γ_tor · (ω_rel · n̂) n̂` |
| Bending               | `K_bend = E · I / L`       | `M_bend = −K_bend · (Δθ − (Δθ · n̂) n̂) − γ_bend · (ω_rel − (ω_rel · n̂) n̂)` |

with `δ = |r_ij| − r₀`, `n̂` = unit bond axis from *i* to *j*, `Δs` the
accumulated shear displacement (re-projected ⊥ to `n̂` each step), and `Δθ`
the accumulated relative rotation angle (split into twist / bending parts
on-the-fly each step).

The shear force is applied at the bond mid-point (`L/2` from each centre),
which produces a lever-arm torque `τ = ±(L/2) n̂ × F_t` on both particles.

## Damping

Per-channel damping uses a critical-damping ratio `β ∈ [0, 1]`:

```
γ   = 2 β √(m* · K_eff)        (force channels:   normal, shear)
γ_M = 2 β √(I_rot* · K_eff)    (moment channels:  twist, bending)
```

with reduced mass `m* = m_i m_j / (m_i + m_j)` and reduced moment of inertia
`I_rot* = I_i I_j / (I_i + I_j)`. For a solid sphere `I_i = ⅖ m_i R_i²`.
Each channel accepts an optional raw-`γ` override that replaces the β-based
formula.

## Breakage (beam-stress criterion)

A bond breaks when either the tensile or shear failure stress is exceeded
at the extreme fibre of the beam:

```
σ = F_n / A  +  2 |M_bend| r_b / J        →  break if σ > σ_max
τ = |F_t| / A  +  |M_tor| r_b / J         →  break if τ > τ_max
```

Broken bonds are removed from both partners' bond lists after the force loop.
Without `sigma_max` / `tau_max` the bonds are unbreakable.

## Configuration

Two ways to set stiffness — **material mode** (paper-standard beam theory)
or **direct mode** (scalar N/m and N·m/rad knobs). Material mode wins per
channel when `youngs_modulus` / `shear_modulus` is set.

```toml
[bonds]
auto_bond = true
bond_tolerance = 1.001
bond_radius_ratio = 1.0

# Material mode:
youngs_modulus = 1.0e9     # E (Pa) → K_n, K_bend
shear_modulus  = 4.0e8     # G (Pa) → K_t, K_tor

# (Direct overrides — used when the corresponding modulus is not set)
# normal_stiffness  = 0.0   # N/m
# shear_stiffness   = 0.0   # N/m
# twist_stiffness   = 0.0   # N·m/rad
# bending_stiffness = 0.0   # N·m/rad

# Critical-damping ratios:
beta_normal  = 1.0
beta_shear   = 1.0
beta_twist   = 1.0
beta_bending = 1.0

# (Raw-γ overrides — bypass the β-based formula)
# normal_damping  = 0.0
# shear_damping   = 0.0
# twist_damping   = 0.0
# bending_damping = 0.0

# Beam-stress breakage (omit to leave bonds unbreakable):
sigma_max = 5.0e7           # Pa, tensile + bending
tau_max   = 3.0e7           # Pa, shear + torsion

# Load bonds from a LAMMPS data file instead of auto-bonding:
# file = "bonds.lammps"
# format = "lammps_data"
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(DemBondPlugin);
app.start();
```

Granular contact forces are automatically skipped between bonded pairs
(via `BondStore::are_excluded`), so the bond is the sole interaction on
those neighbours until it breaks.

## Key types

- **`BondConfig`** — deserialised `[bonds]` section
- **`BondHistoryStore`** — per-atom list of `BondHistoryEntry` (shear
  displacement `delta_t` and rotation angle `delta_theta`); implements
  `AtomData` for MPI communication and atom reordering
- **`BondMetrics`** — step-level strain average and cumulative bonds-broken
  count, published to thermo as `bond_strain` and `bonds_broken`
- **`DemBondPlugin`** — registers resources, auto-bonding / file-loading
  setup systems, and the per-step force computation

## Validation: fiber tensile test

`examples/bond_fiber_tensile/` pulls an 11-sphere fiber (10 bonds) at
constant velocity and fits σ vs ε to recover the input Young's modulus.

Setup: radius 1 mm, density 2500 kg/m³, bond radius = particle radius,
`E = 1 GPa`, `G = 400 MPa` (ν = 0.25), critical damping on all four
channels, left end frozen, right end pulled at `v_x = 0.1 m/s`
(strain rate ≈ 5 /s, fully quasi-static).

Result on 30 000 steps (`dt = 1e-7 s`, ε from 0 → 1.5%):

| quantity                 | value                      |
|--------------------------|----------------------------|
| E input                  | 1.00000 × 10⁹ Pa           |
| E fit (σ / ε slope)      | 1.00005 × 10⁹ Pa           |
| relative error           | 0.005 %                    |
| ε_local / ε_global       | 1.000086 (uniform strain)  |

The match confirms that (a) `K_n = E·A/L` is applied correctly per bond,
(b) load propagates cleanly through the 10-bond chain, and (c) strain
distributes uniformly at quasi-static loading.

```bash
cargo run --release --example bond_fiber_tensile --no-default-features -- \
    examples/bond_fiber_tensile/config.toml
python3 examples/bond_fiber_tensile/validate.py
```

### Overlap variant — contact-suppression check

`examples/bond_fiber_tensile_overlap/` re-runs the test with the spheres
only **1 radius apart** (center-to-center) — every adjacent pair overlaps
by 1 mm. Hertz contact forces between bonded pairs would be on the order
of hundreds of newtons per pair if not suppressed. All 19 neighbour
candidates (10 direct bonds + 9 shared-neighbour "1-3" pairs) are skipped
by `BondStore::are_excluded`, so the result is identical: E fit within
0.005 %.

| variant             | bond length L | K_n        | E fit            | error   |
|---------------------|---------------|------------|------------------|---------|
| touching (`2 r`)    | 2.000 mm      | 1.571 MN/m | 1.00005 × 10⁹ Pa | 0.005 % |
| overlapping (`1 r`) | 1.000 mm      | 3.142 MN/m | 1.00005 × 10⁹ Pa | 0.005 % |

```bash
cargo run --release --example bond_fiber_tensile --no-default-features -- \
    examples/bond_fiber_tensile_overlap/config.toml
python3 examples/bond_fiber_tensile/validate.py \
    examples/bond_fiber_tensile_overlap/data/fiber_tensile.csv
```

## What's next

- 3-point bend (validates `K_bend = E·I/L` independently)
- Torsion test (validates `K_tor = G·J/L`)
- Plastic deformation (Fortran `plastic_bond_*` — softened stiffness past
  yield, persistent max-stress history)
