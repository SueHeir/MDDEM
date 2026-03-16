# dem_bond

Bonded particle model for [MDDEM](https://github.com/SueHeir/MDDEM): inter-particle bonds with normal, tangential, and bending forces, plus breakage criteria.

## Features

- Auto-bonding of initially touching particles at setup
- Normal spring-dashpot bond forces (tension and compression)
- Tangential spring-dashpot with history tracking
- Bending moments from relative angular velocity
- Bond breakage by normal stretch or shear displacement thresholds
- Bond topology loading from LAMMPS data files
- Bond metrics output to thermo (average strain, bonds broken)

## Physics

### Normal Bond Force
Linear spring with damping:
- `F = k_n * delta + gamma_n * v_n` where `delta = dist - r0`
- Tensile (`delta > 0`) and compressive (`delta < 0`)
- Equilibrium length `r0` set at bond creation

### Tangential Bond Force
Spring-history with damping:
- Incremental tangential displacement `delta_t` integrated from relative tangential velocity
- `F_t = -(k_t * delta_t + gamma_t * v_t)`
- No Coulomb cap (bonds are cohesive, not frictional)
- Displacement projected to remain perpendicular to bond each step

### Bending Moment
Angular spring with damping:
- Incremental rotation `delta_theta` integrated from `(omega_j - omega_i) * dt`
- `M = -(k_bend * delta_theta + gamma_bend * omega_rel)`
- Applied as torque on both particles

### Bond Breakage
Bonds break when thresholds are exceeded (optional):
- `break_normal_stretch` — fractional strain `|delta| / r0`
- `break_shear` — tangential displacement magnitude `|delta_t|`

Broken bonds are removed after the force loop. Bond metrics (`bond_strain`, `bonds_broken`) are available as thermo columns.

## Configuration

```toml
[bonds]
auto_bond = true              # bond initially touching particles
bond_tolerance = 1.001        # distance multiplier for auto-bonding (sum of radii * tolerance)
normal_stiffness = 1e6        # N/m
normal_damping = 100.0        # Ns/m
tangential_stiffness = 5e5    # N/m
tangential_damping = 50.0     # Ns/m
bending_stiffness = 1e3       # Nm/rad
bending_damping = 10.0        # Nms/rad
# break_normal_stretch = 0.05  # fractional strain threshold
# break_shear = 0.001          # tangential displacement threshold

# Load bonds from LAMMPS data file (alternative to auto_bond)
# file = "data.lammps"
# format = "lammps_data"
```

## Bond Exclusions

Bonded pairs (1-2 neighbors) and atoms sharing a common bonded neighbor (1-3 neighbors) are excluded from contact force calculations via `BondStore::are_excluded`.

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(DemBondPlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
