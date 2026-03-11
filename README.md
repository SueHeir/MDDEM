# MDDEM

**Molecular Dynamics / Discrete Element Method in Rust**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE)

> **Note:** The initial working example of MDDEM was hand-written, but all code has since been touched and expanded by [Claude Code](https://claude.ai/claude-code). This project explores alternative coding patterns to LAMMPS, prioritizing ergonomics — composable plugins, typed configs, and a Rust-native API — over raw feature count or performance. On a personal note, I am not a MD person, my background is in DEM. When you see "Identical physics: LJ 12-6 with cutoff 2.5" I do not know what 12-6 is, I did not check. If you are an MD person don't be shocked that the MD side doesn't make sense.

## What is MDDEM?

MDDEM (pronounced like "Madem" without the 'a') is a particle simulation engine written in Rust. It supports both Discrete Element Method (DEM) for granular materials and Molecular Dynamics (MD) for continuous-potential systems like Lennard-Jones fluids. MPI parallelization is optional and feature-gated — MDDEM builds and runs on a single process without it.

The design is built around **composability**. A dependency-injection scheduler (inspired by [Bevy](https://github.com/bevyengine/bevy)) and plugin system let you assemble simulations from independent, reusable pieces. Physics models, integrators, neighbor lists, and output formats are all plugins. Systems declare what resources they need as function arguments; the scheduler injects them automatically and handles execution order.

Configuration follows a two-tier approach. **Tier 1** is declarative TOML config for standard simulations — named fields, typed values, validated at startup. **Tier 2** is the Rust API for complex simulations — `main.rs` composes plugins, and custom systems are real functions with full type safety and IDE autocomplete. Both tiers can be mixed: the [hopper](examples/hopper/) example uses TOML config with custom Rust systems for runtime wall control.

## Installation

MDDEM is not on crates.io. Add it as a git dependency:

```toml
[dependencies]
mddem = { git = "https://github.com/SueHeir/MDDEM" }
```

To build with MPI support (default), you need an MPI implementation installed (e.g., OpenMPI, MPICH). To build without MPI:

```toml
[dependencies]
mddem = { git = "https://github.com/SueHeir/MDDEM", default-features = false }
```

To clone and work on MDDEM directly:

```bash
git clone https://github.com/SueHeir/MDDEM.git
cd MDDEM
cargo build --release
```

## Quick Start

### DEM: Granular Gas

**`main.rs`**
```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
```

**`config.toml`**
```toml
[domain]
x_high = 0.025
y_high = 0.025
z_high = 0.025
periodic_x = true
periodic_y = true
periodic_z = true

[neighbor]
skin_fraction = 1.1
bin_size = 0.005

[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4

[[particles.insert]]
material = "glass"
count = 500
radius = 0.001
density = 2500.0
velocity = 0.5

[run]
steps = 10000
thermo = 100
```

### MD: Lennard-Jones Fluid

**`main.rs`**
```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
    app.start();
}
```

**`config.toml`**
```toml
[domain]
x_high = 10.06
y_high = 10.06
z_high = 10.06
periodic_x = true
periodic_y = true
periodic_z = true

[neighbor]
skin_fraction = 1.12
bin_size = 1.0

[lattice]
style = "fcc"
density = 0.85
temperature = 0.85
mass = 1.0
skin = 1.25

[lj]
epsilon = 1.0
sigma = 1.0
cutoff = 2.5

[thermostat]
temperature = 0.85
coupling = 1.0

[run]
steps = 100000
thermo = 1000
```

### Running

```bash
# Single process
cargo run --release -- config.toml

# With MPI (build first, then launch)
cargo build --release
mpiexec -n 4 ./target/release/my_simulation config.toml

# Print the compiled schedule (Graphviz DOT)
cargo run --release -- config.toml --schedule
```

## Configuration

Each plugin owns its TOML section. The full set of config sections:

| Section | Purpose |
|---|---|
| `[comm]` | MPI processor grid (`processors_x/y/z`) |
| `[domain]` | Box bounds, periodic boundaries |
| `[neighbor]` | Skin fraction, bin size |
| `[[dem.materials]]` | Named material types (Young's modulus, Poisson ratio, restitution, friction) |
| `[[particles.insert]]` | Random particle insertion (material, count, radius, density, velocity) |
| `[gravity]` | Body force components (`gx`, `gy`, `gz`) |
| `[[wall]]` | Plane wall contacts (point, normal, optional bounds, material) |
| `[lj]` | LJ 12-6 parameters (epsilon, sigma, cutoff) |
| `[thermostat]` | Nose-Hoover NVT (temperature, coupling) |
| `[lattice]` | FCC initialization (density, temperature, mass) |
| `[measure]` | RDF, MSD, pressure measurement intervals |
| `[run]` or `[[run]]` | Timesteps, thermo interval, per-stage overrides |
| `[dump]` | Atom data output (interval, format: text/binary) |
| `[restart]` | Restart files (interval, format: bincode/json) |
| `[vtp]` | VTP visualization output |

Multiple materials and insert blocks are supported — mixed-material contacts use geometric-mean mixing (LAMMPS convention). Multi-stage runs use `[[run]]` arrays where each stage can override dump/restart/VTP intervals. See the [examples](examples/) for full configs.

## What's Included

### DEM (Granular)
- Hertz elastic normal contact with viscoelastic damping (LAMMPS `hertz/material` equivalent)
- Mindlin tangential spring-history with Coulomb friction cap
- Velocity Verlet for translational + rotational degrees of freedom
- Quaternion-based orientation tracking
- Automatic timestep (5% of Rayleigh wave period)
- Configurable gravity body force
- General plane wall contacts, toggleable at runtime
- Named material types with per-pair mixing

### MD (Molecular)
- Lennard-Jones 12-6 with cutoff, virial accumulator, and tail corrections
- Nose-Hoover NVT thermostat (symmetric Liouville splitting)
- FCC lattice initialization with Maxwell-Boltzmann velocities
- Radial distribution function, mean square displacement, virial pressure

### Infrastructure
- Optional 3D MPI domain decomposition with corner-complete ghost forwarding
- Single-process mode with ghost atoms for periodic boundaries
- Bin-based neighbor lists with CSR storage and forward-only stencil
- Brute force and sweep-and-prune neighbor lists also available
- TOML config with `serde` validation and `deny_unknown_fields` on all config structs
- Dump files (CSV/binary), restart files (bincode/JSON), VTP visualization
- Generic restart serialization — any registered `AtomData` extension is automatically saved/restored
- Multi-stage runs with per-stage output control
- `#[derive(AtomData)]` proc macro for zero-boilerplate per-atom extension structs

## Examples

| Example | Description | Run |
|---------|-------------|-----|
| [granular_basic](examples/granular_basic/) | 500-particle granular gas in a periodic box | `cargo run --example granular_basic -- examples/granular_basic/config.toml` |
| [granular_gas_benchmark](examples/granular_gas_benchmark/) | Haff's cooling law validation with LAMMPS comparison | `cargo run --example granular_gas_benchmark -- examples/granular_gas_benchmark/config.toml` |
| [hopper](examples/hopper/) | 2D slot hopper with angled walls, gravity, and simulation states | `cargo run --example hopper -- examples/hopper/config.toml` |
| [lj_argon](examples/lj_argon/) | LJ fluid validated against liquid Argon (RDF, MSD, pressure) | `cargo run --release --example lj_argon -- examples/lj_argon/config.toml` |
| [toml_single](examples/toml_single/) | Programmatic config — no TOML file needed | `cargo run --example toml_single` |

DEM examples include `validate.py` scripts for physics checks (Haff's law cooling, hopper settling). The `lj_argon` example validates against known liquid Argon properties (RDF, MSD, pressure) and generates diagnostic plots. Run `./validate.sh` to execute all tests and validations.

## Performance

Single-core LJ fluid benchmark comparing MDDEM to LAMMPS (29 Sep 2024 release). Identical physics: LJ 12-6 with cutoff 2.5 sigma, FCC lattice at rho\*=0.8442, Nose-Hoover NVT at T\*=1.44, neighbor rebuild every 20 steps, 200 timesteps. Compiled with `--release` on Apple M1 Pro. RDF/MSD disabled in both codes for fair comparison.

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | Ratio |
|--------:|---------------:|----------------:|------:|
|     108 |         13,920 |          30,616 |  2.2x |
|   1,000 |          1,688 |           2,857 |  1.7x |
|  10,000 |          182.3 |             290 |  1.6x |
|  32,000 |           57.6 |            90.4 |  1.6x |
| 100,920 |           18.3 |            28.4 |  1.6x |

LAMMPS is ~1.6x faster at scale, with consistent O(N) scaling in both codes. The gap is primarily in the force loop (60% of MDDEM runtime), where LAMMPS benefits from decades of hand-tuned SIMD and cache optimization. The neighbor list build (22% of runtime) uses CSR bins with a forward stencil, sorted position caches, sorted neighbor indices for sequential cache access, and unsafe bounds-check elimination.

MPI benchmark (4 processes, 2x2x1 decomposition) on the same hardware:

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | Ratio |
|--------:|---------------:|----------------:|------:|
|     108 |         11,166 |          40,418 |  3.6x |
|   1,000 |          3,179 |           8,836 |  2.8x |
|  10,000 |          497.9 |           1,041 |  2.1x |
|  32,000 |          183.5 |           313.6 |  1.7x |
| 100,920 |           61.3 |           103.0 |  1.7x |

At scale (32k+ atoms), MDDEM achieves ~3.2x speedup on 4 cores vs single-core, while LAMMPS achieves ~3.5x — comparable parallel efficiency. At small atom counts, MPI overhead dominates and both codes show diminishing returns.

## Roadmap

Planned features, organized by implementation wave:

### Wave 1: Foundation
1. **Groups** — Named atom subsets (`[[group]]`) for selective operations (thermostat a region, freeze a surface)
2. **Custom thermo computes** — `ThermoCompute` trait + configurable thermo columns

### Wave 2: Small, High-Value
3. **Langevin thermostat** — Stochastic friction + random force
4. **AddForce / SetForce** — Constant external forces on groups
5. **Shrink-wrap boundaries** — Auto-expanding domain per axis

### Wave 3: Medium Complexity
6. **Energy minimization** — CG + FIRE, runs as a `[[run]]` stage
7. **Fix ave/time** — Running time averages to file

### Wave 4: Box Manipulation
8. **NPT barostat** — Nose-Hoover pressure control with box rescaling
9. **Fix deform** — Box size change over time

### Wave 5: New Physics
10. **Short-range Coulomb** — `k_e*q_i*q_j/r^2` with cutoff (no PPPM)
11. **Bonded particle model** — Bond lists between DEM spheres (parallel bonds, breakage criteria) for modeling cemented/rock-like materials

Every feature ships with an example and validation against analytical solutions or LAMMPS.

## Out of Scope

These are specialized features that won't be in core. Users can write plugins for them:

- EAM / Tersoff / many-body potentials
- ReaxFF / ML potentials (SNAP, ACE)
- Angles / dihedrals (bonds are planned — see roadmap)
- SHAKE / RATTLE constraints
- Units system (TOML config uses explicit units in docs)
- rRESPA multi-timescale integration
- PPPM / Ewald long-range electrostatics
- Rigid body dynamics
- GCMC, NEB, spin dynamics
- Triclinic (non-orthogonal) boxes
- Multiple dump formats (XYZ, DCD, NetCDF) — CSV + VTP + binary covers most needs

## Architecture

| Crate | Description |
|---|---|
| [`mddem`](crates/mddem/) | Umbrella crate: `CorePlugins`, `LJDefaultPlugins`, prelude re-exports |
| [`mddem_scheduler`](crates/mddem_scheduler/) | DI scheduler, resources, schedule sets, ordering, run conditions, states |
| [`mddem_app`](crates/mddem_app/) | App, SubApp, Plugin, PluginGroup, StatesPlugin |
| [`mddem_core`](crates/mddem_core/) | TOML config, domain decomposition, communication, atom data, run management |
| [`mddem_neighbor`](crates/mddem_neighbor/) | Neighbor lists: brute force, sweep-and-prune, bin-based |
| [`mddem_verlet`](crates/mddem_verlet/) | Velocity Verlet translational integration |
| [`mddem_print`](crates/mddem_print/) | Thermo, VTP, dump files, restart files |
| [`mddem_derive`](crates/mddem_derive/) | `#[derive(AtomData)]` proc macro |
| [`dem_atom`](crates/dem_atom/) | Per-atom DEM data, `MaterialTable`, material config |
| [`dem_atom_insert`](crates/dem_atom_insert/) | Random particle insertion with overlap checking |
| [`dem_granular`](crates/dem_granular/) | Hertz normal, Mindlin tangential, rotational dynamics, granular temperature |
| [`dem_gravity`](crates/dem_gravity/) | Gravity body force |
| [`dem_wall`](crates/dem_wall/) | Plane wall contact forces |
| [`md_lj`](crates/md_lj/) | LJ 12-6 pair force with virial and tail corrections |
| [`md_thermostat`](crates/md_thermostat/) | Nose-Hoover NVT thermostat |
| [`md_lattice`](crates/md_lattice/) | FCC lattice initialization |
| [`md_measure`](crates/md_measure/) | RDF, MSD, virial pressure |

A simulation is composed by adding plugin groups to an `App`:

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
```

`CorePlugins` bundles config loading, communication, domain decomposition, neighbor lists, Velocity Verlet, and output. `GranularDefaultPlugins` adds DEM atom data, insertion, contact forces, gravity, and walls. `LJDefaultPlugins` adds FCC lattice, LJ forces, thermostat, and measurements. Individual plugins can be added separately for custom configurations.

## Testing

```bash
cargo test --workspace                    # All tests (with MPI)
cargo test --workspace --no-default-features  # All tests (without MPI)
cargo test -p dem_granular                # Single crate

./validate.sh                             # Full validation: tests + examples + physics checks
./validate.sh --long                      # Production-length runs with full physics validation
./validate.sh --dem                       # DEM examples only
./validate.sh --md                        # MD examples only
```

Unit tests cover: single-process communication, domain decomposition, Velocity Verlet integration, neighbor lists (all three algorithms), Hertz/Mindlin contact forces, rotational dynamics, material mixing, LJ force and virial, Nose-Hoover thermostat, FCC lattice insertion, and RDF/MSD measurement.

Physics validation scripts (`validate.py` per example) check simulation output against analytical solutions: Haff's cooling law for granular gas, settling behavior for hopper, and RDF/MSD/pressure against known liquid Argon properties for LJ fluid.

## Contributing

Vibe-coded PRs are accepted, provided the contributor is qualified in the relevant domain and has personally reviewed the code being submitted. Report issues at [github.com/SueHeir/MDDEM/issues](https://github.com/SueHeir/MDDEM/issues).

## License

MIT OR Apache-2.0
