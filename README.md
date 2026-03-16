# MDDEM

**Molecular Dynamics / Discrete Element Method in Rust**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE)

> **Note:** The initial working example of MDDEM was hand-written, but all code has since been touched and expanded by [Claude Code](https://claude.ai/claude-code). This project explores alternative coding patterns to LAMMPS, prioritizing ergonomics — composable plugins, dependency injections, typed configs, and a Rust-native API. 

## What is MDDEM?

MDDEM (pronounced like "Madem" without the 'a') is a particle simulation engine written in Rust. It supports both Discrete Element Method (DEM) for granular materials and Molecular Dynamics (MD) for continuous-potential systems like Lennard-Jones fluids. 

The design is built around **composability**. A dependency-injection scheduler (inspired by [Bevy](https://github.com/bevyengine/bevy)) and plugin system let you assemble simulations from independent, reusable pieces. Physics models, integrators, neighbor lists, and output formats are all plugins. Systems declare what resources they need as function arguments; the scheduler injects them automatically and handles execution order.

Configuration follows a two-tier approach. **Tier 1** is declarative TOML config for standard simulations — named fields, typed values, validated at startup. **Tier 2** is the Rust API for complex simulations — `main.rs` composes plugins, and custom systems are real functions with full type safety and IDE autocomplete. Both tiers can be mixed: the [hopper](examples/hopper/) example uses TOML config with custom Rust systems for runtime wall control.

## Why is MDDEM?

At first, I wanted to learn more about LAMMPS communication by rewriting it in Rust. I have also had many pain points editing LAMMPS code and working with LAMMPS scripts, and wanted to see if a scheduler with dependency injection would work for something like this (big fan of Bevy).

Now that Claude Code is good enough to debug MPI communication and neighbor list problems (it still struggles a lot with these, don't we all), I have expanded the scope, hoping it can be a nice starting place for anyone wanting to try coding an MD or DEM simulation in Rust. **I would not trust this code for anything you want to publish.** View it as a playground to test out whatever work you're doing. That being said, it's producing reasonable physics results, within 2-3% of the performance of LAMMPS single-core and within 10% on MPI for an LJ 12-6 fluid.

If you're unsure about why this is even a repo, I would look at the [hopper](examples/hopper/) example first. It's very simple, but shows how easy it is to add your own code into the mix.

## Where is MDDEM?

I am still unsure about the TOML input for this. I think having real examples to test the balance between compiling a new executable and having it be a setting in a TOML file will be a constant battle. Maybe JSON files? A Python wrapper is interesting, but I think that can be an external crate as we want to avoid dependency hell. There is a real opportunity here to come up with a solution that works way better than LAMMPS scripts. I also think parameter sweeps should be an option that is easily supported, but that TOML idea/replacement needs more thought.

I gave up on having every system (function handled by scheduler) be very readable (originally for educational purposes). Hot paths are full of unsafe code and many complex optimizations have led to this no longer being a goal of MDDEM; see the performance section. I have also removed nalgebra from the dependencies, so we only optionally depend on the Rust MPI wrapper. The Rust MPI wrapper is not feature-complete with MPI, but I don't think this is a huge deal right now. I think it's missing some features that might speed things up a little bit (non-blocking send/receive of arbitrary vec sizes).

We have a `#[derive(AtomData)]` proc macro for per-atom extension data — it generates pack/unpack, communication, and permutation code. It now supports `Vec<f64>`, `Vec<[f64; 3]>`, and `Vec<[f64; 4]>` fields, plus `#[forward]`, `#[reverse]`, and `#[zero]` attributes for comm and accumulator behavior. DemAtom uses it successfully. We also have `#[derive(StageEnum)]` for named simulation stages — it generates stage name mappings so `StageAdvancePlugin` can automatically advance `[[run]]` stages when state transitions occur. I'd be interested if there are other places where proc or declarative macros could improve ergonomics.

## Installation

MDDEM is not on crates.io yet. Add it as a git dependency:

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

## What's Included

### DEM (Granular)
- Hertz elastic normal contact with viscoelastic damping (LAMMPS `hertz/material` equivalent)
- Hooke linear spring contact model (alternative to Hertz, selectable via config)
- Mindlin tangential spring-history with Coulomb friction cap
- Rolling resistance (constant torque model)
- Twisting friction (resistance to relative spin about contact normal)
- SJKR cohesion (overlap-dependent attractive force)
- JKR adhesion (surface energy-based attraction with pull-off at separation)
- Velocity Verlet for translational + rotational degrees of freedom
- Quaternion-based orientation tracking
- Automatic timestep (5% of Rayleigh wave period)
- Configurable gravity body force
- Wall contacts: plane, cylinder, and sphere geometries with motion (static, constant velocity, oscillating, servo-controlled), toggleable at runtime
- Named material types with per-pair geometric-mean mixing
- Bonded particle model: auto-bonding, normal/tangential/bending spring-dashpot forces, breakage criteria
- FIRE energy minimization (adaptive timestep, stage-aware)
- Particle insertion: random (overlap-free), rate-based (periodic), file-based (CSV, LAMMPS dump, LAMMPS data)
- Size distributions: uniform, Gaussian, log-normal, weighted discrete
- Contact-based heat conduction with per-atom temperature

### MD (Molecular)
- Lennard-Jones 12-6 with cutoff, virial accumulator, and tail corrections; multi-type support with per-type parameters, mixing rules, and explicit pair coefficient overrides
- Nose-Hoover NVT thermostat (symmetric Liouville splitting)
- Langevin thermostat (stochastic friction + random force, group-aware)
- FCC lattice initialization with Maxwell-Boltzmann velocities
- Radial distribution function, mean square displacement, virial pressure

### Infrastructure
- Optional 3D MPI domain decomposition with multi-hop ghost forwarding
- Single-process mode with ghost atoms for periodic boundaries
- Bin-based neighbor lists with CSR storage and forward-only stencil
- Brute force and sweep-and-prune neighbor lists also available
- Named atom groups (`[[group]]`) with type and region filters; structured `Region` primitives (block, sphere, cylinder, plane, union, intersect)
- `PairCoeffTable<T>` generic NxN symmetric pair coefficient storage with geometric/arithmetic mixing
- `DumpRegistry` callback-based per-atom output extension for dump and VTP files
- Fixes: AddForce, SetForce, Freeze, MoveLinear, Viscous (group-based)
- Configurable thermo columns with group filtering (`ke/mobile`, `temp/mobile`)
- TOML config with `serde` validation and `deny_unknown_fields` on all config structs
- Dump files (CSV/binary), restart files (bincode/JSON), VTP visualization
- Generic restart serialization — any registered `AtomData` extension is automatically saved/restored
- Multi-stage runs with named stages (`#[derive(StageEnum)]`), `StageAdvancePlugin` for automatic stage advancement on state transitions, and per-stage config overrides (e.g., changing gravity or temperature between stages)
- `#[derive(AtomData)]` proc macro for zero-boilerplate per-atom extension structs

## Examples

| Example | Description | Run |
|---------|-------------|-----|
| [granular_basic](examples/granular_basic/) | 500-particle granular gas in a periodic box | `cargo run --example granular_basic -- examples/granular_basic/config.toml` |
| [granular_gas_benchmark](examples/granular_gas_benchmark/) | Haff's cooling law validation with LAMMPS comparison | `cargo run --example granular_gas_benchmark -- examples/granular_gas_benchmark/run_debug.toml` |
| [granular_gas_vdist](examples/granular_gas_vdist/) | Granular gas with velocity distribution analysis (Maxwell-Boltzmann comparison) | `cargo run --example granular_gas_vdist -- examples/granular_gas_vdist/config.toml` |
| [granular_shear](examples/granular_shear/) | Polydisperse shear flow between opposing walls with velocity distribution analysis | `cargo run --example granular_shear -- examples/granular_shear/config.toml` |
| [hopper](examples/hopper/) | 2D slot hopper with named stages (`StageEnum`), KE-based state transitions | `cargo run --example hopper -- examples/hopper/config.toml` |
| [dem_compression](examples/dem_compression/) | 3-stage DEM compression (insert → relax → compress) with per-stage config overrides | `cargo run --example dem_compression -- examples/dem_compression/config.toml` |
| [bond_basic](examples/bond_basic/) | Bonded particle model with auto-bonding and breakage | `cargo run --example bond_basic -- examples/bond_basic/config.toml` |
| [fire_packing](examples/fire_packing/) | FIRE energy minimization for particle packing | `cargo run --example fire_packing -- examples/fire_packing/config.toml` |
| [lj_argon](examples/lj_argon/) | LJ fluid validated against liquid Argon (RDF, MSD, pressure) | `cargo run --release --example lj_argon -- examples/lj_argon/config.toml` |
| [lj_langevin](examples/lj_langevin/) | LJ fluid with Langevin thermostat | `cargo run --release --example lj_langevin -- examples/lj_langevin/config.toml` |
| [lj_per_atom_energy](examples/lj_per_atom_energy/) | Per-atom energy output via DumpRegistry | `cargo run --release --example lj_per_atom_energy -- examples/lj_per_atom_energy/config.toml` |
| [group_freeze](examples/group_freeze/) | Group-based freeze fix demonstration | `cargo run --release --example group_freeze -- examples/group_freeze/config.toml` |
| [poiseuille_flow](examples/poiseuille_flow/) | Poiseuille flow with body force and frozen walls | `cargo run --release --example poiseuille_flow -- examples/poiseuille_flow/config.toml` |
| [lj_type_rdf](examples/lj_type_rdf/) | LJ fluid with type-filtered RDF measurement | `cargo run --release --example lj_type_rdf -- examples/lj_type_rdf/config.toml` |
| [polymer_chain](examples/polymer_chain/) | Bead-spring polymer chain with FENE bonds, R_ee and R_g | `cargo run --release --example polymer_chain -- examples/polymer_chain/config.toml` |
| [toml_single](examples/toml_single/) | Programmatic config — no TOML file needed | `cargo run --example toml_single` |

DEM examples include `validate.py` scripts for physics checks (Haff's law cooling, hopper settling). The `lj_argon` example validates against known liquid Argon properties (RDF, MSD, pressure) and generates diagnostic plots. Run `./validate.sh` to execute all tests and validations.

## Performance

Single-core LJ fluid benchmark comparing MDDEM to LAMMPS (29 Sep 2024 release). Identical physics: LJ 12-6 with cutoff 2.5 sigma, FCC lattice at rho\*=0.8442, Nose-Hoover NVT at T\*=1.44, full 6-component virial stress tensor, neighbor rebuild every 20 steps, 200 timesteps. Compiled with `--release` on Apple M1 Pro.

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | Ratio |
|--------:|---------------:|----------------:|------:|
|     108 |         23,305 |          31,145 | 1.34x |
|   1,000 |          2,562 |           2,943 | 1.15x |
|  10,000 |            277 |             293 | 1.06x |
|  32,000 |           88.1 |            92.7 | 1.05x |
| 100,920 |           28.4 |            29.1 | 1.02x |
| 202,612 |           14.1 |            14.5 | 1.03x |

LAMMPS is ~1.02-1.05x faster at scale, with consistent O(N) scaling in both codes. Virial stress is computed conditionally — only on thermo/measurement steps — so the force inner loop uses FMA without virial overhead on most steps. The force loop (~64% of MDDEM runtime) uses LAMMPS-style precomputed constants with a single reciprocal per pair, explicit FMA (`mul_add`) for force accumulation, and cached CSR pointers to prevent alias-induced reloads in the inner loop. The Nose-Hoover thermostat fuses velocity rescaling with Velocity Verlet integration to reduce array passes per timestep. The neighbor list build (~29% of runtime) uses CSR bins with a forward stencil, sorted position caches, and unsafe bounds-check elimination. The DI scheduler caches downcast pointers at system entry so `Res<T>`/`ResMut<T>` access is a direct dereference with no dynamic dispatch in hot loops.

MPI benchmark (4 processes, 2x2x1 decomposition) on the same hardware:

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | MDDEM Speedup | Ratio |
|--------:|---------------:|----------------:|--------------:|------:|
|     108 |         28,893 |          29,565 |         1.24x | 1.02x |
|   1,000 |          5,725 |           9,498 |         2.23x | 1.66x |
|  10,000 |            857 |           1,088 |         3.09x | 1.27x |
|  32,000 |            305 |             335 |         3.46x | 1.10x |
| 100,920 |           98.5 |             108 |         3.47x | 1.10x |

MDDEM MPI achieves 2.2-3.5x speedup over single-core at scale, with the ratio to LAMMPS narrowing to 1.10x at 32k+ atoms. Spatial sorting of atoms by bin is enabled in both single-core and MPI modes, improving cache locality for force and neighbor list computations. Communication uses per-dimension exchange with non-blocking sends, multi-hop ghost forwarding when needed, and lightweight ghost position updates between neighbor rebuilds.

## Code Size

One design goal of MDDEM is keeping the core code small, and unrelated code out of the way. The dependency-injection scheduler is key to this — physics systems are plain functions that declare their inputs, so there's no base-class boilerplate, virtual dispatch plumbing, or manual resource wiring. A new force model is just a function and a plugin registration. 

Here's a rough comparison of the code needed to run a simple LJ fluid simulation:

| | **MDDEM** | **LAMMPS** |
|---|---|---|
| LJ fluid core | ~11,200 lines (Rust) | ~30,000–40,000 lines (C++) |
| Full codebase | ~19,000 lines | ~190,000 lines (core `src/`, no packages) |

**Caveats:** This is an approximate comparison, not a rigorous benchmark. LAMMPS has 30+ years of development, supports far more features, and its core files include infrastructure (e.g., `variable.cpp` at ~5,000 lines for a full expression parser) that MDDEM sidesteps by using TOML config with `serde`. LAMMPS also has deep class hierarchies with virtual dispatch that add lines but provide extensibility MDDEM handles differently via plugins. The MDDEM line count includes tests (~20% of crate code); LAMMPS tests live in a separate directory. Lines of code is a crude metric — it doesn't capture complexity, correctness, or capability.

The MDDEM breakdown for an LJ fluid:

| Crate | Lines | Role |
|---|---|---|
| `mddem_core` | 3,940 | Atom, domain, comm, config, groups, regions |
| `mddem_scheduler` | 2,270 | DI scheduler, `Res`/`ResMut`, ordering |
| `mddem_neighbor` | 1,100 | Bin-based neighbor lists |
| `mddem_print` | 1,100 | Thermo, dump, restart, VTP output |
| `md_thermostat` | 590 | Nose-Hoover NVT + Langevin |
| `md_lj` | 560 | LJ 12-6 pair force |
| `mddem_app` | 510 | App container, plugin trait |
| `mddem_derive` | 450 | `#[derive(AtomData)]` proc macro |
| `md_lattice` | 380 | FCC lattice init |
| `mddem_verlet` | 154 | Velocity Verlet integrator |
| `mddem` | 105 | Plugin group definitions |
| **Total** | **~11,200** | |

The scheduler (`mddem_scheduler`) is the largest single piece, but it's pure infrastructure shared by every simulation type. The actual LJ physics is 560 lines. Adding a new force model doesn't require touching any of the infrastructure code — you write a function, register it in a plugin, and the scheduler handles the rest.

## Roadmap

Planned features, organized by implementation wave:

### Completed
1. ~~**Groups**~~ — Named atom subsets (`[[group]]`) for selective operations
2. ~~**Custom thermo computes**~~ — Configurable thermo columns with group filtering
3. ~~**Langevin thermostat**~~ — Stochastic friction + random force
4. ~~**AddForce / SetForce / Freeze / MoveLinear**~~ — Group-based atom manipulation fixes
5. ~~**Virial stress tensor**~~ — Full 6-component symmetric tensor shared across LJ, bond, and contact forces
6. ~~**Bonded particle model**~~ — Auto-bonding, normal/tangential/bending spring-dashpot forces, breakage criteria
7. ~~**Named stages**~~ — `#[derive(StageEnum)]` with `StageAdvancePlugin` for automatic `[[run]]` stage advancement on state transitions, plus per-stage config overrides
8. ~~**FIRE energy minimization**~~ — Adaptive timestep, stage-aware, can coexist with Velocity Verlet in multi-stage runs
9. ~~**Rolling resistance**~~ — Constant torque model opposing rolling motion
10. ~~**Cohesion models**~~ — SJKR (overlap-dependent) and JKR (surface energy-based adhesion with pull-off)
11. ~~**Particle insertion overhaul**~~ — Size distributions, rate-based insertion, file-based (CSV, LAMMPS dump, LAMMPS data)
12. ~~**Wall motion**~~ — Constant velocity, oscillating, servo-controlled walls

### Planned
- **Shrink-wrap boundaries** — Auto-expanding domain per axis
- **NPT barostat** — Nose-Hoover pressure control with box rescaling
- **Fix deform** — Box size change over time
- **Fix ave/time** — Running time averages to file
- **Short-range Coulomb** — `k_e*q_i*q_j/r^2` with cutoff (no PPPM)

Every feature ships with an example and validation against analytical solutions or LAMMPS.

## Out of Scope

These are specialized features that won't be in core. Users can write plugins for them:

- EAM / Tersoff / many-body potentials
- ReaxFF / ML potentials (SNAP, ACE)
- Angles / dihedrals
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
| [`mddem`](crates/mddem/) | Umbrella crate: `CorePlugins`, `LJDefaultPlugins`, `GranularDefaultPlugins`, prelude re-exports |
| [`mddem_scheduler`](crates/mddem_scheduler/) | DI scheduler, resources, schedule sets, ordering, run conditions, states |
| [`mddem_app`](crates/mddem_app/) | App, SubApp, Plugin, PluginGroup, StatesPlugin |
| [`mddem_core`](crates/mddem_core/) | TOML config, domain decomposition, communication, atom data, pair coefficients, regions, groups, run management |
| [`mddem_neighbor`](crates/mddem_neighbor/) | Neighbor lists: brute force, sweep-and-prune, bin-based |
| [`mddem_verlet`](crates/mddem_verlet/) | Velocity Verlet translational integration |
| [`mddem_print`](crates/mddem_print/) | Thermo (configurable columns), VTP, dump files, restart files, `DumpRegistry` |
| [`mddem_derive`](crates/mddem_derive/) | `#[derive(AtomData)]` and `#[derive(StageEnum)]` proc macros |
| [`mddem_fixes`](crates/mddem_fixes/) | AddForce, SetForce, Freeze, MoveLinear, Viscous fixes; Gravity body force |
| [`mddem_fire`](crates/mddem_fire/) | FIRE energy minimization (adaptive timestep, stage-aware) |
| [`dem_atom`](crates/dem_atom/) | Per-atom DEM data, `MaterialTable`, particle insertion (random/rate/file), radius distributions |
| [`dem_granular`](crates/dem_granular/) | Hertz-Mindlin and Hooke contact (with rolling resistance, twisting friction, SJKR cohesion, JKR adhesion), rotational dynamics, granular temperature |
| [`dem_wall`](crates/dem_wall/) | Plane, cylinder, and sphere wall contact forces with motion (static, constant velocity, oscillating, servo) |
| [`dem_thermal`](crates/dem_thermal/) | Contact-based heat conduction with per-atom temperature |
| [`dem_bond`](crates/dem_bond/) | Bonded particle model: auto-bonding, normal/tangential/bending forces, breakage |
| [`mddem_velocity_distribution`](crates/mddem_velocity_distribution/) | Velocity distribution analysis: speed histograms, Maxwell-Boltzmann comparison, kurtosis, per-component PDFs |
| [`md_lj`](crates/md_lj/) | LJ 12-6 pair force with virial, tail corrections, and multi-type support |
| [`md_thermostat`](crates/md_thermostat/) | Nose-Hoover NVT and Langevin thermostats (group-aware) |
| [`md_lattice`](crates/md_lattice/) | FCC lattice initialization |
| [`md_bond`](crates/md_bond/) | Harmonic and FENE bond potentials for bead-spring models |
| [`md_polymer`](crates/md_polymer/) | Polymer chain initialization, R_ee and R_g measurements |
| [`md_measure`](crates/md_measure/) | RDF, MSD, virial pressure |
| [`md_type_rdf`](crates/md_type_rdf/) | Type-filtered RDF: g(r) between specific atom type pairs |
| [`mddem_test_utils`](crates/mddem_test_utils/) | Shared test helpers |

A simulation is composed by adding plugin groups to an `App`:

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
```

`CorePlugins` bundles config loading, communication, domain decomposition, neighbor lists, groups, and output. `GranularDefaultPlugins` adds DEM atom data, particle insertion, Hertz-Mindlin contact forces (with rolling resistance and cohesion/adhesion), rotational dynamics, Velocity Verlet, and granular temperature. `LJDefaultPlugins` adds FCC lattice, LJ forces, Nose-Hoover thermostat (with fused Velocity Verlet integration), and measurements. Individual plugins can be added separately for custom configurations — add `VelocityVerletPlugin` explicitly when not using a thermostat that provides integration.

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
