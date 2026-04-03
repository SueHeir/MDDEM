# MDDEM

**Molecular Dynamics / Discrete Element Method in Rust**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE)


>Many features exist in this codebase that have not been tested, examples include all testing done within this codebase.

## What is MDDEM?

MDDEM is a particle simulation engine written in Rust, supporting both **Discrete Element Method (DEM)** for granular materials and potentially **Molecular Dynamics (MD)** (still learning this side).

The framework is built around **composability**. A dependency-injection scheduler inspired by [Bevy](https://github.com/bevyengine/bevy) and a plugin system let you assemble simulations from independent, reusable pieces. Physics models, integrators, neighbor lists, and output formats are all plugins. Systems declare their resource dependencies as function arguments; the scheduler injects them automatically and resolves execution order.

Configuration follows a **two-tier** approach:

- **Tier 1 — Declarative TOML** for standard simulations: named fields, typed values, validated at startup.
- **Tier 2 — Rust API** for complex simulations: `main.rs` composes plugins, and custom systems are real functions with full type safety and IDE support.

Both tiers can be mixed freely. The [hopper](examples/hopper/) example uses TOML config alongside custom Rust systems for runtime wall control.

## Motivation

MDDEM began as a Rust reimplementation of LAMMPS communication patterns, motivated by a desire to **explore whether a scheduler with dependency injection could work for particle simulations**. 


## Design Notes

MDDEM uses TOML for input configuration. I don't like this, Still exploring alternative ideas to input configurations. 


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

MDDEM is not yet published on crates.io. Add it as a git dependency:

**`Cargo.toml`**
```toml
[dependencies]
mddem = { git = "https://github.com/SueHeir/MDDEM" }
```

To build without MPI (single-process mode):

```toml
[dependencies]
mddem = { git = "https://github.com/SueHeir/MDDEM", default-features = false }
```

**`config.toml`**
```toml
[domain]
x_high = 0.025
y_high = 0.025
z_high = 0.025
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"

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
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"

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

# Print the compiled schedule for your specific simulation (Graphviz DOT)
cargo run --release -- config.toml --schedule
```

`CorePlugins` bundles config loading, communication, domain decomposition, neighbor lists, groups, and output. `GranularDefaultPlugins` adds DEM contact physics, rotational dynamics, particle insertion, and Velocity Verlet integration. `LJDefaultPlugins` adds FCC lattice, LJ forces, Nosé-Hoover thermostat with fused Verlet integration, and measurements. Individual plugins can be added separately for custom configurations. Please read sim_schduler and sim_app readme for more information.

## Architecture

| Crate | Description |
|---|---|
| [`mddem`](crates/mddem/) | Umbrella crate: `CorePlugins`, `LJDefaultPlugins`, `GranularDefaultPlugins`, prelude |
| [`sim_scheduler`](crates/sim_scheduler/) | Dependency-injection scheduler with resources, schedule sets, ordering, and run conditions |
| [`sim_app`](crates/sim_app/) | App, SubApp, Plugin, PluginGroup, StatesPlugin |
| [`mddem_core`](crates/mddem_core/) | Config, domain decomposition, communication, atom data, regions, groups |
| [`mddem_neighbor`](crates/mddem_neighbor/) | Neighbor lists: brute force, sweep-and-prune, bin-based |
| [`mddem_verlet`](crates/mddem_verlet/) | Velocity Verlet translational integration |
| [`mddem_print`](crates/mddem_print/) | Thermo, dump files (CSV/binary), VTP visualization, restart files |
| [`mddem_derive`](crates/mddem_derive/) | `#[derive(AtomData)]` and `#[derive(StageEnum)]` proc macros |
| [`mddem_fixes`](crates/mddem_fixes/) | AddForce, SetForce, Freeze, MoveLinear, Viscous, NveLimit; Gravity |
| [`mddem_fire`](crates/mddem_fire/) | FIRE energy minimization |
| [`mddem_deform`](crates/mddem_deform/) | Box deformation (engineering strain rate, velocity, target size) |
| [`mddem_velocity_distribution`](crates/mddem_velocity_distribution/) | Velocity distribution analysis and Maxwell-Boltzmann comparison |
| [`dem_atom`](crates/dem_atom/) | Per-atom DEM data, material table, particle insertion, size distributions |
| [`dem_granular`](crates/dem_granular/) | Hertz/Hooke contact, Mindlin tangential, rolling/twisting friction, JKR/DMT/SJKR adhesion, rotational dynamics |
| [`dem_wall`](crates/dem_wall/) | Plane, cylinder, sphere, cone, and region-surface walls with motion control |
| [`dem_thermal`](crates/dem_thermal/) | Contact-based heat conduction (particle-particle and particle-wall) |
| [`dem_bond`](crates/dem_bond/) | Bonded particle model: normal/tangential/bending forces, auto-bonding, breakage |
| [`dem_clump`](crates/dem_clump/) | Multisphere/clump rigid body composites |
| [`dem_contact_analysis`](crates/dem_contact_analysis/) | Coordination number, fabric tensor, rattler detection, per-contact CSV output |
| [`dem_measure_plane`](crates/dem_measure_plane/) | Measurement planes for mass flow rate and crossing statistics |
| [`md_lj`](crates/md_lj/) | LJ 12-6 pair force with virial, tail corrections, and multi-type support |
| [`md_thermostat`](crates/md_thermostat/) | Nosé-Hoover NVT and Langevin thermostats |
| [`md_lattice`](crates/md_lattice/) | FCC lattice initialization with Maxwell-Boltzmann velocities |
| [`md_measure`](crates/md_measure/) | RDF, MSD, virial pressure |
| [`md_msd`](crates/md_msd/) | Per-type mean squared displacement with diffusion coefficients |
| [`md_type_rdf`](crates/md_type_rdf/) | Type-filtered radial distribution function |
| [`mddem_test_utils`](crates/mddem_test_utils/) | Shared test utilities |




## Performance (at one point)

Single-core LJ fluid benchmark comparing MDDEM to LAMMPS (29 Sep 2024 release). Identical physics: LJ 12-6 with cutoff 2.5σ, FCC lattice at ρ*=0.8442, Nosé-Hoover NVT at T*=1.44, full 6-component virial stress tensor, neighbor rebuild every 20 steps, 200 timesteps. Compiled with `--release` on Apple M1 Pro.

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | Ratio |
|--------:|---------------:|----------------:|------:|
|     108 |         23,305 |          31,145 | 1.34× |
|   1,000 |          2,562 |           2,943 | 1.15× |
|  10,000 |            277 |             293 | 1.06× |
|  32,000 |           88.1 |            92.7 | 1.05× |
| 100,920 |           28.4 |            29.1 | 1.02× |
| 202,612 |           14.1 |            14.5 | 1.03× |

At scale (10k+ atoms), MDDEM is within 2–5% of LAMMPS with consistent O(N) scaling. 

**MPI benchmark** (4 processes, 2×2×1 decomposition):

| Atoms   | MDDEM (step/s) | LAMMPS (step/s) | MDDEM Speedup | Ratio |
|--------:|---------------:|----------------:|--------------:|------:|
|     108 |         28,893 |          29,565 |         1.24× | 1.02× |
|   1,000 |          5,725 |           9,498 |         2.23× | 1.66× |
|  10,000 |            857 |           1,088 |         3.09× | 1.27× |
|  32,000 |            305 |             335 |         3.46× | 1.10× |
| 100,920 |           98.5 |             108 |         3.47× | 1.10× |

MDDEM achieves 2.2–3.5× (4 mpi processes) speedup over single-core at scale, with the LAMMPS ratio narrowing to 1.10× at 32k+ atoms.

## License

MIT OR Apache-2.0
