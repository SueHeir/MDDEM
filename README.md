# MDDEM: Molecular Dynamics - Discrete Element Method

> **Disclaimer:** This is a toy project created to explore coding patterns in Rust. Much of the code was written with the assistance of Claude (Anthropic's AI). Vibe-coded pull requests are accepted, provided the contributor is qualified in the relevant domain and has personally reviewed the code being submitted.

MDDEM (pronounced like "Madem" without the 'a') is a Molecular Dynamics / Discrete Element Method codebase written in Rust with optional MPI parallelization.
It uses [rsmpi](https://github.com/rsmpi/rsmpi) as a Rust wrapper around MPI (feature-gated behind `mpi_backend`) and [nalgebra](https://github.com/dimforge/nalgebra) as a math library. MPI is optional — MDDEM builds and runs on a single process without it.

LAMMPS/LIGGGHTS-style MPI communication (exchange, borders, reverse force) is fully implemented for periodic boxes of spheres with Hertz normal contact, Mindlin tangential friction, and rotational dynamics.

## Building

By default, MDDEM builds with MPI support via the `mpi_backend` feature flag. On systems where MPI/libffi-sys is unavailable (e.g. aarch64 macOS without Homebrew libffi), you can build without MPI:

```bash
# With MPI (default)
cargo build --release

# Without MPI (single-process only)
cargo build --release --no-default-features
```

When built without `mpi_backend`, a `SingleProcessComm` backend is used automatically — no code changes needed. This mode supports all physics and periodic boundaries on a single process.

## Running

```bash
mpiexec -n 4 ./target/release/MDDEM ./examples/granular_basic/config.toml
```

The path to a TOML configuration file is passed as the first argument.

Pass `--schedule` to print the compiled schedule and write a Graphviz DOT file:

```bash
mpiexec -n 1 ./target/release/MDDEM ./examples/granular_basic/config.toml --schedule
```

## Design Philosophy

MDDEM avoids the pitfalls of LAMMPS-style scripting through a deliberate two-tier design.

LAMMPS input scripts occupy the worst possible design point: complex enough to require programming (multi-stage runs, conditional logic, variable expressions), but implemented as a weak scripting language with positional numeric arguments, order-dependent commands, arcane variable syntax (`${x}` vs `v_x` vs `$(v_x)`), and `fix`/`unfix` lifecycle management. The result is 500-line scripts that nobody can debug.

MDDEM replaces this with two clean tiers:

### Tier 1: Declarative TOML Config

For standard simulations, everything is specified in a TOML configuration file with named fields, typed values, and compile-time validation via `serde`. No positional arguments, no order dependence, no string-based ID management. Multi-stage simulations use `[[run]]` arrays:

```toml
[[run]]
name = "settling"
steps = 10000
thermo = 100

[[run]]
name = "production"
steps = 100000
thermo = 1000
dump_interval = 500
```

Each plugin owns its config section. Bad values produce clear serde error messages at startup, not silent wrong physics at step 50,000.

### Tier 2: Rust API

When TOML isn't enough, users write Rust directly. The simulation is a library — `main.rs` composes plugins, and custom systems are real functions with full type safety, IDE autocomplete, and the entire Rust ecosystem:

```rust
fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
       .add_plugins(GranularDefaultPlugins);

    // Custom system with DI, type checking, and real programming constructs
    app.add_update_system(
        my_custom_force.run_if(in_state(Phase::Production)),
        ScheduleSet::PostForce,
    );
    app.start();
}

fn my_custom_force(mut atoms: ResMut<Atom>, domain: Res<Domain>) {
    // Real code in a real language
}
```

There is no middle-ground scripting language. Simple things stay simple (TOML). Complex things use a real language (Rust).

## Testing

Tests run without MPI, using the single-process backend:

```bash
# Run all tests
cargo test --workspace --no-default-features

# Run tests for a specific crate
cargo test -p mddem_core --no-default-features
cargo test -p mddem_verlet
cargo test -p mddem_neighbor
cargo test -p dem_granular
```

Tests cover:
- `SingleProcessComm` backend (rank, size, reductions)
- `CartesianDecomposition` (single-proc and multi-proc domain splitting)
- Velocity Verlet integration (initial and final half-steps)
- Neighbor lists (brute force, sweep-and-prune, and bin-based)
- Hertz normal contact force (repulsive for overlap, zero for gap)
- Mindlin tangential friction (spring history, Coulomb cap)
- Rotational dynamics (angular acceleration, quaternion updates)
- Material mixing (single-material, multi-material symmetry)

## Examples

| Example | Description |
|---------|-------------|
| [granular_basic](examples/granular_basic/) | Basic 500-particle granular gas in a periodic box |
| [benchmark](examples/benchmark/) | Haff's cooling law validation with LAMMPS comparison |
| [toml_single](examples/toml_single/) | Single short run for quick testing |


## Input

Parameters are organized into TOML sections, each owned by its plugin:

```toml
[comm]
processors_x = 2
processors_y = 2
processors_z = 1

[domain]
x_low = 0.0
x_high = 0.025
y_low = 0.0
y_high = 0.025
z_low = 0.0
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

# Single-stage run (backwards compatible)
[run]
thermo = 100
steps = 10000

# Or multi-stage: use [[run]] for sequential stages
# [[run]]
# name = "settling"
# steps = 10000
# thermo = 100
#
# [[run]]
# name = "production"
# steps = 100000
# thermo = 1000
# dump_interval = 500

[dump]
interval = 1000
format = "text"

[restart]
interval = 10000
format = "bincode"
read = false

[vtp]
interval = 2000
```

Materials are defined as named types under `[[dem.materials]]`. Particles reference a material by name in `[[particles.insert]]`. Multiple material types and insert blocks are supported — mixed-material contacts use geometric-mean mixing for restitution and friction (LAMMPS convention).

### Parameters

| Section | Parameter | Description |
|---|---|---|
| `[comm]` | `processors_x/y/z` | MPI domain decomposition (default 1) |
| `[domain]` | `x/y/z_low/high` | Simulation box bounds (m) |
| `[domain]` | `periodic_x/y/z` | Boundary condition per axis |
| `[neighbor]` | `skin_fraction` | Neighbor list cutoff multiplier |
| `[neighbor]` | `bin_size` | Minimum bin size for bin-based neighbor list |
| `[[dem.materials]]` | `name` | Material type name |
| `[[dem.materials]]` | `youngs_mod` | Young's modulus (Pa) |
| `[[dem.materials]]` | `poisson_ratio` | Poisson ratio |
| `[[dem.materials]]` | `restitution` | Normal restitution coefficient (0-1) |
| `[[dem.materials]]` | `friction` | Coulomb friction coefficient (default 0.4) |
| `[[particles.insert]]` | `material` | Name of material type to use |
| `[[particles.insert]]` | `count` | Number of spheres to insert randomly |
| `[[particles.insert]]` | `radius` | Particle radius (m) |
| `[[particles.insert]]` | `density` | Particle density (kg/m^3) |
| `[[particles.insert]]` | `velocity` | Initial RMS velocity (m/s, Gaussian per component) |
| `[run]` or `[[run]]` | `name` | Stage name for logging (optional, multi-stage only) |
| `[run]` or `[[run]]` | `steps` | Number of timesteps |
| `[run]` or `[[run]]` | `thermo` | Print thermodynamic output every N steps |
| `[[run]]` | `dump_interval` | Override dump interval for this stage (optional) |
| `[[run]]` | `restart_interval` | Override restart interval for this stage (optional) |
| `[[run]]` | `vtp_interval` | Override VTP interval for this stage (optional) |
| `[dump]` | `interval` | Dump atom data every N steps (0 = disabled) |
| `[dump]` | `format` | Dump format: `"text"` (CSV) or `"binary"` |
| `[restart]` | `interval` | Write restart files every N steps (0 = disabled) |
| `[restart]` | `format` | Restart format: `"bincode"` or `"json"` |
| `[restart]` | `read` | Read restart file on startup (default false) |
| `[vtp]` | `interval` | Write VTP visualization files every N steps (0 = disabled) |

## Physics

- **Normal contact**: Hertz elastic contact with viscoelastic damping (LAMMPS `hertz/material` equivalent)
- **Tangential contact**: Mindlin spring-history model with Coulomb friction cap and viscous tangential damping
- **Damping**: beta derived from restitution coefficient e via beta = -ln(e) / sqrt(pi^2 + ln^2(e))
- **Integration**: Velocity Verlet for both translational and rotational degrees of freedom
- **Rotational dynamics**: Quaternion-based orientation tracking, angular velocity integration (I = 2/5 mr^2 for solid spheres)
- **Timestep**: Automatically computed as 5% of the Rayleigh wave period
- **MPI**: Optional 3D domain decomposition with ghost atom forwarding to diagonal neighbors (corner-complete). Falls back to single-process mode when built without the `mpi_backend` feature.

Not yet implemented: gravity.

## Code Layout

MDDEM is built around a dependency-injection scheduler inspired by [Bevy](https://github.com/bevyengine/bevy). All simulation state lives in typed resources. Systems declare the resources they need as function arguments and the scheduler injects them automatically.

| Crate | Description |
|---|---|
| [`mddem_scheduler`](crates/mddem_scheduler/) | DI scheduler, resources, schedule sets, ordering, run conditions, states |
| [`mddem_app`](crates/mddem_app/) | App, SubApp, Plugin, PluginGroup, StatesPlugin |
| [`mddem_core`](crates/mddem_core/) | Core simulation: TOML config loading, domain decomposition, communication (MPI or single-process), atom data structures, run/cycle management |
| [`mddem_neighbor`](crates/mddem_neighbor/) | Neighbor lists: brute force, sweep-and-prune, bin-based |
| [`mddem_verlet`](crates/mddem_verlet/) | Velocity Verlet translational integration (initial + final half-steps) |
| [`mddem_print`](crates/mddem_print/) | Output: thermo, VTP visualization, granular temperature, dump files, restart files |
| [`dem_atom`](crates/dem_atom/) | Per-atom DEM data (radius, density) with pack/unpack, `MaterialTable` for per-material and per-pair mixing, material config |
| [`dem_atom_insert`](crates/dem_atom_insert/) | DEM particle insertion: random placement with overlap checking, populates both `Atom` and `DemAtom` fields |
| [`dem_granular`](crates/dem_granular/) | Granular physics: Hertz normal contact, Mindlin tangential friction, rotational dynamics, `GranularDefaultPlugins` |
| [`mddem`](crates/mddem/) | Umbrella crate: `CorePlugins`, prelude re-exports |

`main.rs` composes the simulation by adding plugin groups to an `App`:

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
```

`CorePlugins` bundles TOML config loading, communication, domain decomposition, neighbor lists, run/cycle management, Velocity Verlet integration, and output. MPI finalization is handled automatically via cleanup callbacks. Individual plugins can be added separately for custom configurations.

### Using App programmatically

You can construct an `App` directly with a programmatic `Config` table, bypassing TOML file parsing:

```rust
use mddem::prelude::*;

fn main() {
    let mut table = toml::Table::new();

    let mut comm = toml::Table::new();
    comm.insert("processors_x".into(), 1.into());
    comm.insert("processors_y".into(), 1.into());
    comm.insert("processors_z".into(), 1.into());
    table.insert("comm".into(), comm.into());

    let mut domain = toml::Table::new();
    domain.insert("x_high".into(), toml::Value::Float(0.025));
    domain.insert("y_high".into(), toml::Value::Float(0.025));
    domain.insert("z_high".into(), toml::Value::Float(0.025));
    domain.insert("periodic_x".into(), true.into());
    domain.insert("periodic_y".into(), true.into());
    domain.insert("periodic_z".into(), true.into());
    table.insert("domain".into(), domain.into());

    let mut mat = toml::Table::new();
    mat.insert("name".into(), "glass".into());
    mat.insert("youngs_mod".into(), toml::Value::Float(8.7e9));
    mat.insert("poisson_ratio".into(), toml::Value::Float(0.3));
    mat.insert("restitution".into(), toml::Value::Float(0.95));
    mat.insert("friction".into(), toml::Value::Float(0.4));
    let mut dem = toml::Table::new();
    dem.insert("materials".into(), toml::Value::Array(vec![toml::Value::Table(mat)]));
    table.insert("dem".into(), dem.into());

    let mut insert = toml::Table::new();
    insert.insert("material".into(), "glass".into());
    insert.insert("count".into(), 100.into());
    insert.insert("radius".into(), toml::Value::Float(0.001));
    insert.insert("density".into(), toml::Value::Float(2500.0));
    insert.insert("velocity".into(), toml::Value::Float(0.5));
    let mut particles = toml::Table::new();
    particles.insert("insert".into(), toml::Value::Array(vec![toml::Value::Table(insert)]));
    table.insert("particles".into(), particles.into());

    let mut run = toml::Table::new();
    run.insert("steps".into(), 1000.into());
    run.insert("thermo".into(), 100.into());
    table.insert("run".into(), run.into());

    let mut app = App::new();
    app.add_resource(Input { filename: String::new(), output_dir: None });
    app.add_resource(Config { table });
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
```

When `Config` is already present before `CorePlugins` builds, `InputPlugin` skips CLI parsing entirely. MPI finalization is handled automatically. Useful for embedding MDDEM as a library or running from a custom driver.

Per-atom DEM data (radius, density) is stored in `DemAtom`, a typed extension registered with `AtomDataRegistry`. The registry packs and unpacks these fields automatically during MPI communication alongside the base `Atom` fields. Per-material properties (Young's modulus, Poisson ratio, restitution, friction) live in `MaterialTable`, populated at plugin build time from `[[dem.materials]]` config.

## Future Goals

- **Library crate**: Publish to crates.io so simulations can be composed by importing plugins
- **Gravity**: Body force support
- **GPU readiness**: Flat neighbor list arrays; grid-based neighbor detection; f32 GPU kernels
- **LEBC**: Lees-Edwards boundary conditions for shear flow
- **Polydispersity**: Per-insert-block radii supported; continuous size distributions planned

## Completed

- **Multi-stage runs**: `[[run]]` TOML arrays for sequential simulation stages with per-stage thermo/dump/restart/vtp intervals
- **SoA refactor**: Flat `Vec<f64>` per field for cache efficiency (~9x speedup)
