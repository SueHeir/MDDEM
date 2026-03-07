# MDDEM: Molecular Dynamics - Discrete Element Method

> **Disclaimer:** This is a toy project created to explore coding patterns in Rust. Much of the code was written with the assistance of Claude (Anthropic's AI). Vibe-coded pull requests are accepted, provided the contributor is qualified in the relevant domain and has personally reviewed the code being submitted.

MDDEM (pronounced like "Madem" without the 'a') is a Molecular Dynamics / Discrete Element Method codebase written in Rust with optional MPI parallelization.
It uses [rsmpi](https://github.com/rsmpi/rsmpi) as a Rust wrapper around MPI (feature-gated behind `mpi_backend`) and [nalgebra](https://github.com/dimforge/nalgebra) as a math library. MPI is optional — MDDEM builds and runs on a single process without it.

LAMMPS/LIGGGHTS-style MPI communication (exchange, borders, reverse force) is fully implemented for periodic boxes of spheres with Hertz normal contact, Mindlin tangential friction, and rotational dynamics.

## Design Philosophy

MDDEM is built around **composability**. A dependency-injection scheduler (inspired by [Bevy](https://github.com/bevyengine/bevy)) and plugin system let you assemble simulations from independent, reusable pieces. Physics models, integrators, neighbor list algorithms, and output formats are all plugins — swap one out, add a new one, or combine them without touching the rest of the codebase. Systems declare what resources they need as function arguments; the scheduler injects them automatically and handles execution order.

Configuration follows a **two-tier** approach:

**Tier 1 — Declarative TOML config** for standard simulations. Named fields, typed values, validated at startup via `serde`. Each plugin owns its config section, and multi-stage runs are expressed as `[[run]]` arrays.

**Tier 2 — Rust API** for complex simulations. The simulation is a library — `main.rs` composes plugins, and custom systems are real functions with full type safety, IDE autocomplete, and the entire Rust ecosystem. For example, the [hopper](examples/hopper/) example adds a custom system that monitors kinetic energy and removes a wall when particles settle, using `run_if(in_state(...))` for phase-dependent logic.

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

Examples are compiled executables. For single-process runs:

```bash
cargo run --example granular_basic -- examples/granular_basic/config.toml
```

For MPI runs, build first then launch with `mpiexec`:

```bash
cargo build-examples                # alias for: cargo build --release --examples
mpiexec -n 4 ./target/release/examples/granular_basic examples/granular_basic/config.toml
```

Pass `--schedule` to print the compiled schedule and write a Graphviz DOT file:

```bash
cargo run --example granular_basic -- examples/granular_basic/config.toml --schedule
```

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

[run]
thermo = 100
steps = 10000

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

### Multi-stage runs

For sequential simulation stages with different settings, use `[[run]]` (TOML array) instead of `[run]`:

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

Each stage can override `dump_interval`, `restart_interval`, and `vtp_interval` for that stage only; unset values fall back to the global `[dump]`/`[restart]`/`[vtp]` config. Single-stage `[run]` is still supported.

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
| `[[particles.insert]]` | `velocity_x/y/z` | Directional initial velocity components (m/s, additive with random) |
| `[[particles.insert]]` | `region_x/y/z_low/high` | Insertion sub-region bounds (optional, defaults to domain) |
| `[gravity]` | `gx`, `gy`, `gz` | Gravitational acceleration components (m/s^2, default 0, 0, -9.81) |
| `[[wall]]` | `point_x/y/z` | A point on the wall plane (m) |
| `[[wall]]` | `normal_x/y/z` | Inward normal vector (normalized internally) |
| `[[wall]]` | `bound_x/y/z_low/high` | Optional bounding box to clip wall to a finite region |
| `[[wall]]` | `material` | Material name for contact properties |
| `[[wall]]` | `name` | Optional wall name (for runtime toggling) |
| `[run]` or `[[run]]` | `name` | Stage name for logging (optional) |
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
- **Gravity**: Configurable body force (`GravityPlugin`)
- **Walls**: General plane wall contact with Hertz repulsion (`WallPlugin`), supports axis-aligned and angled walls, toggleable at runtime for staged simulations
- **MPI**: Optional 3D domain decomposition with ghost atom forwarding to diagonal neighbors (corner-complete). Falls back to single-process mode when built without the `mpi_backend` feature.

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
| [`dem_gravity`](crates/dem_gravity/) | Configurable gravity body force (`GravityPlugin`) |
| [`dem_wall`](crates/dem_wall/) | General plane wall contact forces with Hertz repulsion (`WallPlugin`), runtime-toggleable walls |
| [`mddem`](crates/mddem/) | Umbrella crate: `CorePlugins`, prelude re-exports |

A simulation is composed by adding plugin groups to an `App`:

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
```

`CorePlugins` bundles TOML config loading, communication, domain decomposition, neighbor lists, run/cycle management, Velocity Verlet integration, and output. MPI finalization is handled automatically via cleanup callbacks. Individual plugins can be added separately for custom configurations.

### Programmatic config

You can construct an `App` directly with a programmatic `Config` table, bypassing TOML file parsing (see the [toml_single](examples/toml_single/) example):

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

Examples are compiled executables, each with a `main.rs` and supporting files:

| Example | Description | Run |
|---------|-------------|-----|
| [granular_basic](examples/granular_basic/) | 500-particle granular gas in a periodic box | `cargo run --example granular_basic -- examples/granular_basic/config.toml` |
| [benchmark](examples/benchmark/) | Haff's cooling law validation with LAMMPS comparison | `cargo run --example benchmark -- examples/benchmark/config.toml` |
| [toml_single](examples/toml_single/) | Programmatic config — no TOML file needed | `cargo run --example toml_single` |
| [hopper](examples/hopper/) | 2D slot hopper with angled funnel walls, gravity, and simulation states | `cargo run --example hopper -- examples/hopper/config.toml` |

The `granular_basic`, `benchmark`, and `hopper` examples use TOML config files (Tier 1). The `toml_single` example builds its config entirely in Rust code (Tier 2), demonstrating programmatic setup for parameter sweeps or embedding MDDEM as a library. The `hopper` example combines TOML config with a custom Rust setup system (Tier 2) for runtime wall control.

## Future Goals

Each goal is a physics simulation that doubles as a benchmark. Implementing each one requires adding specific new features, and the analytical/experimental comparisons validate correctness. Together they test the composability and flexibility of the plugin architecture.

### 1. Granular Column Collapse (Dam Break)

A tall column of particles is released and collapses under gravity, spreading until it comes to rest. The granular analogue of a dam break.

**New features needed:**
- Lattice-packed particle insertion (FCC/cubic packing within a region, extending `dem_atom_insert`)
- Open or absorbing domain boundaries for particles that leave the simulation box

**Why it matters:** Tests large-displacement transient dynamics where particles move many body-diameters. Validates multi-stage simulation (settle with confining walls, then remove walls to trigger collapse). Exercises MPI load balancing as particles rapidly redistribute across processors. Stresses the neighbor list under fast-changing spatial configurations.

**Validation:** Normalized runout scales as (L_final - L_0)/L_0 ~ a^0.9 and deposit height as H_final/H_0 ~ a^(-0.6) where a = H_0/L_0 is the initial aspect ratio. These power laws are remarkably robust and material-independent ([Lube et al., J. Fluid Mech. 508, 2004](https://doi.org/10.1017/S0022112004009036)).

### 2. Angle of Repose (Funnel Discharge)

Particles pour continuously from a funnel onto a flat surface, forming a conical pile. The pile angle is a fundamental bulk property determined by friction and rolling resistance.

**New features needed:**
- Rolling resistance torque model (elastic-plastic spring, Type C per [Ai et al., Granular Matter 13, 2011](https://doi.org/10.1007/s10035-010-0229-0)) — a new torque plugin that composes with existing Mindlin tangential forces and rotational Verlet integration
- Runtime particle insertion — a system that creates particles at specified intervals during the run phase, not just at setup

**Why it matters:** This is the single most common DEM calibration benchmark in the literature, used by a [16-group international round-robin study](https://www.sciencedirect.com/science/article/pii/S003808062300001X). Rolling resistance is a new physics plugin that must integrate cleanly with existing contact forces, directly testing plugin composability. Without rolling resistance, smooth spheres produce unrealistically low angles (~20 deg vs 25-35 deg for real materials).

**Validation:** For glass beads with mu=0.5, angle of repose ~25 +/- 2 deg. The angle should be independent of pile size (scale invariance). Analytical expressions from [Albert et al., PNAS 118, 2021](https://www.pnas.org/doi/10.1073/pnas.2107965118).

### 3. Inclined Chute Flow (Bagnold Profile)

Steady-state granular flow down a rough inclined surface. Particles reach equilibrium where gravity balances friction, producing a characteristic velocity profile through the flow depth.

**New features needed:**
- Frozen/fixed particles — a particle flag that participates in neighbor finding and contact forces but skips integration, used to create a rough wall base
- Spatial binning for velocity profiles — analysis output that bins particle velocities by depth to extract the flow profile

**Why it matters:** Tests that Hertz/Mindlin contact forces produce correct *bulk* rheology, not just pairwise correctness. Validates periodic boundaries under sustained non-equilibrium shear flow. Tests steady-state energy balance (gravitational input = frictional dissipation). The velocity profile measurement exercises analysis/output infrastructure.

**Validation:** Bagnold velocity profile v(z) ~ [1 - (1-z/H)^(3/2)] ([Silbert et al., Phys. Rev. E 64, 2001](https://doi.org/10.1103/PhysRevE.64.051302)). The mu(I) rheology curve can be extracted and compared to the empirical law mu(I) = mu_s + (mu_2 - mu_s)/(I_0/I + 1) ([Jop et al., Nature 441, 2006](https://doi.org/10.1038/nature04801)).

### 4. Simple Shear with Lees-Edwards Boundaries (mu(I) Rheology)

Uniform simple shear at controlled shear rate and pressure. The gold standard for measuring bulk granular constitutive behavior, eliminating wall effects entirely.

**New features needed:**
- Lees-Edwards boundary conditions (LEBC) — deforming periodic boundaries where images are displaced at a controlled velocity. Requires changes to domain wrapping, ghost atom velocity offsets, neighbor list image displacement, and force relative-velocity calculations
- Simple barostat — Berendsen-style rescaling of the box dimension perpendicular to shear to maintain target pressure

**Why it matters:** The most architecturally demanding benchmark. LEBC requires fundamental changes to the periodic boundary infrastructure — boundaries that move and deform over time. Tests that MPI domain decomposition handles a continuously deforming domain. The mu(I) curve integrates all contact mechanics into a single measurable relationship, making it the most fundamental constitutive test of the force model.

**Validation:** mu(I) = mu_s + (mu_2 - mu_s)/(I_0/I + 1) with mu_s ~ 0.38, mu_2 ~ 0.64, I_0 ~ 0.28 for frictional spheres ([da Cruz et al., Phys. Rev. E 72, 2005](https://doi.org/10.1103/PhysRevE.72.021309)). Volume fraction phi(I) = phi_max - (phi_max - phi_min)*I with phi_max ~ 0.64.

### 5. Vibrated Granular Bed (Convection and Brazil Nut Effect)

A container of particles is vibrated sinusoidally from below. Above a critical acceleration, particles detach from the base and complex phenomena emerge: convection cells form and large intruder particles rise to the surface (the Brazil Nut Effect).

**New features needed:**
- Moving/oscillating walls — walls whose position varies as x(t) = A*sin(omega*t), with wall velocity feeding into the contact force calculation
- Per-step wall update system — updates wall positions and velocities each timestep
- Intruder particle tracking — per-particle position-vs-time output for tagged particles
- Multi-size particle insertion — polydisperse mixtures with large intruder particles among smaller ones

**Why it matters:** Tests the full spectrum of granular dynamics in a single simulation: dense packing, fluidization, free flight, and re-compaction. Validates time-dependent boundary conditions (prerequisite for rotating drums, shakers, and industrial applications). Tests multi-material/multi-size composability. The convection cell pattern validates correct collective behavior, not just pairwise accuracy.

**Validation:** Critical acceleration for fluidization: Gamma_c = A*omega^2/g = 1. Convection onset at Gamma ~ 2-3 ([Wildman et al., Phys. Rev. Lett. 86, 2001](https://doi.org/10.1103/PhysRevLett.86.3304)). Intruder rise velocity scales linearly with vibration velocity amplitude.

### Molecular Dynamics

The DEM benchmarks above validate granular contact mechanics. The following MD benchmarks progressively build general-purpose molecular simulation capability, each layering new features on top of the previous ones.

### 6. Lennard-Jones Fluid (Argon)

A box of Lennard-Jones atoms at controlled temperature and density. The "hello world" of molecular dynamics.

**New features needed:**
- `md_lj` — Lennard-Jones 12-6 pair potential with cutoff and energy shift. The first *continuous* pair potential (attractive + repulsive, vs. DEM's purely repulsive overlap-based forces)
- `md_thermostat` — Nose-Hoover thermostat (NVT ensemble). Extends the Velocity Verlet integrator with thermostat coupling via auxiliary chain variables
- `md_measure` — Radial distribution function g(r) computation and virial pressure calculation
- Reduced (LJ) units system alongside existing SI units

**Why it matters:** Proves the plugin architecture supports continuous pair potentials alongside DEM contact forces without modifying the scheduler or core. Tests that neighbor lists, ghost communication, and periodic boundaries work correctly for a fundamentally different force model. The thermostat plugin tests that the integration pipeline can be modified by plugins (inserting half-step velocity scaling).

**Validation:** Johnson-Zollweg-Gubbins equation of state — pressure vs. density/temperature to within 1-2%. g(r) at T\*=0.85, rho\*=0.85 (liquid argon near triple point): first peak at r/sigma ~ 1.09 ([Verlet, Phys. Rev. 159, 1967](https://doi.org/10.1103/PhysRev.159.98)). Diffusion coefficient D\* ~ 0.033 at T\*=1.0 via mean-square displacement ([Rahman, Phys. Rev. 136, 1964](https://doi.org/10.1103/PhysRev.136.A405)).

### 7. Lennard-Jones Crystal Melting

An FCC crystal is heated through the melting transition, testing lattice initialization, phase detection, and the NPT ensemble.

**New features needed:**
- `md_lattice` — Lattice initializer generating FCC, BCC, HCP, or SC crystal structures with specified lattice constant
- `md_barostat` — Nose-Hoover barostat (NPT ensemble). Couples to simulation box dimensions, rescales particle positions when the box resizes, and triggers domain/neighbor list updates
- Variable box size support in core — `Domain` must support dynamic resizing, `Comm` must handle changing ghost cutoffs and processor boundaries

**Why it matters:** The barostat is the first feature that modifies the simulation box at runtime, stress-testing domain decomposition, ghost communication, and neighbor list rebuilds as processor boundaries shift every step. Lattice initialization tests that the plugin system cleanly swaps setup strategies (lattice vs. random insertion). Phase coexistence (solid-liquid interface in one box) tests handling regions of very different local structure.

**Validation:** LJ melting point at zero pressure: T_m\* = 0.694 +/- 0.006 via two-phase coexistence method ([Mastny and de Pablo, J. Chem. Phys. 127, 2007](https://doi.org/10.1063/1.2753149)). Latent heat delta_H\* ~ 1.05 ([Hansen and Verlet, Phys. Rev. 184, 1969](https://doi.org/10.1103/PhysRev.184.151)). Lindemann criterion: melting at ~10% RMS displacement of nearest-neighbor distance.

### 8. Kremer-Grest Polymer Melt

A melt of coarse-grained bead-spring polymer chains. Each chain is a sequence of LJ beads connected by FENE springs — the standard model for polymer dynamics and entanglement.

**New features needed:**
- `md_bond` — FENE bond potential: U = -0.5\*k\*R0^2\*ln(1 - (r/R0)^2). Iterates over a bond list and computes forces along bond vectors
- Bond migration in MPI — when atoms migrate across processor boundaries, their bond connectivity must migrate with them
- Chain initialization — random-walk initial configurations with soft push-off equilibration
- Molecule topology infrastructure — molecule IDs, bond lists, neighbor list exclusions for bonded pairs

**Why it matters:** Bonded interactions are the first *topological* forces (connectivity-dependent, not proximity-dependent), testing a fundamentally different force computation pattern. Bond migration across MPI boundaries is a hard test of communication infrastructure — a chain spanning 3 processors must compute forces correctly. The FENE singularity at r = R0 tests robustness. Long equilibration runs (10^6-10^7 steps) test numerical stability and performance.

**Validation:** Rouse dynamics for short chains (N < N_e ~ 85): tau_R ~ N^2, monomer MSD ~ t^0.5. Reptation for long chains (N > N_e): MSD ~ t^0.25, D ~ N^-2.3. Entanglement length N_e ~ 85 beads ([Kremer and Grest, J. Chem. Phys. 92, 1990](https://doi.org/10.1063/1.458541)). End-to-end distance R_e^2 = C_inf\*N\*b^2 with C_inf ~ 1.7.

### 9. SPC/E Water

A box of SPC/E water molecules at ambient conditions. The standard benchmark for electrostatic solvers and rigid-body constraints.

**New features needed:**
- `md_coulomb` — Particle-Particle Particle-Mesh (PPPM) or Ewald summation for long-range electrostatics. Requires FFT (via `rustfft`), charge assignment to a mesh, reciprocal-space force calculation, and MPI mesh decomposition
- `md_rigid` — SHAKE/RATTLE constraint algorithm for fixed bond lengths and angles
- Partial charges on atoms — add `charge: Vec<f64>` to `Atom`
- Mixed LJ + Coulomb pair interactions in a single force loop
- Topology exclusions — neighbor list must exclude bonded pairs from non-bonded interactions

**Why it matters:** PPPM/Ewald is a fundamentally different computation pattern: a global FFT operation rather than local pair interactions. Tests whether the plugin architecture accommodates global solvers that need the full domain. SHAKE constraints modify positions *after* the integrator, testing scheduler flexibility. Multi-site molecules with different LJ and charge parameters test the multi-type infrastructure.

**Validation:** SPC/E density at 300 K, 1 atm: 998 +/- 5 kg/m^3 (exp: 997). g_OO(r) first peak at 2.75 A with height ~3.0 ([Berendsen et al., J. Phys. Chem. 91, 1987](https://doi.org/10.1021/j100308a038)). Self-diffusion D = 2.4e-5 cm^2/s (exp: 2.3e-5). Dielectric constant epsilon ~ 71 (exp: 78.4) — a stringent test of electrostatic accuracy.

### 10. Transport Properties via Green-Kubo and NEMD

Shear viscosity, thermal conductivity, and diffusion for a Lennard-Jones fluid using both equilibrium (Green-Kubo) and non-equilibrium (NEMD) methods.

**New features needed:**
- `md_correlator` — Time correlation function infrastructure with multiple-tau correlator algorithm for efficient O(N log N) storage over long correlations
- `md_stress` — Per-atom and system-wide stress tensor from the virial. Requires pair forces exposed *before* summation into per-atom totals
- `md_nemd` — Lees-Edwards sliding boundaries for shear viscosity; Muller-Plathe reverse NEMD for thermal conductivity
- Triclinic simulation box — Lees-Edwards requires a tilted box with time-dependent tilt factor
- Block averaging for statistical error estimation

**Why it matters:** Green-Kubo requires accumulating time correlations over millions of steps, testing measurement infrastructure and correct ensemble generation. NEMD methods modify boundary conditions themselves — the most invasive type of plugin. Per-atom stress requires force loops to expose intermediate pair-force data, testing whether force plugins can share internal state through the resource system. Comparing equilibrium and non-equilibrium results for the same property provides an internal consistency check.

**Validation:** Shear viscosity at T\*=1.0, rho\*=0.85: eta\* = 3.26 +/- 0.07 — Green-Kubo and NEMD must agree ([Hess, Phys. Rev. E 66, 2002](https://doi.org/10.1103/PhysRevE.66.021202)). Thermal conductivity lambda\* = 7.0 +/- 0.3. NEMD at multiple shear rates should show Newtonian plateau at low rates and shear-thinning at high rates.

### Infrastructure Goals

- **GPU acceleration**: Flat neighbor list arrays, grid-based neighbor detection, maybe f32 GPU force kernels. 
- **Compile Time Dependency Plugin Checks**: We don't want people to worry about plugins missing other required plugins, or plugins that are known to not work together.
