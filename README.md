# MDDEM: Molecular Dynamics - Discrete Element Method
MDDEM (pronounced like "Madem" without the 'a') is an MPI-parallelized Molecular Dynamics / Discrete Element Method codebase written in Rust.
It uses [rsmpi](https://github.com/rsmpi/rsmpi) as a Rust wrapper around MPI and [nalgebra](https://github.com/dimforge/nalgebra) as a math library.

LAMMPS/LIGGGHTS-style MPI communication (exchange, borders, reverse force) is fully implemented for periodic boxes of spheres with Hertz normal contact, Mindlin tangential friction, and rotational dynamics.

## Running

```bash
mpiexec -n 4 ./target/release/MDDEM ./input
```

The path to the input script is passed as the first argument. Example input:

```
processors        2 2 1
neighbor          1.1 0.005
domain            0.0 0.025 0.0 0.025 0.0 0.025
periodic          p p p

randomparticleinsert    500 0.001 2500 8.7e9 0.3
randomparticlevelocity  0.5
dampening               0.95
friction_coefficient    0.4
thermo                  500
run                     5000000
```

| Command | Arguments | Description |
|---|---|---|
| `processors` | nx ny nz | MPI domain decomposition |
| `neighbor` | skin_fraction bin_min_size | Neighbor list cutoff multiplier and minimum bin size |
| `domain` | xlo xhi ylo yhi zlo zhi | Simulation box bounds (m) |
| `periodic` | x y z | Boundary condition per axis (`p` = periodic) |
| `randomparticleinsert` | N radius density youngs_mod poisson_ratio | Insert N spheres randomly |
| `randomparticlevelocity` | v | Assign random velocities with RMS speed v (m/s) |
| `dampening` | e | Normal restitution coefficient (0–1) |
| `friction_coefficient` | mu | Coulomb friction coefficient for tangential contacts (default 0.4) |
| `thermo` | interval | Print thermodynamic output every N steps; also controls GranularTemp.txt output |
| `run` | N | Number of timesteps |

## Physics

- **Normal contact**: Hertz elastic contact with viscoelastic damping (LAMMPS `hertz/material` equivalent)
- **Tangential contact**: Mindlin spring-history model with Coulomb friction cap and viscous tangential damping
- **Damping**: β derived from restitution coefficient e via β = −ln(e) / √(π² + ln²(e))
- **Integration**: Velocity Verlet for both translational and rotational degrees of freedom
- **Rotational dynamics**: Quaternion-based orientation tracking, angular velocity integration (I = 2/5 mr² for solid spheres)
- **Timestep**: Automatically computed as 5% of the Rayleigh wave period
- **MPI**: 3D domain decomposition with ghost atom forwarding to diagonal neighbors (corner-complete)

Not yet implemented: gravity.

## Benchmark: Haff's Cooling Law

A granular gas in a periodic box with no external forcing cools inelastically. Haff's law predicts:

$$T(t) = \frac{T_0}{(1 + t/t_c)^2} \quad \Rightarrow \quad T \sim t^{-2} \text{ for } t \gg t_c$$

Running `input_benchmark` (500 particles, φ = 13.4%, e = 0.95, 5M steps ≈ 20×t_c) and analysing with `data/haff_analysis.py` gives a measured late-time log-log slope of **−1.83** (theoretical maximum at 20×t_c is −1.91).

```bash
mpiexec -n 4 ./target/release/MDDEM ./input_benchmark
cd data && python haff_analysis.py
```

<img src="data/haff_comparison.png">

## Code Layout

MDDEM is built around a dependency-injection scheduler inspired by [Bevy](https://github.com/bevyengine/bevy). All simulation state lives in typed resources. Systems declare the resources they need as function arguments and the scheduler injects them automatically.

The per-step schedule runs in this order:

```rust
pub enum ScheduleSet {
    PreInitalIntegration,
    InitalIntegration,
    PostInitalIntegration,
    PreExchange,
    Exchange,
    PreNeighbor,
    Neighbor,
    PreForce,
    Force,
    PostForce,
    PreFinalIntegration,
    FinalIntegration,
    PostFinalIntegration,
}
```

Features and systems are registered via plugins:

```rust
pub struct CommunicationPlugin;

impl Plugin for CommunicationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Comm::new())
            .add_setup_system(read_input, ScheduleSetupSet::PreSetup)
            .add_setup_system(setup, ScheduleSetupSet::PostSetup)
            .add_update_system(exchange, ScheduleSet::Exchange)
            .add_update_system(borders, ScheduleSet::PreNeighbor)
            .add_update_system(reverse_send_force, ScheduleSet::PostForce);
    }
}
```

`main.rs` composes the simulation by adding all required plugins:

```rust
fn main() {
    App::new()
        .add_plugins(InputPlugin)
        .add_plugins(CommunicationPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin { brute_force: false })
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(VerletPlugin)
        .add_plugins(PrintPlugin)
        .start();
}
```

Per-atom DEM data (radius, Young's modulus, etc.) is stored in `DemAtom`, a typed extension registered with `AtomDataRegistry`. The registry packs and unpacks these fields automatically during MPI communication alongside the base `Atom` fields.

## Scheduler Features

The scheduler borrows several patterns from [Bevy](https://github.com/bevyengine/bevy) to make physics plugins composable and expressive. All features below are zero-cost at steady state — they are resolved at startup and impose no per-step overhead beyond the system function calls themselves.


---

### 2. `Local<T>` — Per-System Persistent State

`Local<T>` gives a system its own private state that persists across timesteps, initialized with `T::default()` on first use. Unlike `ResMut<T>`, a `Local` is not shared with any other system — it is owned exclusively by the system instance it was injected into.

```rust
pub fn tangential_force(
    atoms:        Res<Atom>,
    neighbor:     Res<Neighbor>,
    mut history:  Local<HashMap<(u32, u32), Vector3<f64>>>,
) {
    // `history` retains spring displacements from the previous step.
    // No need to register a global resource just to carry this data.
    for &(i, j) in neighbor.neighbor_list.iter() {
        let entry = history.entry((atoms.tag[i], atoms.tag[j])).or_default();
        // ... update spring displacement, compute tangential force ...
    }
}
```

**DEM use cases**
- Mindlin–Deresiewicz tangential spring history (contact-pair displacement accumulation)
- Per-system step counters or timers without global resources
- Cached neighbor list statistics between rebuilds

**MD use cases**
- FENE bond extension history
- Thermostat state (Nosé–Hoover chain variables) local to the thermostat system
- Per-system RNG state for stochastic force methods (Langevin)

---

### 3. Run Conditions — `.run_if()`

A run condition is any DI function that returns `bool`. Attach one to a system with `.run_if()`; the system is skipped when the condition returns `false`. Conditions use the same dependency injection as systems and can themselves hold `Local` state.

```rust
// Built-in pattern: run every N steps
pub fn every_n_steps(n: u64) -> impl Fn(Res<Verlet>) -> bool {
    move |verlet: Res<Verlet>| verlet.total_cycle % n == 0
}

// In a plugin's build():
app.add_update_system(
    write_restart.run_if(every_n_steps(10_000)),
    ScheduleSet::PostFinalIntegration,
);

app.add_update_system(
    print_granular_temperature.run_if(every_n_steps(500)),
    ScheduleSet::PostFinalIntegration,
);
```

Conditions compose naturally with ordering and states:

```rust
app.add_update_system(
    compute_heat_flux
        .run_if(in_state(SimPhase::Production))
        .label("heat_flux"),
    ScheduleSet::PostForce,
);
```

**DEM use cases**
- Restart file writing every N steps
- VTK/VTP output at a coarser interval than thermo output
- Neighbor-list validity checks (rebuild only when displacement threshold exceeded)
- Contact statistics collection during production only, not during initial settling

**MD use cases**
- Radial distribution function accumulation every N steps
- Mean-square displacement logging during NVT production after an NVE equilibration
- Pressure tensor averaging on a coarser schedule than force evaluation

---

### 4. System Ordering — `.label()`, `.before()`, `.after()`

Within a `ScheduleSet`, systems normally run in registration order. Explicit ordering constraints let you express dependencies without hard-coding plugin registration order. The scheduler performs a topological sort (Kahn's algorithm) at startup and panics on cycles.

```rust
// In ForcePlugin::build():
app.add_update_system(
    hertz_normal_force.label("hertz"),
    ScheduleSet::Force,
);
app.add_update_system(
    tangential_force.label("tangential").after("hertz"),
    ScheduleSet::Force,
);
app.add_update_system(
    lubrication_force.after("hertz").before("tangential"),
    ScheduleSet::Force,
);
```

Labels are plain strings and scoped to their `ScheduleSet` — `"hertz"` in `Force` and `"hertz"` in `PostForce` are independent.

**DEM use cases**
- Normal contact must be computed before tangential contact (tangential force depends on normal overlap)
- Cohesive/van-der-Waals corrections applied after the base Hertz kernel
- Heat conduction through contacts computed after contact geometry is known

**MD use cases**
- Short-range pair forces before long-range corrections (e.g., PPPM mesh forces after real-space forces)
- Bond forces before angle/dihedral forces that read the same atom force array
- Constraint projection (SHAKE/RATTLE) strictly after all unconstrained forces are accumulated

---

### 5. Simulation States

States let a simulation move through named phases (e.g., settling → production) without `if` guards scattered across system bodies. Each state transition is deferred to `PostFinalIntegration` so the current step always completes with a consistent state.

```rust
#[derive(Clone, PartialEq, Default)]
enum SimPhase {
    #[default]
    Settling,
    Production,
}

// In main.rs:
App::new()
    .add_plugins(StatesPlugin { initial: SimPhase::Settling })
    // Force computation only active during production
    .add_update_system(
        compute_forces.run_if(in_state(SimPhase::Production)),
        ScheduleSet::Force,
    )
    // Transition system: switch to Production after 50k settling steps
    .add_update_system(maybe_transition, ScheduleSet::PostFinalIntegration)
    ...

fn maybe_transition(verlet: Res<Verlet>, mut next: ResMut<NextState<SimPhase>>) {
    if verlet.total_cycle == 50_000 {
        next.set(SimPhase::Production);
    }
}
```

`in_state(S)` is itself a run condition and composes with `.run_if()`:

```rust
app.add_update_system(
    accumulate_rdf
        .run_if(in_state(SimPhase::Production))
        .run_if(every_n_steps(100)),
    ScheduleSet::PostFinalIntegration,
);
```

**DEM use cases**
- **Settling → Shear**: pack particles under gravity until kinetic energy drops below a threshold, then enable Lees–Edwards boundary shear
- **Fill → Compress → Release**: multi-stage die-compaction workflow; each stage activates different boundary motion systems
- **Initialization → Production → Quench**: granular cooling study where force model or restitution coefficient changes between phases

**MD use cases**
- **Equilibration → NVT → NPT**: run NVE first to relax bad contacts, couple thermostat, then couple barostat
- **Melting → Quench**: temperature ramp systems active only in the appropriate phase
- **Steered MD → Unbiased MD**: bias potential applied only during the pulling phase

---

### 6. Plugin Groups

A `PluginGroup` bundles multiple plugins into a single `add_plugins()` call. This is the intended mechanism for shipping a default simulation stack as a library crate.

```rust
pub struct GranularDefaultPlugins;

impl PluginGroup for GranularDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(DemAtomPlugin)
            .add(HertzNormalForcePlugin)
            .add(MindlinTangentialForcePlugin)
            .add(RotationalDynamicsPlugin)
    }
}
```

Groups can be composed into larger groups or overridden at the `main.rs` level. A research simulation that needs a custom force model can add additional plugins on top of the defaults:

```rust
fn main() {
    App::new()
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(CohesiveForcePlugin)   // additional force on top of defaults
        ...
        .start();
}
```

**DEM use cases**
- `GranularDefaultPlugins` bundles Hertz normal + Mindlin tangential + rotational dynamics; users extend with custom plugins
- Separate `DEMContactPlugin` from `DEMBondPlugin`; bond simulations add both, contact-only simulations add just the first
- Test harnesses use a stripped group (no I/O plugins) for fast unit-level integration tests

**MD use cases**
- `MDDefaultPlugins` ships Lennard-Jones + Velocity Verlet; a polymer simulation adds `FENEBondPlugin` on top
- GPU-offloaded force plugins can live in a separate group that is only added when a CUDA device is detected

---

## Future Goals

- **Library crate**: Publish to crates.io so simulations can be composed by importing plugins
- **Gravity**: Body force support
- **GPU readiness**: Split SoA arrays into separate x/y/z vecs; flat neighbor list arrays; grid-based neighbor detection
- **LEBC**: Lees–Edwards boundary conditions for shear flow
- **Polydispersity**: Variable radius support throughout the pipeline
