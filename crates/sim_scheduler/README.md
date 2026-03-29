# sim_scheduler

A lightweight, Bevy-inspired dependency-injection scheduler for scientific simulations.

## What It Does

Systems are functions that declare the resources they need as parameters. The scheduler automatically injects typed resources, manages execution order across user-defined lifecycle phases (`Schedule`), and supports conditional execution and simulation states.

## Key Types

- **`Schedule`** — Trait for defining custom execution phases. Implement on your own enum or use `#[derive(Schedule)]`.
- **`Res<T>` / `ResMut<T>`** — Resource access (read-only / mutable). The scheduler validates that all required resources are registered before execution starts.
- **`Local<T>`** — Per-system persistent state, unshared with other systems.
- **`Option<Res<T>>`** — Optional resources; systems are not skipped if missing.

## Ordering & Conditions

Within a schedule phase, systems are topologically sorted using `.before()` / `.after()` constraints. Run conditions gate execution:

```rust
struct Timestep { current: u64 }

fn every_n_steps(n: u64) -> impl Fn(Res<Timestep>) -> bool {
    move |ts: Res<Timestep>| ts.current % n == 0
}

app.add_update_system(
    dump_particles.run_if(every_n_steps(1_000)),
    VerletSchedule::PostFinalIntegration,
);
```

## Quick Example

A minimal velocity-Verlet DEM schedule:

```rust
use sim_scheduler::prelude::*;

#[derive(Clone, Copy, Debug, Schedule)]
enum VerletSchedule {
    #[phase(0)] InitialIntegration,
    #[phase(1)] Exchange,
    #[phase(2)] Neighbor,
    #[phase(3)] Force,
    #[phase(4)] FinalIntegration,
    #[phase(5)] PostFinalIntegration,
}

struct Particles { pos: Vec<[f64; 3]>, vel: Vec<[f64; 3]>, force: Vec<[f64; 3]> }

fn initial_integrate(mut p: ResMut<Particles>) {
    // First half-step: update velocities and positions
}

fn compute_hertz_forces(mut p: ResMut<Particles>) {
    // Hertz-Mindlin contact force loop
}

fn final_integrate(mut p: ResMut<Particles>) {
    // Second half-step: update velocities
}

let mut scheduler = Scheduler::default();
scheduler.add_resource(Particles { pos: vec![], vel: vec![], force: vec![] });
scheduler.add_update_system(initial_integrate, VerletSchedule::InitialIntegration);
scheduler.add_update_system(compute_hertz_forces, VerletSchedule::Force);
scheduler.add_update_system(final_integrate, VerletSchedule::FinalIntegration);
```

## System Groups

`SystemGroup` bundles multiple systems into a single composite unit with its own
internal phase ordering and optional looping. 

### Defining inner phases

Inner phases use the same `Schedule` trait (or `#[derive(Schedule)]`)
as top-level phases:

```rust
use sim_scheduler::prelude::*;

#[derive(Clone, Copy, Debug, Schedule)]
enum RelaxPhase {
    #[phase(0)] ComputeForces,
    #[phase(1)] MoveParticles,
    #[phase(2)] CheckOverlap,
}

struct Overlap(f64);

fn compute_repulsion(mut p: ResMut<Particles>) {
    // Compute soft repulsive forces to resolve overlaps
}

fn move_particles(mut p: ResMut<Particles>) {
    // Damped displacement toward equilibrium
}

fn measure_overlap(p: Res<Particles>, mut ovlp: ResMut<Overlap>) {
    // Find maximum particle-particle overlap ratio
}

fn has_overlap(ovlp: Res<Overlap>) -> bool {
    ovlp.0 > 1e-4
}

// Relax overlaps after insertion — runs as a single system in the outer schedule
app.add_update_system(
    SystemGroup::new("overlap_relaxation")
        .add_system(compute_repulsion, RelaxPhase::ComputeForces)
        .add_system(move_particles,    RelaxPhase::MoveParticles)
        .add_system(measure_overlap,   RelaxPhase::CheckOverlap)
        .loop_while(has_overlap, 100),   // iterate until resolved or 100 cap
    VerletSchedule::PostFinalIntegration,
);
```

Each outer timestep runs the three inner systems in phase order, repeating until
`has_overlap` returns `false` or the 100-iteration cap is hit.

### Nesting groups

Groups can contain other groups. You should be careful where you do this. System calls have a small amount of overhead.


### Composability

Because `SystemGroup` implements `IntoSystem`, it gets the same API
as ordinary systems — `.run_if()`, `.label()`, `.before()`, `.after()`:

```rust
app.add_update_system(
    SystemGroup::new("overlap_relaxation")
        .add_system(compute_repulsion, RelaxPhase::ComputeForces)
        .add_system(move_particles,    RelaxPhase::MoveParticles)
        .loop_while(has_overlap, 100)
        .label("relaxation")
        .after("insert_particles")
        .run_if(in_stage("settling")),
    VerletSchedule::PostFinalIntegration,
);
```

### Timing and visualization

Per-system timing is recorded automatically. At the end of a run, MDDEM prints
a sorted breakdown showing where wall-clock time was spent:

```
--- Per-system timing (10000 steps) ---
System                                               Time(s)        %
----------------------------------------------------------------------
hertz_mindlin_contact                                  1.2340    45.2%
build_neighbor_list                                    0.5678    20.8%
initial_integrate                                      0.2100     7.7%
final_integrate                                        0.1980     7.3%
exchange_atoms                                         0.1456     5.3%
forward_comm_positions                                 0.0987     3.6%
dump_particles                                         0.0543     2.0%
----------------------------------------------------------------------
TOTAL                                                  2.7304   100.0%
```

SystemGroups show an indented breakdown of their inner systems:

```
overlap_relaxation                                     0.3210    11.8%
  ComputeForces: compute_repulsion                     0.1500     5.5%
  MoveParticles: move_particles                        0.1100     4.0%
  CheckOverlap: measure_overlap                        0.0610     2.2%
```

Pass `--schedule` to emit a Graphviz DOT file (`schedule.dot`) showing the full
execution graph. SystemGroups appear as subgraph clusters with phase
sub-clusters, execution-order edges, and green back-edges for loop conditions.

See inline crate documentation for full details on system states, labels, and run conditions.
