# mddem_scheduler

Dependency-injection scheduler for MDDEM, inspired by [Bevy](https://github.com/bevyengine/bevy). All simulation state lives in typed resources. Systems declare the resources they need as function arguments and the scheduler injects them automatically.

## Schedule Sets

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

Setup systems run once before the run loop, in order: `PreSetup`, `Setup`, `PostSetup`.

## Resources: `Res<T>` and `ResMut<T>`

Systems declare shared state via `Res<T>` (read) and `ResMut<T>` (read-write). The scheduler borrows them from a central `HashMap<TypeId, RefCell<Box<dyn Any>>>`.

```rust
fn my_system(atoms: Res<Atom>, mut forces: ResMut<ForceArray>) {
    // atoms is read-only, forces is mutable
}
```

## `Local<T>` -- Per-System Persistent State

`Local<T>` gives a system its own private state that persists across timesteps, initialized with `T::default()` on first use. Unlike `ResMut<T>`, a `Local` is not shared with any other system.

```rust
pub fn tangential_force(
    atoms:        Res<Atom>,
    neighbor:     Res<Neighbor>,
    mut history:  Local<HashMap<(u32, u32), Vector3<f64>>>,
) {
    // `history` retains spring displacements from the previous step.
    for &(i, j) in neighbor.neighbor_list.iter() {
        let entry = history.entry((atoms.tag[i], atoms.tag[j])).or_default();
        // ... update spring displacement, compute tangential force ...
    }
}
```

## Run Conditions -- `.run_if()`

A run condition is any DI function that returns `bool`. Attach one to a system with `.run_if()`; the system is skipped when the condition returns `false`.

```rust
pub fn every_n_steps(n: u64) -> impl Fn(Res<Verlet>) -> bool {
    move |verlet: Res<Verlet>| verlet.total_cycle % n == 0
}

app.add_update_system(
    write_restart.run_if(every_n_steps(10_000)),
    ScheduleSet::PostFinalIntegration,
);
```

## System Ordering -- `.label()`, `.before()`, `.after()`

Within a `ScheduleSet`, systems normally run in registration order. Explicit ordering constraints let you express dependencies without hard-coding plugin registration order. The scheduler performs a topological sort (Kahn's algorithm) at startup and panics on cycles.

```rust
app.add_update_system(
    hertz_normal_force.label("hertz"),
    ScheduleSet::Force,
);
app.add_update_system(
    tangential_force.label("tangential").after("hertz"),
    ScheduleSet::Force,
);
```

Labels are plain strings and scoped to their `ScheduleSet`.

## Simulation States

States let a simulation move through named phases (e.g., settling -> production) without `if` guards scattered across system bodies. Each state transition is deferred to `PostFinalIntegration` so the current step always completes with a consistent state.

```rust
#[derive(Clone, PartialEq, Default)]
enum SimPhase {
    #[default]
    Settling,
    Production,
}

app.add_update_system(
    compute_forces.run_if(in_state(SimPhase::Production)),
    ScheduleSet::Force,
);
```

## Schedule Visualization

Pass `--schedule` on the command line to print the compiled schedule to the terminal and write a Graphviz DOT file:

```bash
mpiexec -n 1 ./target/release/MDDEM ./input --schedule
```

This produces `schedule.dot` in the working directory. Generate an image with:

```bash
dot -Tpng schedule.dot -o output.png
```

The DOT output includes:
- Setup systems grouped by `ScheduleSetupSet` (blue clusters)
- Update systems grouped by `ScheduleSet` (yellow clusters)
- Red dashed edges for `.before()`/`.after()` ordering constraints
- Blue edges for implicit ScheduleSet ordering
- Green loop-back edge showing the per-step run loop

Example output:

<img src="../../output.png">
