# sim_scheduler

A lightweight, Bevy-inspired dependency-injection scheduler for scientific simulations.

## What It Does

Systems are plain functions that declare the resources they need as parameters. The scheduler automatically injects typed resources, manages execution order across user-defined lifecycle phases (`Schedule`), and supports conditional execution and simulation states.

## Key Types

- **`Schedule`** — Trait for defining custom execution phases. Implement on your own enum or use `#[derive(Schedule)]`.
- **`Res<T>` / `ResMut<T>`** — Resource access (read-only / mutable). The scheduler validates that all required resources are registered before execution starts.
- **`Local<T>`** — Per-system persistent state, unshared with other systems.
- **`Option<Res<T>>`** — Optional resources; systems are not skipped if missing.

## Ordering & Conditions

Within a schedule phase, systems are topologically sorted using `.before()` / `.after()` constraints. Run conditions gate execution:

```rust
fn every_n_steps(n: u64) -> impl Fn(Res<RunState>) -> bool {
    move |rs: Res<RunState>| rs.total_cycle % n == 0
}

app.add_update_system(
    write_restart.run_if(every_n_steps(10_000)),
    ScheduleSet::PostFinalIntegration,
);
```

## Quick Example

```rust
use sim_scheduler::prelude::*;

struct Temperature(f64);

fn compute_forces(temp: Res<Temperature>) {
    let _t = temp.0;
}

let mut scheduler = Scheduler::default();
scheduler.add_resource(Temperature(300.0));
// Use any type implementing Schedule as the phase argument
scheduler.add_update_system(compute_forces, MySchedule::Force);
```

## System Groups

`SystemGroup` bundles multiple systems into a single composite unit with its own
internal phase ordering and optional looping. This is useful when part of your
simulation needs to iterate until some convergence criterion is met before the
outer timestep moves on — for example, solving a coupled field on a mesh, or
running a Newton-Raphson correction loop.

### Defining inner phases

Inner phases use the same `Schedule` trait (or `#[derive(Schedule)]`)
as top-level phases — no magic numbers:

```rust
use sim_scheduler::prelude::*;

#[derive(Clone, Copy, Debug, Schedule)]
enum SolverPhase {
    #[phase(0)] Assemble,
    #[phase(1)] Solve,
    #[phase(2)] Check,
}
```

### Building a looping group

Say you have a coupled field solve that needs to run repeatedly within each
timestep. You assemble some data, run the solver, then check if the answer has
settled. If the residual is still too large the loop repeats, up to a safety cap:

```rust
struct FieldData { /* ... */ }
struct Residual(f64);

fn assemble_system(mut field: ResMut<FieldData>) {
    // Build the system of equations from current particle/mesh state
}

fn run_solver(mut field: ResMut<FieldData>) {
    // Solve the linear or nonlinear system
}

fn compute_residual(field: Res<FieldData>, mut res: ResMut<Residual>) {
    // Measure how much the solution changed this iteration
    res.0 = 1e-4; // placeholder
}

fn not_converged(res: Res<Residual>) -> bool {
    res.0 > 1e-6
}

// Register the group — it looks like a single system to the outer schedule
app.add_update_system(
    SystemGroup::new("field_solve")
        .add_system(assemble_system,  SolverPhase::Assemble)
        .add_system(run_solver,       SolverPhase::Solve)
        .add_system(compute_residual, SolverPhase::Check)
        .loop_while(not_converged, 50),   // converge or stop after 50 iters
    ScheduleSet::Force,
);
```

Each outer timestep runs the three inner systems in phase order, repeating until
`not_converged` returns `false` or the 50-iteration cap is hit.

### Nesting groups

Groups can contain other groups. For example, an outer correction loop can embed
an inner linear solve:

```rust
let linear_solve = SystemGroup::new("linear_solve")
    .add_system(assemble_matrix, LinearPhase::Assemble)
    .add_system(iterate_solver,  LinearPhase::Solve)
    .add_system(check_linear,    LinearPhase::Check)
    .loop_while(linear_not_converged, 200);

let outer_loop = SystemGroup::new("outer_loop")
    .add_system(compute_rhs,         OuterPhase::Setup)
    .add_group(linear_solve,         OuterPhase::Solve)
    .add_system(update_solution,     OuterPhase::Update)
    .add_system(check_outer,         OuterPhase::Check)
    .loop_while(outer_not_converged, 20);

app.add_update_system(outer_loop, ScheduleSet::Force);
```

### Composability

Because `SystemGroup` implements `IntoSystem`, it gets the same fluent API
as ordinary systems — `.run_if()`, `.label()`, `.before()`, `.after()`:

```rust
app.add_update_system(
    SystemGroup::new("my_solve")
        .add_system(solve_a, Phase::A)
        .add_system(solve_b, Phase::B)
        .loop_while(not_converged, 100)
        .label("my_solve")
        .after("neighbor_build")
        .run_if(in_stage("active")),
    ScheduleSet::Force,
);
```

### Timing and visualization

Per-inner-system timing is recorded automatically. The end-of-run summary shows
an indented breakdown inside each group:

```
--- Per-system timing (1000 steps) ---
System                                               Time(s)        %
----------------------------------------------------------------------
field_solve                                            0.8765    32.0%
  Assemble: assemble_system                            0.1100     4.0%
  Solve: run_solver                                    0.4321    15.8%
  Check: compute_residual                              0.1234     4.5%
```

When `--schedule` is passed, the DOT file expands each group into a subgraph
cluster with phase sub-clusters, execution-order edges, and a green back-edge
for the loop condition — so the iterative structure is visible in the schedule
graph.

See inline crate documentation for full details on system states, labels, and run conditions.
