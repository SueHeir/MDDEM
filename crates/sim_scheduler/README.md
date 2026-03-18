# sim_scheduler

A lightweight, Bevy-inspired dependency-injection scheduler for scientific simulations.

## What It Does

Systems are plain functions that declare the resources they need as parameters. The scheduler automatically injects typed resources, manages execution order across user-defined lifecycle phases (`SchedulePhase`), and supports conditional execution and simulation states.

## Key Types

- **`SchedulePhase`** — Trait for defining custom execution phases. Implement on your own enum or use `#[derive(SchedulePhase)]`.
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
// Use any type implementing SchedulePhase as the phase argument
scheduler.add_update_system(compute_forces, MySchedule::Force);
```

See inline crate documentation for full details on system states, labels, and run conditions.
