# mddem_app

Application framework for MDDEM. Provides `App`, `SubApp`, `Plugin`, `PluginGroup`, and `StatesPlugin` on top of `mddem_scheduler`.

## App

`App` is the top-level entry point. It holds a main `SubApp` and composes the simulation by adding plugins, resources, and systems.

```rust
fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}
```

## Plugins

A `Plugin` registers resources, setup systems, and update systems with the `App`:

```rust
pub struct CommunicationPlugin;

impl Plugin for CommunicationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(CommBuffers::default());

        app.add_setup_system(comm_read_input, ScheduleSetupSet::PreSetup)
            .add_setup_system(comm_setup, ScheduleSetupSet::PostSetup)
            .add_update_system(borders, ScheduleSet::PreNeighbor)
            .add_update_system(reverse_send_force, ScheduleSet::PostForce);
    }
}
```

Plugins are unique by default -- adding the same plugin twice will panic. Override `is_unique()` to return `false` if multiple instances are meaningful.

## Plugin Groups

A `PluginGroup` bundles multiple plugins into a single `add_plugins()` call:

```rust
pub struct GranularDefaultPlugins;

impl PluginGroup for GranularDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(DemAtomPlugin)
            .add(DemAtomInsertPlugin)
            .add(VelocityVerletPlugin)
            .add(HertzMindlinContactPlugin)
            .add(RotationalDynamicsPlugin)
            .add(GranularTempPlugin)
    }
}
```

Groups can be composed into larger groups or overridden at the `main.rs` level. A research simulation that needs a custom force model can add additional plugins on top of the defaults:

```rust
fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(CohesiveForcePlugin);   // additional force on top of defaults
    app.start();
}
```

### Disabling Plugins

Use `.disable::<P>()` on a `PluginGroupBuilder` to skip a plugin from a group. This is useful when you want most of a group's defaults but need to replace one plugin with your own:

```rust
pub struct MyPlugins;

impl PluginGroup for MyPlugins {
    fn build(self) -> PluginGroupBuilder {
        GranularDefaultPlugins.build()
            .disable::<HertzMindlinContactPlugin>()
            .add(CustomContactForcePlugin)
    }
}
```

Disabled plugins are simply not added — no error is raised if the disabled plugin was never in the group.

**Interaction with `requires_label()`**: If a disabled plugin registers a system that another system depends on via `.requires_label()`, the scheduler will panic at startup with a clear error. This is intentional — it prevents silent misconfiguration when replacing plugins. When disabling a plugin, make sure your replacement provides the same string labels (via `.label("name")`), or update the dependent systems. This is why replaceable cross-plugin ordering contracts should use string labels rather than function handles — a replacement system can adopt the same `.label()` string, but it cannot match another function's type identity.

**DEM use cases**
- `GranularDefaultPlugins` bundles Hertz-Mindlin fused contact + Velocity Verlet + rotational dynamics; users extend with custom plugins
- Separate `DEMContactPlugin` from `DEMBondPlugin`; bond simulations add both, contact-only simulations add just the first
- Test harnesses use a stripped group (no I/O plugins) for fast unit-level integration tests

**MD use cases**
- `MDDefaultPlugins` ships Lennard-Jones + Velocity Verlet; a polymer simulation adds `FENEBondPlugin` on top
- GPU-offloaded force plugins can live in a separate group that is only added when a CUDA device is detected

## StatesPlugin

Registers `CurrentState<S>` and `NextState<S>` resources and wires up end-of-step state transitions at `PostFinalIntegration`:

```rust
#[derive(Clone, PartialEq, Default)]
enum Phase { #[default] Settling, Production }

let mut app = App::new();
app.add_plugins(StatesPlugin { initial: Phase::Settling })
    .add_update_system(
        compute_forces.run_if(in_state(Phase::Production)),
        ScheduleSet::Force,
    );
app.start();
```

### Combining with `StageEnum` and `StageAdvancePlugin`

`StatesPlugin` on its own knows nothing about `[[run]]` stages — it only manages `CurrentState<S>` / `NextState<S>` and provides `in_state()` run conditions. You can use it standalone for runtime logic (enabling/disabling systems based on state) without named `[[run]]` stages.

`StageAdvancePlugin` is the bridge between the state machine and the `[[run]]` stage system. It watches `CurrentState<S>` for changes and triggers early `[[run]]` stage advancement when a state transition occurs. You need both plugins together when you want state transitions to also advance stages:

```rust
#[derive(Clone, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("filling")]
    Filling,
    #[stage("flowing")]
    Flowing,
}

// StatesPlugin: state machine (CurrentState, NextState, in_state())
// StageAdvancePlugin: bridges state transitions to [[run]] stage advancement
app.add_plugins(StatesPlugin { initial: Phase::Filling })
    .add_plugins(StageAdvancePlugin::<Phase>::new());
```

Stage advancement happens in two ways: a `[[run]]` stage can complete its full step count (the scheduler advances automatically), or a state transition via `next_state.set()` can trigger early advancement through `StageAdvancePlugin`. Each `[[run]]` stage can have its own step count, thermo interval, and config overrides.

Transition states by writing to `ResMut<NextState<S>>`:

```rust
fn maybe_transition(run_state: Res<RunState>, mut next: ResMut<NextState<Phase>>) {
    if run_state.total_cycle == 50_000 {
        next.set(Phase::Production);
    }
}
```

## Multi-Stage Simulations

MDDEM supports multi-stage runs via `[[run]]` TOML arrays. Each stage runs sequentially with its own step count, thermo interval, and optional config overrides. Any config section can be overridden per-stage using dotted-key syntax:

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
vtp_interval = 1000

[[run]]
name = "compress"
steps = 500000
thermo = 1000
gravity.gz = -981.0        # override [gravity] gz for this stage only
```

Per-stage overrides (like `gravity.gz` above) are applied on top of the base config when the stage begins. Setup systems re-run at each stage boundary, so they pick up the updated values automatically.

The scheduler executes stages sequentially, running `Setup -> Run -> Setup -> Run -> ... -> End`. Each stage transition increments `SchedulerManager.index`, and setup systems re-run to pick up the new stage's configuration. Systems that should only run once (e.g., lattice initialization) can use the `first_stage_only()` run condition:

```rust
app.add_setup_system(
    fcc_insert.run_if(first_stage_only()),
    ScheduleSetupSet::Setup,
);
```

Single-stage `[run]` (table syntax) is still supported for backwards compatibility.

### Combining with StatesPlugin

`StatesPlugin` provides finer-grained control within or across stages. For example, a system can be active only during a specific named state regardless of which `[[run]]` stage is executing:

```rust
app.add_update_system(
    apply_shear_boundary
        .run_if(in_state(Phase::Shearing)),
    ScheduleSet::PreExchange,
);
```

### Programmatic multi-stage setup

For complex workflows beyond what TOML can express, construct `RunConfig` directly in Rust:

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_resource(RunConfig {
        stages: vec![
            StageConfig { name: Some("settle".into()), steps: 10000, thermo: 100, ..Default::default() },
            StageConfig { name: Some("run".into()), steps: 50000, thermo: 500, ..Default::default() },
        ],
    });
    app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
    app.start();
}
```
