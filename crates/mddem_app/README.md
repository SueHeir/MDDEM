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
        app.add_resource(Comm::new())
            .add_setup_system(read_input, ScheduleSetupSet::PreSetup)
            .add_setup_system(setup, ScheduleSetupSet::PostSetup)
            .add_update_system(exchange, ScheduleSet::Exchange)
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
            .add(HertzNormalForcePlugin)
            .add(MindlinTangentialForcePlugin)
            .add(RotationalDynamicsPlugin)
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

**DEM use cases**
- `GranularDefaultPlugins` bundles Hertz normal + Mindlin tangential + rotational dynamics; users extend with custom plugins
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

Transition states by writing to `ResMut<NextState<S>>`:

```rust
fn maybe_transition(run_state: Res<RunState>, mut next: ResMut<NextState<Phase>>) {
    if run_state.total_cycle == 50_000 {
        next.set(Phase::Production);
    }
}
```
