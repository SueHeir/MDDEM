# sim_app

Plugin-based application framework for scientific simulations.

## Overview

`sim_app` provides the core abstractions for building modular simulation applications:

- **`App`** — Central container holding resources, systems, and plugins
- **`Plugin`** — Trait for self-contained modules that register resources and systems
- **`PluginGroup`** — Bundles plugins together for reuse and composition
- **`SubApp`** — Self-contained scheduler with its own resource store (one per App)

## Quick Start

```rust
use sim_app::prelude::*;
use sim_scheduler::SchedulePhase;

// Define a custom schedule phase
#[derive(Clone, Copy, Debug)]
enum MySchedule { Update }

impl SchedulePhase for MySchedule {
    fn to_index(&self) -> u32 { 0 }
    fn name(&self) -> &'static str { "Update" }
}

// Define a plugin
struct MyPlugin;
impl Plugin for MyPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(my_system, MySchedule::Update);
    }
}

fn my_system() {
    // Your simulation logic here
}

fn main() {
    App::new()
        .add_plugins(MyPlugin)
        .start();
}
```

## Plugin Groups

Group multiple plugins for composition and reuse:

```rust
impl PluginGroup for MyPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(PhysicsPlugin)
            .add(OutputPlugin)
    }
}

// Extend or replace plugins
impl PluginGroup for MyCustomPlugins {
    fn build(self) -> PluginGroupBuilder {
        MyPlugins.build()
            .disable::<OutputPlugin>()
            .add(CustomOutputPlugin)
    }
}
```

## State Management

Use `StatesPlugin` for runtime state machines:

```rust
#[derive(Clone, PartialEq, Default)]
enum Phase { #[default] Settling, Production }

app.add_plugins(StatesPlugin { initial: Phase::Settling })
    .add_update_system(
        my_system.run_if(in_state(Phase::Production)),
        ScheduleSet::Force,
    );
```

Combine with `StageAdvancePlugin` and `#[derive(StageEnum)]` for multi-stage workflows coordinated with TOML `[[run]]` stages.

## Key Features

- **Modular design** — Every feature is a plugin; compose and replace with ease
- **Dependency tracking** — Plugins declare dependencies; registration order is validated
- **Unique plugin tracking** — Prevents duplicate plugin registration by default
- **Config generation** — Plugins provide default TOML snippets via `--generate-config`
- **Cleanups** — Register cleanup functions to run after simulation
- **Direct scheduler access** — Use `app.main_mut()` for advanced scheduling control
