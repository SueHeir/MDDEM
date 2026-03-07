# dem_gravity

Configurable gravity body force plugin for [MDDEM](https://github.com/SueHeir/MDDEM) simulations.

Applies `F = m * g` to all local (non-ghost) atoms each timestep at `ScheduleSet::Force`.

## Configuration

```toml
[gravity]
gx = 0.0       # x-component (m/s^2, default 0)
gy = 0.0       # y-component (m/s^2, default 0)
gz = -9.81     # z-component (m/s^2, default -9.81)
```

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
    .add_plugins(GranularDefaultPlugins)
    .add_plugins(GravityPlugin);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
