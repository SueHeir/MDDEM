# dem_granular

Granular physics for [MDDEM](https://github.com/SueHeir/MDDEM): Hertz normal contact, Mindlin tangential friction, and rotational dynamics.

## Physics Models

### Hertz Normal Contact (`HertzNormalForcePlugin`)
Elastic Hertz contact with viscoelastic damping (equivalent to LAMMPS `pair_style granular hertz/material`):
- Normal force: `F_n = k_n * delta^(3/2) - gamma_n * v_n`
- Effective stiffness: `k_n = (4/3) * E* * sqrt(R*)`
- Damping from restitution coefficient via `beta = -ln(e) / sqrt(pi^2 + ln^2(e))`

### Mindlin Tangential Friction (`MindlinTangentialForcePlugin`)
Spring-history tangential force with Coulomb friction cap:
- Tangential spring accumulates relative tangential displacement
- Spring truncated at Coulomb limit: `|F_t| <= mu * |F_n|`
- Viscous tangential damping
- Ordered after Hertz via `.after("hertz_normal")`

### Rotational Dynamics (`RotationalDynamicsPlugin`)
Quaternion-based Velocity Verlet for angular degrees of freedom:
- Moment of inertia: `I = (2/5) * m * r^2` (solid sphere)
- Angular velocity and angular momentum integration
- Quaternion orientation updates

## Usage

### Granular Temperature (`GranularTempPlugin`)
Time series of granular temperature `T_g = (1/3N) * sum(m * v^2)` written to `data/GranularTemp.txt`. Included in `GranularDefaultPlugins` by default.

### Contact History
Tangential spring-history contacts are stored per-atom in a `ContactHistoryStore` registered as `AtomData`. Contact history automatically travels with atoms during MPI exchange, is reordered during bin-sorting, and is saved/restored in restart files.

## Usage

`GranularDefaultPlugins` bundles all plugins plus `DemAtomPlugin` and `DemAtomInsertPlugin`:

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
app.start();
```

Individual plugins can be added separately for custom force model configurations.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
