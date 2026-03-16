# dem_granular

Granular physics for [MDDEM](https://github.com/SueHeir/MDDEM): Hertz-Mindlin contact with rolling resistance and cohesion, rotational dynamics, and granular temperature.

## Physics Models

### Hertz-Mindlin Contact (`HertzMindlinContactPlugin`)

Fused normal + tangential + rolling resistance contact force in a single pair loop (equivalent to LAMMPS `pair_style granular hertz/material`).

**Normal force** (three modes, mutually exclusive):

*Standard Hertz* (default):
- `F_n = k_n * delta^(3/2) - gamma_n * v_n` (clamped non-negative)
- Effective stiffness: `k_n = (4/3) * E* * sqrt(R*)`
- Damping from restitution coefficient via `beta = -ln(e) / sqrt(pi^2 + ln^2(e))`

*JKR adhesion* (when `surface_energy > 0`):
- Adds adhesive force: `F_adhesion = 1.5 * pi * gamma * R*`
- Particles attract within pull-off distance even without geometric overlap
- Full contact regime: Hertz + damping - adhesion (net force can be attractive)
- Adhesion-only regime: small gap but within pull-off distance — only adhesion applied, no tangential/history

*SJKR cohesion* (when `cohesion_energy > 0`):
- `F_cohesion = cohesion_energy * pi * delta * R*` subtracted from normal force
- Simpler than JKR — purely overlap-dependent, no pull-off at separation
- Net force can be attractive during overlap

**Tangential force** (Mindlin spring-history):
- Incremental spring displacement rotated to current bond direction each step
- Coulomb cap: `|F_t| <= mu * |F_n|`
- Viscous tangential damping: `gamma_t = 2 * sqrt(5/3) * beta * sqrt(k_t * m_r)`
- Torques accumulated on both particles

**Rolling resistance** (when `rolling_friction > 0`):
- Constant torque model opposing rolling motion
- Computes relative angular velocity, removes twisting component (normal projection)
- Rolling torque magnitude: `tau = mu_r * |F_n| * R*`
- Applied equally and oppositely to both particles

Contact forces contribute to the shared `VirialStress` tensor for stress analysis.

The separate `HertzNormalForcePlugin` and `MindlinTangentialForcePlugin` are also available for custom configurations, but `GranularDefaultPlugins` uses the fused plugin for better performance.

### Rotational Dynamics (`RotationalDynamicsPlugin`)

Quaternion-based Velocity Verlet for angular degrees of freedom:
- Moment of inertia: `I = (2/5) * m * r^2` (solid sphere)
- Angular velocity and angular momentum integration
- Quaternion orientation updates

### Granular Temperature (`GranularTempPlugin`)

Time series of granular temperature `T_g = (1/3N) * sum(m * v^2)` written to `data/GranularTemp.txt`. Columns: step, physical time, granular temperature, global KE, momentum magnitude. Included in `GranularDefaultPlugins` by default.

### Contact History

Tangential spring-history contacts are stored per-atom in a `ContactHistoryStore` registered as `AtomData`. Contact history automatically travels with atoms during MPI exchange, is reordered during bin-sorting, and is saved/restored in restart files. Stale contacts are pruned every step.

## Configuration

Material properties that control contact behavior are set in `[[dem.materials]]`:

```toml
[[dem.materials]]
name = "glass"
youngs_mod = 8.7e9
poisson_ratio = 0.3
restitution = 0.95
friction = 0.4
rolling_friction = 0.1       # rolling resistance coefficient (default 0.0)
# cohesion_energy = 0.05      # SJKR cohesion J/m² (default 0.0)
# surface_energy = 0.05       # JKR adhesion J/m² (default 0.0)
```

## Usage

`GranularDefaultPlugins` bundles all plugins plus `DemAtomPlugin`, `DemAtomInsertPlugin`, and `VelocityVerletPlugin` (translational integration):

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(GranularDefaultPlugins);
app.start();
```

Individual plugins can be added separately for custom force model configurations.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
