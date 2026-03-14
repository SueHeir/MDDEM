# DEM Compression

3-stage DEM compression workflow: insert particles, relax under gravity, then compress with 100x gravity. Demonstrates named stages with `StageEnum`, KE-based early termination, and per-stage config overrides.

## Stages

**Insert:** 300 particles are randomly inserted into a walled box and settle under normal gravity (9.81 m/s^2).

**Relax:** Once KE drops below 1e-5 J, the simulation advances to a relaxation stage that waits for KE to drop further (below 1e-7 J), ensuring particles are fully settled.

**Compress:** Gravity is increased to 981 m/s^2 (100x) via a per-stage config override, compressing the particle bed.

### How `main.rs` works

```rust
#[derive(Clone, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("insert")]
    Insert,
    #[stage("relax")]
    Relax,
    #[stage("compress")]
    Compress,
}
```

A `Phase` enum defines three simulation stages. `#[derive(StageEnum)]` generates the `StageName` trait, mapping each variant to its `[[run]]` stage name. Two custom systems handle early termination:

```rust
app.add_update_system(
    check_insert_settled.run_if(in_state(Phase::Insert)),
    ScheduleSet::PostFinalIntegration,
);
app.add_update_system(
    check_relaxed.run_if(in_state(Phase::Relax)),
    ScheduleSet::PostFinalIntegration,
);
```

Each system monitors KE and calls `next_state.set(...)` when the threshold is reached. `StageAdvancePlugin` automatically advances the `[[run]]` stage on each state transition.

The compress stage has no early termination — it runs for its full step count.

### Config: Per-stage overrides

```toml
[[run]]
name = "insert"
steps = 500000
thermo = 5000

[[run]]
name = "relax"
steps = 200000
thermo = 2000

[[run]]
name = "compress"
steps = 500000
thermo = 1000
gravity.gz = -981.0
```

The `gravity.gz = -981.0` override in the compress stage increases gravity 100x without changing the base `[gravity]` config. Per-stage overrides use dotted-key syntax to modify any config section for that stage only.

## Run

```bash
# Single-process
cargo run --example dem_compression -- examples/dem_compression/config.toml

# With MPI
cargo build --release --example dem_compression
mpiexec -n 4 ./target/release/examples/dem_compression examples/dem_compression/config.toml
```

## Parameters

| Parameter | Value |
|-----------|-------|
| Particles | 300 |
| Radius | 0.001 m |
| Density | 2500 kg/m^3 |
| Young's modulus | 8.7 GPa |
| Poisson ratio | 0.3 |
| Restitution | 0.5 |
| Friction | 0.4 |
| Gravity (insert/relax) | -9.81 m/s^2 (z) |
| Gravity (compress) | -981.0 m/s^2 (z) |
| Domain | 0.02 x 0.02 x 0.06 m |
| Boundaries | Non-periodic (all axes) |
| Insert stage | 500,000 steps, thermo 5000 |
| Relax stage | 200,000 steps, thermo 2000 |
| Compress stage | 500,000 steps, thermo 1000 |
| Insert KE threshold | 1e-5 J |
| Relax KE threshold | 1e-7 J |
