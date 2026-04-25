# BPM MPI bond-migration test

A 3-sphere bonded chain drifting through a periodic domain that is split
across MPI ranks. Verifies that bonds **stay recognised** as atoms migrate
across rank boundaries — no silent "partner not found" skips.

## Setup

- 3 glass-like spheres, radius `r = 1 mm`, spaced `2 r` apart on the x-axis
- 2 bonds (auto-bonded at setup), BPM with `E = 1 GPa`, `G = 400 MPa`
- Domain `x ∈ [0, 10 mm]`, **periodic** in x (fiber laps the box)
- All three atoms drift at `v_x = 1 m/s`
- Timestep `dt = 1e-7 s`, run 200 000 steps (= 20 ms, ~2 periodic laps)

## What's measured

Every 1000 steps the recorder does an `all_reduce_sum` over:

| field             | meaning                                                 |
|-------------------|---------------------------------------------------------|
| `bond_count`      | total bond force evaluations this step, summed globally |
| `bond_missing`    | bonds skipped because the partner was not local / ghost |
| `nlocal_global`   | sum of local atom counts across ranks (should = 3)      |

Pass criterion: **`bond_missing == 0` at every sample and
`bond_count == 2` at every sample** (one per bond in the chain).

## Runs

### Single-process sanity

```bash
cargo run --release --example bond_mpi_drift --no-default-features -- \
    examples/bond_mpi_drift/config.toml
```

### 2-rank MPI (this is the real test)

```bash
cargo build --release --example bond_mpi_drift
mpiexec -n 2 target/release/examples/bond_mpi_drift \
    examples/bond_mpi_drift/config_mpi2.toml
```

`config_mpi2.toml` sets `processors_x = 2`, splitting the domain at
`x = 5 mm`. The chain starts straddling that plane (atoms at 3, 5, 7 mm) and
loops through the periodic box at 1 m/s, so each atom crosses the rank
boundary every 5 ms (~50 000 steps) for the full 200 000-step run.

## Verified result

Both the single-process and 2-rank MPI runs produce, every sample line:

```
step    N  atoms=3  bond_count=2  bond_missing=0  [min=2, max_miss=0]  OK
```

At setup, the plugin prints:

```
DemBond: auto-bonded 2 pairs
DemBond: extended ghost_cutoff 0.002400 → 0.005000
    (max r₀ = 0.002000 × multiplier 2.50)
```

confirming `DemBondPlugin` has pre-sized the ghost skin to cover the bond
reach before `neighbor_setup` runs.

## How to reproduce a failure

To see what a broken configuration looks like, set
`ghost_cutoff_multiplier = 0.0` in `[bonds]` **and** widen
`bond_tolerance` so auto-bonding creates bonds longer than the default
ghost skin (for example, place particles 4 mm apart with
`bond_tolerance = 2.1`). On MPI runs with ≥ 2 ranks in the bonded
direction, `bond_missing` will become non-zero and rank 0 will print the
"skipped N bond(s)" warning.
