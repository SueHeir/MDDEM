# mddem_neighbor

Neighbor list algorithms for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Determines which particle pairs are close enough to potentially interact, avoiding O(N^2) all-pairs checks.

## Algorithms

- **Brute Force** — O(N^2) all-pairs check. Useful for small systems and testing.
- **Sweep and Prune** — Sorts particles along one axis, then prunes pairs that are too far apart. Good general-purpose performance.
- **Bin-based** — Assigns particles to spatial bins (cells) and only checks neighboring bins. GPU-ready with flat CSR storage (`sorted_atoms`, `bin_start` arrays) and forward-only stencil. Atoms are sorted by bin for cache-friendly access.

## Pair Iteration

Force systems iterate over neighbor pairs using `neighbor.pairs(nlocal)`:

```rust
fn my_force(atoms: Res<Atom>, neighbor: Res<Neighbor>) {
    let nlocal = atoms.nlocal;
    for (i, j) in neighbor.pairs(nlocal) {
        // i is always a local atom, j may be local or ghost
    }
}
```

## Configuration

```toml
[neighbor]
skin_fraction = 1.1    # Neighbor list cutoff = skin_fraction * max_interaction_radius
bin_size = 0.005       # Minimum bin size for bin-based neighbor list
```

## Usage

`NeighborPlugin` is included in `CorePlugins`. To use a specific algorithm:

```rust
use mddem_neighbor::{NeighborPlugin, NeighborStyle};

// CorePlugins uses Bin by default; override with a different algorithm:
app.add_plugins(NeighborPlugin {
    style: NeighborStyle::SweepAndPrune,
});
```

The neighbor list is rebuilt automatically when particles have moved more than half the skin distance since the last build. Bin-sorting also reorders all registered `AtomData` extensions via `apply_permutation`.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
