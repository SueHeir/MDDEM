# mddem_neighbor

Neighbor list algorithms for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Determines which particle pairs are close enough to potentially interact, avoiding O(N^2) all-pairs checks.

## Algorithms

- **Brute Force** — O(N^2) all-pairs check. Useful for small systems and testing.
- **Sweep and Prune** — Sorts particles along one axis, then prunes pairs that are too far apart. Good general-purpose performance.
- **Bin-based** — Assigns particles to spatial bins (cells) and only checks neighboring bins. GPU-ready with flat linked-list cell storage (`bin_head`, `bin_next`, `bin_stencil` arrays).

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

app.add_plugins(NeighborPlugin {
    style: NeighborStyle::SweepAndPrune,
});
```

The neighbor list is rebuilt automatically when particles have moved more than half the skin distance since the last build.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
