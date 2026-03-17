# mddem_neighbor

Neighbor list construction for [MDDEM](https://github.com/SueHeir/MDDEM) simulations. Efficiently identifies particle pairs within interaction range, providing three strategies from O(N²) brute-force to O(N) spatial binning.

## Algorithms

| Algorithm | Complexity | Best for |
|---|---|---|
| **BruteForce** | O(N²) | Tiny systems (< 100 atoms), debugging |
| **SweepAndPrune** | O(N log N) | Small-to-medium systems without binning |
| **Bin** | O(N) expected | Production runs, large systems |

All strategies produce a **half neighbor list** in CSR (Compressed Sparse Row) format: each local atom `i` stores neighbors `j > i`, avoiding duplicate pairs.

## Key Types

- **`Neighbor`**: Main state holding CSR indices, bin grid, and rebuild tracking.
- **`NeighborConfig`**: TOML configuration for rebuild strategy and binning.
- **`NeighborStyle`**: Enum selecting BruteForce, SweepAndPrune, or Bin.
- **`PairIter`**: Iterator over (i, j) pairs from CSR list.

## Configuration

Add to your TOML simulation config:

```toml
[neighbor]
skin_fraction = 1.12       # Multiplier on cutoff radius (> 1.0 reduces rebuilds)
bin_size = 1.0             # Spatial bin width for bin-based strategy
every = 0                  # Rebuild interval: 0 = displacement-based only
check = true               # With every > 0, also check displacement threshold
sort_every = 1000          # Atom reordering by bin every N steps (0 = disabled)
rebuild_on_pbc_wrap = false # Force rebuild when atoms cross periodic boundaries
```

## Usage

Iterate over neighbor pairs:

```rust
for (i, j) in neighbor.pairs(nlocal) {
    let r_sq = distance_squared(&atoms.pos[i], &atoms.pos[j]);
    if r_sq < cutoff_sq {
        // process force pair
    }
}
```

## Rebuild Strategies

- **Displacement-based** (`every = 0`): Rebuilds when max atom displacement exceeds `(skin_fraction - 1) × min_cutoff`.
- **Periodic** (`every = N`): Rebuilds every N steps.
- **Hybrid** (`every = N, check = true`): Rebuilds on displacement OR every N steps, whichever comes first (like LAMMPS `neigh_modify every N check yes`).

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
