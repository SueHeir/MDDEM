# md_type_rdf

Type-filtered radial distribution function (RDF) plugin for MDDEM. Measures pair correlation functions g(r) between specific atom type pairs in multi-component systems.

## Overview

This crate computes g(r) for individual atom type pairs—essential for binary LJ mixtures, alloys, water models, polymer blends, and other multi-type simulations. Unlike the simpler `md_measure` crate (which computes a global RDF over all atoms), this plugin enables simultaneous measurement of multiple type-filtered RDFs: g₀₀(r), g₀₁(r), g₁₁(r), etc.

## Key Types

- **`TypeRdfEntry`**: Configuration for one (type_i, type_j) RDF pair. Deserialized from `[[type_rdf]]` TOML array sections.
- **`TypeRdfAccumulator`**: Accumulates histogram bins and sample count for a single type pair. Stores cumulative normalized g(r); divide by `n_samples` for time-averaged result.
- **`TypeRdfAccumulators`**: Collection of all active accumulators (one per config entry).
- **`TypeRdfPlugin`**: The main plugin that registers systems and resources.

## Configuration

Use `[[type_rdf]]` array sections in your TOML config:

```toml
[[type_rdf]]
type_i = 0           # First atom type index
type_j = 0           # Second atom type index
bins = 200           # Histogram bins (default: 200)
cutoff = 4.0         # RDF cutoff distance (default: 3.0)
interval = 100       # Accumulate every N steps (default: 100)
output_interval = 5000  # Write output every N steps (default: 1000)

[[type_rdf]]
type_i = 0
type_j = 1
bins = 200
cutoff = 4.0
interval = 100
output_interval = 5000
```

## Output

Each configured pair produces a separate text file: `data/type_rdf_<type_i>_<type_j>.txt`. Files are written in 2-column format with headers:

```
# Type-filtered RDF: types (0, 1), 10 samples
# r g(r)
0.015000 0.000000
0.045000 0.000075
...
```

For an ideal gas, g(r) ≈ 1.0 at all distances. Peaks indicate preferred inter-particle distances.

## How It Works

- **Brute-force O(N²) pair counting**: Uses minimum-image convention for periodic boundaries. More expensive than neighbor lists, but guarantees correctness for arbitrary RDF cutoffs (which often exceed the force cutoff).
- **Accumulation interval**: Histogram updated every `interval` steps to reduce computational overhead.
- **Normalization**: g(r) = (V / N_pairs) × n(r) / (4π/3 × (r_high³ - r_low³)), where N_pairs = N_i×(N_i-1)/2 for same-type and N_i×N_j for cross-type.
- **Parallel support**: Communicates partial histograms and atom counts via MPI (ranks > 1).

## Usage Example

In a binary LJ mixture simulation:

```toml
[system]
# ... your system setup ...

[[type_rdf]]
type_i = 0         # Type A atoms (solvent)
type_j = 0         # Type A–A pair correlation
bins = 150
cutoff = 4.0
interval = 50
output_interval = 1000

[[type_rdf]]
type_i = 0
type_j = 1         # Type A–B cross correlation
bins = 150
cutoff = 4.0
interval = 50
output_interval = 1000
```

After the run, check `data/type_rdf_0_0.txt` and `data/type_rdf_0_1.txt` for the g(r) curves.
