# md_lj

Lennard-Jones 12-6 pair force for [MDDEM](https://github.com/SueHeir/MDDEM): the standard continuous pair potential for molecular dynamics.

## Physics

### LJ 12-6 Pair Force (`LJForcePlugin`)
Standard Lennard-Jones potential with cutoff:
- Force: `f/r = 24*eps/r^2 * (2*(sigma/r)^12 - (sigma/r)^6)`
- Repulsive at `r < sigma`, attractive at `sigma < r < cutoff`
- Full virial stress tensor accumulation (via `VirialStressPlugin` from `mddem_core`)
- Ghost atom scaling (0.5 for cross-boundary pairs)

### Tail Corrections
Long-range corrections computed once at setup from density and cutoff:
- Energy tail: `E_tail = (8/3)*pi*N*rho*eps*sigma^3 * [(1/3)(sigma/rc)^9 - (sigma/rc)^3]`
- Pressure tail: `P_tail = (16/3)*pi*rho^2*eps*sigma^3 * [(2/3)(sigma/rc)^9 - (sigma/rc)^3]`

## Config

### Single-type (backward compatible)
```toml
[lj]
epsilon = 1.0    # well depth (reduced units)
sigma = 1.0      # length scale
cutoff = 2.5     # cutoff distance in sigma units
```

### Multi-type
```toml
[lj]
cutoff = 2.5
mixing = "geometric"     # "geometric" (default) or "arithmetic"

[[lj.types]]
epsilon = 1.0
sigma = 1.0

[[lj.types]]
epsilon = 0.5
sigma = 0.8

# Optional explicit pair overrides
[[lj.pair_coeffs]]
types = [0, 1]
epsilon = 0.75
sigma = 0.9
```

When `types` is present, the plugin builds an NxN `PairCoeffTable<LJPairCoeffs>` with mixed parameters. Explicit `pair_coeffs` entries override the mixed values for specific pairs.

## Resources

- `LJConfig` — deserialized config
- `LJPairTable` — precomputed NxN pair coefficient table (`PairCoeffTable<LJPairCoeffs>`)
- `VirialStress` — full symmetric virial stress tensor (from `mddem_core`, shared with bond/contact forces)
- `LJTailCorrections` — energy and pressure tail corrections (computed once at setup)

## Usage

```rust
use mddem::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins).add_plugins(LJDefaultPlugins);
app.start();
```

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
