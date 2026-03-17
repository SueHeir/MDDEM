# Brazilian Disk Tensile Test Benchmark

## Physics

The Brazilian disk test (indirect tensile test) compresses a cylindrical disk
between two flat platens.  The loading induces a nearly uniform tensile stress
along the vertical diametral plane, causing failure by a vertical crack through
the center.

The analytical tensile strength from the peak compressive load *P* is:

$$\sigma_t = \frac{2P}{\pi D t}$$

where *D* is the disk diameter and *t* is the thickness.

## Model

| Parameter | Value |
|-----------|-------|
| Disk radius | 10 mm |
| Particle radius | 0.5 mm |
| Particle density | 2500 kg/m³ |
| Young's modulus | 50 MPa |
| Bond normal stiffness | 10⁸ N/m |
| Bond break strain | 0.5% |
| Platen speed | 5 mm/s (each) |

The simulation is **quasi-2D** (one particle layer in the y-direction) with
~250 particles arranged on a hexagonal close-packed lattice.

### Two-stage approach

1. **Packing** — FIRE energy minimization relaxes the hex lattice so all
   particles reach equilibrium before bonds are created.
2. **Loading** — Flat platens move inward at constant velocity, compressing
   the bonded disk until tensile failure.

## Running

```bash
cargo run --release --example bench_brazilian_disk --no-default-features \
    -- examples/bench_brazilian_disk/config.toml
```

Runtime: ~1–3 minutes in release mode.

## Validation

```bash
python3 examples/bench_brazilian_disk/validate.py
```

Checks:
1. No NaN/Inf in output data
2. Load builds up (elastic phase exists)
3. Brittle failure: post-peak load drops below 50% of peak
4. Bonds break during the test
5. Tensile strength falls within physically reasonable range (10²–10⁸ Pa)

## Plots

```bash
python3 examples/bench_brazilian_disk/plot.py
```

Generates:
- `load_displacement.png` — Load vs platen displacement with peak annotation
- `load_bond_breakage.png` — Load and cumulative bond breakage vs time

![Load-Displacement Curve](load_displacement.png)
![Bond Breakage](load_bond_breakage.png)
