# BPM cantilever test — `[[pin]]` hard constraint

A 10-sphere bonded chain anchored at one end by `[[pin]]` bends under its
own weight and settles to the uniform-load cantilever deflection predicted
by beam theory. This exercises the exact failure mode the user's team hit:
a pinned anchor whose neighbour is loaded transversely — bond shear
produces a lever-arm torque that (with wrong sign conventions) created
explosive feedback.

Fixes that make this example pass:

1. **`[[pin]]`** — a true hard constraint. Captures position at setup,
   restores `pos, vel, force` (on `Atom`) AND `omega, torque, ang_mom`
   (on `DemAtom`) every step at `PreInitialIntegration` and `PostForce`.
   Without rotation zeroing, bond shear on the anchor's neighbour creates
   a lever-arm torque that rotates the "pinned" atom, which creates more
   shear velocity → positive feedback.
2. **Bond force sign correction** — shear, twist, and bending moments
   were inverted in the applied sign. Fixed to match the Fortran BPM
   reference: `F_t = K_t·Δs + γ·v_t` applied as `+F_t` on atom *i*
   (lower tag) and `−F_t` on atom *j*. Same convention for `M_tor` and
   `M_bend`. This damps relative motion instead of amplifying it.
3. **Extended `ghost_cutoff`** (from the MPI fix) — unrelated to the pin
   blowup but still needed for bonds that span rank boundaries.

## Setup

- 10 glass-like spheres, radius 1 mm, spaced 2 mm on the x-axis
- Left end (tag 0) at `x = 0` in the `"anchor"` group, pinned via `[[pin]]`
- `E = 1 GPa`, `G = 400 MPa`, bond radius = particle radius
- Critical damping (`β = 1.0`) on all four bond channels
- Gravity `g = −9.81 m/s²` in `z`
- `dt = 1 × 10⁻⁷ s`, 100 000 steps (10 ms) — enough to settle

## Run

```bash
cargo run --release --example bond_cantilever --no-default-features -- \
    examples/bond_cantilever/config.toml
```

Every 1 000 steps the recorder prints:

```
  step  100000  tip_z=-1.007e-6  max_strain=1.12e-11  bonds=9  missing=0  OK
```

## Verified result

After 100 000 steps (10 ms):

| quantity                | measured        | beam theory (`δ = q·L⁴ / (8·E·I)`) |
|-------------------------|-----------------|------------------------------------|
| tip deflection `z_9`    | `-1.01 × 10⁻⁶ m` | `-9.56 × 10⁻⁷ m`                  |
| bond count (per step)   | 9               | 9 (10 atoms, 9 bonds)              |
| missing-partner skips   | 0               | 0                                  |
| max bond strain         | `1.12 × 10⁻¹¹`  | (bending, not stretch)             |

Agreement with beam theory is within ~5 %. No bond breaks, no partner
skips, no exponential blowup.
