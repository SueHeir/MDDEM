# mddem_derive

Procedural macros for deriving `AtomData` trait implementations and stage enumerations in [MDDEM](https://github.com/SueHeir/MDDEM).

## AtomData Derive

Generates serialization, permutation, and communication methods for per-atom data containers. All fields must be `Vec<f64>` or `Vec<[f64; N]>`:

```rust
use mddem_derive::AtomData;

#[derive(AtomData)]
pub struct DemAtom {
    #[forward]
    pub omega: Vec<[f64; 3]>,        // angular velocity
    #[reverse]
    #[zero]
    pub torque: Vec<[f64; 3]>,       // accumulated torque
    pub radius: Vec<f64>,            // particle size
}
```

### Field Attributes

| Attribute | Behavior |
|-----------|----------|
| `#[forward]` | Packed/unpacked during forward communication (overwrite on unpack) |
| `#[reverse]` | Packed/unpacked during reverse communication (accumulate via `+=`) |
| `#[zero]` | Resized and zeroed at the start of each timestep |

### Generated Methods

- `pack` / `unpack` — serialize all fields for atom migration
- `truncate` / `swap_remove` — resize all field vectors together
- `apply_permutation` — reorder vectors by a permutation index
- `pack_forward` / `unpack_forward` — communicate `#[forward]` fields
- `pack_reverse` / `unpack_reverse` — communicate `#[reverse]` fields (additive)
- `zero` — resize and zero `#[zero]` fields for `n` atoms

## StageEnum Derive

Maps enum variants to TOML `[[run]]` stage names via `#[stage("name")]` attributes:

```rust
use mddem_derive::StageEnum;

#[derive(Clone, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("settle")]
    Settle,
    #[stage("compress")]
    Compress,
    #[stage("shear")]
    Shear,
}
```

### Generated Methods

- `stage_name(&self) -> &'static str` — returns the stage name for this variant
- `stage_names() -> &'static [&'static str]` — all stage names in order
- `num_stages() -> usize` — number of variants

Use with `StageAdvancePlugin` to automatically advance `[[run]]` stages when state transitions occur.

## Validation

Both macros validate at compile time:
- `AtomData`: all fields must be `Vec<f64>` or `Vec<[f64; N]>`
- `StageEnum`: all variants must have unique `#[stage("name")]` attributes

Violating constraints produces clear error messages.
