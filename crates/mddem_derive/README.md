# mddem_derive

Proc-macro crate providing `#[derive(AtomData)]` and `#[derive(StageEnum)]` for [MDDEM](https://github.com/SueHeir/MDDEM).

## Usage

Derive the `AtomData` trait on any struct whose fields are all `Vec<f64>`:

```rust
use mddem_derive::AtomData;

#[derive(AtomData, Default)]
pub struct DemAtom {
    pub radius: Vec<f64>,
    pub density: Vec<f64>,
}
```

This generates implementations of:
- `as_any` / `as_any_mut` — downcasting from trait object
- `truncate` / `swap_remove` — atom removal
- `pack` / `unpack` — MPI communication (one `f64` per field per atom)
- `apply_permutation` — bin-sort reordering

The generated `pack` size equals the number of fields (e.g., 2 for `DemAtom`).

## Compile-Time Validation

The derive macro checks that every field is `Vec<f64>` at compile time. Non-`Vec<f64>` fields produce a clear error:

```
error: AtomData derive: field `name` must be `Vec<f64>`, got `String`
```

## StageEnum

Derive the `StageName` trait on an enum to map variants to `[[run]]` stage names:

```rust
use mddem_derive::StageEnum;

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

This generates implementations of:
- `stage_name(&self) -> &'static str` — returns the stage name for a variant
- `stage_names() -> &'static [&'static str]` — returns all stage names in variant order
- `num_stages() -> usize` — returns the number of stages

Every variant must have a `#[stage("name")]` attribute. Use with `StageAdvancePlugin` to automatically advance `[[run]]` stages when state transitions occur.

Part of the [MDDEM](https://github.com/SueHeir/MDDEM) workspace.
