//! Proc-macro crate providing `#[derive(AtomData)]` and `#[derive(StageEnum)]`.
//!
//! # `AtomData` derive
//!
//! Generates an implementation of the `AtomData` trait for structs whose fields
//! are `Vec<f64>` or `Vec<[f64; N]>`. Each field becomes a per-atom data column
//! that the framework can pack, unpack, permute, and communicate.
//!
//! ## Field attributes
//!
//! | Attribute     | Effect on generated code |
//! |---------------|--------------------------|
//! | `#[forward]`  | Include in `pack_forward` / `unpack_forward` (overwrite on unpack) |
//! | `#[reverse]`  | Include in `pack_reverse` / `unpack_reverse` (additive `+=` on unpack) |
//! | `#[zero]`     | Include in `zero()` — resize and fill with zeros each step |
//!
//! ## Example
//!
//! ```rust,ignore
//! use mddem_derive::AtomData;
//!
//! #[derive(AtomData)]
//! pub struct ThermalAtom {
//!     /// Temperature (K). Sent to ghost atoms via forward communication.
//!     #[forward]
//!     pub temperature: Vec<f64>,
//!
//!     /// Heat flux accumulator (W). Summed from ghosts via reverse communication,
//!     /// then zeroed at the start of each step.
//!     #[reverse]
//!     #[zero]
//!     pub heat_flux: Vec<f64>,
//! }
//! ```
//!
//! The derive macro generates:
//! - `pack` / `unpack` — serialize **all** fields for atom migration
//! - `truncate` / `swap_remove` — resize all field vectors together
//! - `apply_permutation` — reorder all field vectors by a permutation
//! - `pack_forward` / `unpack_forward` — communicate `#[forward]` fields (overwrite)
//! - `pack_reverse` / `unpack_reverse` — communicate `#[reverse]` fields (accumulate)
//! - `zero` — zero out `#[zero]` fields for `n` atoms
//!
//! # `StageEnum` derive
//!
//! Generates an implementation of `sim_scheduler::StageName` for enums where
//! each variant is annotated with `#[stage("name")]`.
//!
//! ```rust,ignore
//! #[derive(Clone, PartialEq, Default, StageEnum)]
//! enum Phase {
//!     #[default]
//!     #[stage("settle")]
//!     Settle,
//!     #[stage("compress")]
//!     Compress,
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

/// Classification of a field's inner element type.
///
/// Used to determine how many `f64` values each element occupies when packing
/// into a flat buffer.
enum FieldKind {
    /// `Vec<f64>` — one `f64` per atom.
    Scalar,
    /// `Vec<[f64; N]>` — `N` `f64`s per atom.
    Array(usize),
}

/// Inspect a field's [`Type`] and return its [`FieldKind`], or `None` if the
/// type is not a supported `Vec<f64>` or `Vec<[f64; N]>`.
fn classify_field(ty: &Type) -> Option<FieldKind> {
    let Type::Path(type_path) = ty else { return None };
    let segments = &type_path.path.segments;
    if segments.len() != 1 || segments[0].ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segments[0].arguments else { return None };
    if args.args.len() != 1 {
        return None;
    }
    let syn::GenericArgument::Type(inner) = &args.args[0] else { return None };

    // Check for plain f64
    if let Type::Path(inner_path) = inner {
        if inner_path.path.is_ident("f64") {
            return Some(FieldKind::Scalar);
        }
    }

    // Check for [f64; N]
    if let Type::Array(arr) = inner {
        if let Type::Path(elem_path) = arr.elem.as_ref() {
            if elem_path.path.is_ident("f64") {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(lit_int),
                    ..
                }) = &arr.len
                {
                    if let Ok(n) = lit_int.base10_parse::<usize>() {
                        return Some(FieldKind::Array(n));
                    }
                }
            }
        }
    }

    None
}

/// Return `true` if `field` carries an attribute with the given `name`
/// (e.g. `#[forward]`).
fn has_attr(field: &syn::Field, name: &str) -> bool {
    field.attrs.iter().any(|a| a.path().is_ident(name))
}

/// Parsed information about a single struct field.
struct FieldInfo {
    /// The field's identifier (e.g. `temperature`).
    ident: syn::Ident,
    /// Whether the inner element is a scalar or fixed-size array.
    kind: FieldKind,
    /// Marked with `#[forward]` — included in forward communication.
    is_forward: bool,
    /// Marked with `#[reverse]` — included in reverse communication.
    is_reverse: bool,
    /// Marked with `#[zero]` — zeroed at the start of each step.
    is_zero: bool,
}

/// Derive macro that implements the `AtomData` trait for a struct.
///
/// Every field must be `Vec<f64>` or `Vec<[f64; N]>` — the macro will emit a
/// compile error otherwise. Fields may be annotated with any combination of:
///
/// - **`#[forward]`** — the field is packed/unpacked during forward
///   communication (ghost atoms receive the owning processor's value via
///   overwrite).
/// - **`#[reverse]`** — the field is packed/unpacked during reverse
///   communication (the owning processor accumulates contributions from ghost
///   atoms via `+=`).
/// - **`#[zero]`** — the field is resized to `n` atoms and filled with zeros
///   in the `zero()` method, which runs at the start of each timestep.
///
/// # Generated methods
///
/// | Method | Includes | Behavior |
/// |--------|----------|----------|
/// | `pack` / `unpack` | all fields | Migration: serialize all per-atom data |
/// | `truncate` / `swap_remove` | all fields | Keep field vecs in sync |
/// | `apply_permutation` | all fields | Reorder by permutation |
/// | `pack_forward` / `unpack_forward` | `#[forward]` fields | Overwrite on unpack |
/// | `pack_reverse` / `unpack_reverse` | `#[reverse]` fields | Additive (`+=`) on unpack |
/// | `zero` | `#[zero]` fields | Resize + fill with 0.0 |
///
/// # Example
///
/// ```rust,ignore
/// #[derive(AtomData)]
/// pub struct DemAtom {
///     #[forward]
///     pub omega: Vec<[f64; 3]>,
///     #[reverse]
///     #[zero]
///     pub torque: Vec<[f64; 3]>,
///     pub radius: Vec<f64>,
/// }
/// ```
///
/// # Panics
///
/// Produces a compile-time error if:
/// - Applied to an enum, union, or tuple struct
/// - Any field is not `Vec<f64>` or `Vec<[f64; N]>`
#[proc_macro_derive(AtomData, attributes(forward, reverse, zero))]
pub fn derive_atom_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    &input,
                    "AtomData can only be derived for structs with named fields \
                     (e.g. `struct Foo { field: Vec<f64> }`), not tuple or unit structs",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &input,
                "AtomData can only be derived for structs, not enums or unions",
            )
            .to_compile_error()
            .into();
        }
    };

    // Parse and classify every field.
    let mut field_infos = Vec::new();
    for field in fields.iter() {
        let ident = field.ident.as_ref().expect("named field has ident").clone();
        let Some(kind) = classify_field(&field.ty) else {
            return syn::Error::new_spanned(
                field,
                format!(
                    "AtomData: field `{ident}` has an unsupported type. \
                     Every field must be `Vec<f64>` or `Vec<[f64; N]>` \
                     (e.g. `Vec<[f64; 3]>`)."
                ),
            )
            .to_compile_error()
            .into();
        };
        field_infos.push(FieldInfo {
            ident,
            kind,
            is_forward: has_attr(field, "forward"),
            is_reverse: has_attr(field, "reverse"),
            is_zero: has_attr(field, "zero"),
        });
    }

    // --- truncate / swap_remove (all fields) ---
    let truncate_stmts: Vec<_> = field_infos
        .iter()
        .map(|f| {
            let id = &f.ident;
            quote! { self.#id.truncate(n); }
        })
        .collect();

    let swap_remove_stmts: Vec<_> = field_infos
        .iter()
        .map(|f| {
            let id = &f.ident;
            quote! { self.#id.swap_remove(i); }
        })
        .collect();

    // --- pack (all fields, sequential into buffer) ---
    let pack_stmts: Vec<_> = field_infos
        .iter()
        .map(|f| {
            let id = &f.ident;
            match &f.kind {
                FieldKind::Scalar => quote! { buf.push(self.#id[i]); },
                FieldKind::Array(_) => quote! { buf.extend_from_slice(&self.#id[i]); },
            }
        })
        .collect();

    // --- unpack (all fields, read at compile-time offsets) ---
    let (unpack_stmts, total_size) = build_unpack_stmts(&field_infos);

    // --- apply_permutation (all fields) ---
    let perm_stmts: Vec<_> = field_infos
        .iter()
        .map(|f| {
            let id = &f.ident;
            match &f.kind {
                FieldKind::Scalar => quote! {
                    {
                        let scratch: Vec<f64> = perm.iter().map(|&p| self.#id[p]).collect();
                        self.#id[..n].copy_from_slice(&scratch);
                    }
                },
                FieldKind::Array(n_val) => {
                    let n_lit = *n_val;
                    quote! {
                        {
                            let scratch: Vec<[f64; #n_lit]> = perm.iter().map(|&p| self.#id[p]).collect();
                            self.#id[..n].copy_from_slice(&scratch);
                        }
                    }
                }
            }
        })
        .collect();

    // --- forward communication (#[forward] fields) ---
    let forward_methods = build_comm_methods(&field_infos, CommDirection::Forward);

    // --- reverse communication (#[reverse] fields) ---
    let reverse_methods = build_comm_methods(&field_infos, CommDirection::Reverse);

    // --- zero (#[zero] fields) ---
    let zero_method = build_zero_method(&field_infos);

    let expanded = quote! {
        impl AtomData for #name {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }

            fn truncate(&mut self, n: usize) {
                #(#truncate_stmts)*
            }

            fn swap_remove(&mut self, i: usize) {
                #(#swap_remove_stmts)*
            }

            fn pack(&self, i: usize, buf: &mut Vec<f64>) {
                #(#pack_stmts)*
            }

            fn unpack(&mut self, buf: &[f64]) -> usize {
                #(#unpack_stmts)*
                #total_size
            }

            fn apply_permutation(&mut self, perm: &[usize], n: usize) {
                #(#perm_stmts)*
            }

            #forward_methods
            #reverse_methods
            #zero_method
        }
    };

    expanded.into()
}

/// Build `unpack` statements for all fields, returning the statements and the
/// total number of `f64` values consumed per atom.
fn build_unpack_stmts(
    fields: &[FieldInfo],
) -> (Vec<proc_macro2::TokenStream>, usize) {
    let mut stmts = Vec::new();
    let mut offset: usize = 0;
    for f in fields {
        let id = &f.ident;
        match &f.kind {
            FieldKind::Scalar => {
                stmts.push(quote! { self.#id.push(buf[#offset]); });
                offset += 1;
            }
            FieldKind::Array(n) => {
                let indices: Vec<_> = (0..*n).map(|j| offset + j).collect();
                stmts.push(quote! { self.#id.push([#(buf[#indices]),*]); });
                offset += n;
            }
        }
    }
    (stmts, offset)
}

/// Direction of ghost-atom communication.
enum CommDirection {
    /// Forward: owner → ghost (overwrite).
    Forward,
    /// Reverse: ghost → owner (accumulate via `+=`).
    Reverse,
}

/// Build `pack_*`, `unpack_*`, and `*_comm_size` methods for either forward
/// or reverse communication.
fn build_comm_methods(
    fields: &[FieldInfo],
    direction: CommDirection,
) -> proc_macro2::TokenStream {
    let is_forward = matches!(direction, CommDirection::Forward);
    let selected: Vec<_> = fields
        .iter()
        .filter(|f| if is_forward { f.is_forward } else { f.is_reverse })
        .collect();

    if selected.is_empty() {
        return quote! {};
    }

    let mut comm_size: usize = 0;
    let mut pack_stmts = Vec::new();
    let mut unpack_stmts = Vec::new();

    for f in &selected {
        let id = &f.ident;
        let off = comm_size;
        match &f.kind {
            FieldKind::Scalar => {
                pack_stmts.push(quote! { buf.push(self.#id[i]); });
                if is_forward {
                    unpack_stmts.push(quote! { self.#id[i] = buf[#off]; });
                } else {
                    unpack_stmts.push(quote! { self.#id[i] += buf[#off]; });
                }
                comm_size += 1;
            }
            FieldKind::Array(n) => {
                pack_stmts.push(quote! { buf.extend_from_slice(&self.#id[i]); });
                if is_forward {
                    let indices: Vec<_> = (0..*n).map(|j| off + j).collect();
                    unpack_stmts.push(quote! { self.#id[i] = [#(buf[#indices]),*]; });
                } else {
                    let elem_stmts: Vec<_> = (0..*n)
                        .map(|j| {
                            let idx = off + j;
                            quote! { self.#id[i][#j] += buf[#idx]; }
                        })
                        .collect();
                    unpack_stmts.push(quote! { #(#elem_stmts)* });
                }
                comm_size += n;
            }
        }
    }

    if is_forward {
        quote! {
            fn pack_forward(&self, i: usize, buf: &mut Vec<f64>) {
                #(#pack_stmts)*
            }
            fn unpack_forward(&mut self, i: usize, buf: &[f64]) -> usize {
                #(#unpack_stmts)*
                #comm_size
            }
            fn forward_comm_size(&self) -> usize { #comm_size }
        }
    } else {
        quote! {
            fn pack_reverse(&self, i: usize, buf: &mut Vec<f64>) {
                #(#pack_stmts)*
            }
            fn unpack_reverse(&mut self, i: usize, buf: &[f64]) -> usize {
                #(#unpack_stmts)*
                #comm_size
            }
            fn reverse_comm_size(&self) -> usize { #comm_size }
        }
    }
}

/// Build the `zero` method for fields marked with `#[zero]`.
fn build_zero_method(fields: &[FieldInfo]) -> proc_macro2::TokenStream {
    let zero_fields: Vec<_> = fields.iter().filter(|f| f.is_zero).collect();

    if zero_fields.is_empty() {
        return quote! {};
    }

    let zero_stmts: Vec<_> = zero_fields
        .iter()
        .map(|f| {
            let id = &f.ident;
            match &f.kind {
                FieldKind::Scalar => quote! {
                    self.#id.resize(n, 0.0);
                    self.#id[..n].fill(0.0);
                },
                FieldKind::Array(n_val) => {
                    let n_lit = *n_val;
                    quote! {
                        self.#id.resize(n, [0.0; #n_lit]);
                        self.#id[..n].fill([0.0; #n_lit]);
                    }
                }
            }
        })
        .collect();

    quote! {
        fn zero(&mut self, n: usize) {
            #(#zero_stmts)*
        }
    }
}

/// Derive macro that implements `sim_scheduler::StageName` for an enum.
///
/// Every variant must carry a `#[stage("name")]` attribute with a unique string
/// literal identifying the stage in TOML configuration.
///
/// # Generated methods
///
/// - `stage_name(&self) -> &'static str` — returns the stage name for the
///   current variant.
/// - `stage_names() -> &'static [&'static str]` — returns a slice of all stage
///   names in declaration order.
/// - `num_stages() -> usize` — returns the number of variants.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Clone, PartialEq, Default, StageEnum)]
/// enum Phase {
///     #[default]
///     #[stage("settle")]
///     Settle,
///     #[stage("compress")]
///     Compress,
/// }
/// ```
///
/// # Panics
///
/// Produces a compile-time error if:
/// - Applied to a struct or union (must be an enum)
/// - Any variant is missing the `#[stage("...")]` attribute
/// - Two variants share the same stage name
#[proc_macro_derive(StageEnum, attributes(stage))]
pub fn derive_stage_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let variants = match &input.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "StageEnum can only be derived for enums, not structs or unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut match_arms = Vec::new();
    let mut stage_names = Vec::new();

    for variant in variants {
        let ident = &variant.ident;

        // Find #[stage("name")] attribute
        let stage_attr = variant.attrs.iter().find(|a| a.path().is_ident("stage"));
        let Some(attr) = stage_attr else {
            return syn::Error::new_spanned(
                variant,
                format!(
                    "StageEnum: variant `{ident}` is missing a #[stage(\"name\")] attribute. \
                     Every variant must specify its TOML stage name, e.g.:\n\n    \
                     #[stage(\"my_stage\")]\n    {ident},"
                ),
            )
            .to_compile_error()
            .into();
        };

        // Parse the string literal from #[stage("name")]
        let stage_name: syn::LitStr = match attr.parse_args() {
            Ok(lit) => lit,
            Err(_) => {
                return syn::Error::new_spanned(
                    attr,
                    "StageEnum: #[stage(...)] expects a string literal, \
                     e.g. #[stage(\"settle\")]",
                )
                .to_compile_error()
                .into();
            }
        };

        let name_str = stage_name.value();
        stage_names.push(name_str.clone());
        match_arms.push(quote! { #name::#ident => #name_str, });
    }

    // Check for duplicate stage names
    for (i, a) in stage_names.iter().enumerate() {
        for b in &stage_names[i + 1..] {
            if a == b {
                return syn::Error::new_spanned(
                    &input,
                    format!(
                        "StageEnum: duplicate stage name \"{a}\". \
                         Each variant must have a unique stage name."
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let num_stages = stage_names.len();

    // Build from_index match arms: 0 => Some(Name::Variant0), 1 => ...
    let variant_idents: Vec<_> = variants.iter().map(|v| &v.ident).collect();
    let indices: Vec<_> = (0..variant_idents.len()).collect();

    let expanded = quote! {
        impl sim_scheduler::StageName for #name {
            fn stage_name(&self) -> &'static str {
                match self {
                    #(#match_arms)*
                }
            }

            fn stage_names() -> &'static [&'static str] {
                &[#(#stage_names),*]
            }

            fn num_stages() -> usize {
                #num_stages
            }

            fn from_index(i: usize) -> Option<Self> {
                match i {
                    #(#indices => Some(#name::#variant_idents),)*
                    _ => None,
                }
            }
        }
    };

    expanded.into()
}

/// Derive macro that implements `sim_scheduler::Schedule` for an enum.
///
/// Every variant must carry a `#[phase(N)]` attribute with a numeric literal
/// specifying the execution order index.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Clone, Copy, Debug, PartialEq, Schedule)]
/// enum CfdSchedule {
///     #[phase(0)]
///     Setup,
///     #[phase(1)]
///     AssembleFluxes,
///     #[phase(2)]
///     SolvePressure,
/// }
/// ```
///
/// # Panics
///
/// Produces a compile-time error if:
/// - Applied to a struct or union (must be an enum)
/// - Any variant is missing the `#[phase(N)]` attribute
/// - Two variants share the same phase index
#[proc_macro_derive(Schedule, attributes(phase))]
pub fn derive_schedule_phase(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let variants = match &input.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "Schedule can only be derived for enums, not structs or unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut match_index_arms = Vec::new();
    let mut match_name_arms = Vec::new();
    let mut phase_indices = Vec::new();

    for variant in variants {
        let ident = &variant.ident;

        // Find #[phase(N)] attribute
        let phase_attr = variant.attrs.iter().find(|a| a.path().is_ident("phase"));
        let Some(attr) = phase_attr else {
            return syn::Error::new_spanned(
                variant,
                format!(
                    "Schedule: variant `{ident}` is missing a #[phase(N)] attribute. \
                     Every variant must specify its ordering index, e.g.:\n\n    \
                     #[phase(0)]\n    {ident},"
                ),
            )
            .to_compile_error()
            .into();
        };

        // Parse the integer literal from #[phase(N)]
        let phase_index: syn::LitInt = match attr.parse_args() {
            Ok(lit) => lit,
            Err(_) => {
                return syn::Error::new_spanned(
                    attr,
                    "Schedule: #[phase(...)] expects an integer literal, \
                     e.g. #[phase(0)]",
                )
                .to_compile_error()
                .into();
            }
        };

        let index_val: u32 = match phase_index.base10_parse() {
            Ok(v) => v,
            Err(_) => {
                return syn::Error::new_spanned(
                    &phase_index,
                    "Schedule: #[phase(...)] must be a valid u32 integer",
                )
                .to_compile_error()
                .into();
            }
        };

        // Check for duplicates
        if phase_indices.contains(&index_val) {
            return syn::Error::new_spanned(
                &phase_index,
                format!(
                    "Schedule: duplicate phase index {index_val}. \
                     Each variant must have a unique phase index."
                ),
            )
            .to_compile_error()
            .into();
        }
        phase_indices.push(index_val);

        let variant_name = ident.to_string();
        match_index_arms.push(quote! { #name::#ident => #index_val, });
        match_name_arms.push(quote! { #name::#ident => #variant_name, });
    }

    let expanded = quote! {
        impl sim_scheduler::Schedule for #name {
            fn to_index(&self) -> u32 {
                match self {
                    #(#match_index_arms)*
                }
            }

            fn name(&self) -> &'static str {
                match self {
                    #(#match_name_arms)*
                }
            }
        }
    };

    expanded.into()
}
