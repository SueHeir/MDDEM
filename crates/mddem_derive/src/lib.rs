//! Proc-macro crate providing `#[derive(AtomData)]` and `#[derive(StageEnum)]`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

/// Classification of supported field types.
enum FieldKind {
    /// `Vec<f64>` — 1 f64 per element
    Scalar,
    /// `Vec<[f64; N]>` — N f64s per element
    Array(usize),
}

/// Classify a field type as Scalar, Array(N), or None if unsupported.
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

/// Check if a field has a given attribute (e.g. `#[forward]`).
fn has_attr(field: &syn::Field, name: &str) -> bool {
    field.attrs.iter().any(|a| a.path().is_ident(name))
}

/// Derive macro for `AtomData` trait.
///
/// Supported field types: `Vec<f64>`, `Vec<[f64; 3]>`, `Vec<[f64; 4]>`.
///
/// Field attributes:
/// - `#[forward]` — include in `pack_forward`/`unpack_forward` (overwrite on unpack)
/// - `#[reverse]` — include in `pack_reverse`/`unpack_reverse` (additive `+=` on unpack)
/// - `#[zero]` — include in `zero()` (fill with zeros)
#[proc_macro_derive(AtomData, attributes(forward, reverse, zero))]
pub fn derive_atom_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(&input, "AtomData derive requires named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input, "AtomData derive only supports structs")
                .to_compile_error()
                .into();
        }
    };

    // Classify all fields
    struct FieldInfo {
        ident: syn::Ident,
        kind: FieldKind,
        is_forward: bool,
        is_reverse: bool,
        is_zero: bool,
    }

    let mut field_infos = Vec::new();
    for field in fields.iter() {
        let ident = field.ident.as_ref().unwrap().clone();
        let Some(kind) = classify_field(&field.ty) else {
            return syn::Error::new_spanned(
                field,
                format!(
                    "AtomData derive: field `{}` must be `Vec<f64>` or `Vec<[f64; N]>`",
                    ident
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

    // --- truncate / swap_remove ---
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

    // --- pack ---
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

    // --- unpack ---
    // Need cumulative offset tracking
    let mut unpack_stmts = Vec::new();
    let mut total_size: usize = 0;
    for f in &field_infos {
        let id = &f.ident;
        let off = total_size;
        match &f.kind {
            FieldKind::Scalar => {
                unpack_stmts.push(quote! { self.#id.push(buf[#off]); });
                total_size += 1;
            }
            FieldKind::Array(n) => {
                let indices: Vec<_> = (0..*n).map(|j| off + j).collect();
                unpack_stmts.push(quote! { self.#id.push([#(buf[#indices]),*]); });
                total_size += n;
            }
        }
    }

    // --- apply_permutation ---
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

    // --- forward comm ---
    let has_forward = field_infos.iter().any(|f| f.is_forward);
    let forward_methods = if has_forward {
        let mut fwd_size: usize = 0;
        let mut pack_fwd_stmts = Vec::new();
        let mut unpack_fwd_stmts = Vec::new();
        for f in &field_infos {
            if !f.is_forward {
                continue;
            }
            let id = &f.ident;
            let off = fwd_size;
            match &f.kind {
                FieldKind::Scalar => {
                    pack_fwd_stmts.push(quote! { buf.push(self.#id[i]); });
                    unpack_fwd_stmts.push(quote! { self.#id[i] = buf[#off]; });
                    fwd_size += 1;
                }
                FieldKind::Array(n) => {
                    pack_fwd_stmts.push(quote! { buf.extend_from_slice(&self.#id[i]); });
                    let indices: Vec<_> = (0..*n).map(|j| off + j).collect();
                    unpack_fwd_stmts.push(quote! { self.#id[i] = [#(buf[#indices]),*]; });
                    fwd_size += n;
                }
            }
        }
        quote! {
            fn pack_forward(&self, i: usize, buf: &mut Vec<f64>) {
                #(#pack_fwd_stmts)*
            }
            fn unpack_forward(&mut self, i: usize, buf: &[f64]) -> usize {
                #(#unpack_fwd_stmts)*
                #fwd_size
            }
            fn forward_comm_size(&self) -> usize { #fwd_size }
        }
    } else {
        quote! {}
    };

    // --- reverse comm ---
    let has_reverse = field_infos.iter().any(|f| f.is_reverse);
    let reverse_methods = if has_reverse {
        let mut rev_size: usize = 0;
        let mut pack_rev_stmts = Vec::new();
        let mut unpack_rev_stmts = Vec::new();
        for f in &field_infos {
            if !f.is_reverse {
                continue;
            }
            let id = &f.ident;
            let off = rev_size;
            match &f.kind {
                FieldKind::Scalar => {
                    pack_rev_stmts.push(quote! { buf.push(self.#id[i]); });
                    unpack_rev_stmts.push(quote! { self.#id[i] += buf[#off]; });
                    rev_size += 1;
                }
                FieldKind::Array(n) => {
                    pack_rev_stmts.push(quote! { buf.extend_from_slice(&self.#id[i]); });
                    let elem_stmts: Vec<_> = (0..*n)
                        .map(|j| {
                            let idx = off + j;
                            quote! { self.#id[i][#j] += buf[#idx]; }
                        })
                        .collect();
                    unpack_rev_stmts.push(quote! { #(#elem_stmts)* });
                    rev_size += n;
                }
            }
        }
        quote! {
            fn pack_reverse(&self, i: usize, buf: &mut Vec<f64>) {
                #(#pack_rev_stmts)*
            }
            fn unpack_reverse(&mut self, i: usize, buf: &[f64]) -> usize {
                #(#unpack_rev_stmts)*
                #rev_size
            }
            fn reverse_comm_size(&self) -> usize { #rev_size }
        }
    } else {
        quote! {}
    };

    // --- zero ---
    let has_zero = field_infos.iter().any(|f| f.is_zero);
    let zero_method = if has_zero {
        let zero_stmts: Vec<_> = field_infos
            .iter()
            .filter(|f| f.is_zero)
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
    } else {
        quote! {}
    };

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

/// Derive macro for `StageName` trait.
///
/// Generates a `stage_name(&self) -> &'static str` method mapping each variant
/// to its `#[stage("name")]` attribute value.
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
#[proc_macro_derive(StageEnum, attributes(stage))]
pub fn derive_stage_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let variants = match &input.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return syn::Error::new_spanned(&input, "StageEnum can only be derived for enums")
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
                format!("StageEnum: variant `{}` is missing #[stage(\"name\")] attribute", ident),
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
                    "StageEnum: #[stage] attribute must contain a string literal, e.g. #[stage(\"name\")]",
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
                    format!("StageEnum: duplicate stage name \"{}\"", a),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let num_stages = stage_names.len();

    let expanded = quote! {
        impl mddem_scheduler::StageName for #name {
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
        }
    };

    expanded.into()
}
