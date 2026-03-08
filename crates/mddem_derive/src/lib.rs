//! Proc-macro crate providing `#[derive(AtomData)]` for per-atom extension structs.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Fields, Type};

/// Derive macro for `AtomData` trait.
///
/// All fields must be `Vec<f64>`. Generates implementations of:
/// - `as_any` / `as_any_mut`
/// - `truncate` / `swap_remove`
/// - `pack` / `unpack`
/// - `apply_permutation`
#[proc_macro_derive(AtomData)]
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

    // Validate all fields are Vec<f64>
    for field in fields.iter() {
        if !is_vec_f64(&field.ty) {
            let field_name = field.ident.as_ref().unwrap();
            return syn::Error::new_spanned(
                field,
                format!(
                    "AtomData derive: field `{}` must be `Vec<f64>`, got `{}`",
                    field_name,
                    quote!(#field).to_string()
                ),
            )
            .to_compile_error()
            .into();
        }
    }

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_count = field_names.len();

    let truncate_stmts = field_names.iter().map(|f| quote! { self.#f.truncate(n); });
    let swap_remove_stmts = field_names.iter().map(|f| quote! { self.#f.swap_remove(i); });
    let pack_stmts = field_names.iter().map(|f| quote! { buf.push(self.#f[i]); });
    let unpack_stmts = field_names
        .iter()
        .enumerate()
        .map(|(idx, f)| quote! { self.#f.push(buf[#idx]); });
    let perm_stmts = field_names.iter().map(|f| {
        quote! {
            {
                let scratch: Vec<f64> = perm.iter().map(|&p| self.#f[p]).collect();
                self.#f[..n].copy_from_slice(&scratch);
            }
        }
    });

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
                #field_count
            }

            fn apply_permutation(&mut self, perm: &[usize], n: usize) {
                #(#perm_stmts)*
            }
        }
    };

    expanded.into()
}

fn is_vec_f64(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Vec" {
            if let syn::PathArguments::AngleBracketed(args) = &segments[0].arguments {
                if args.args.len() == 1 {
                    if let syn::GenericArgument::Type(Type::Path(inner)) = &args.args[0] {
                        return inner.path.is_ident("f64");
                    }
                }
            }
        }
    }
    false
}
