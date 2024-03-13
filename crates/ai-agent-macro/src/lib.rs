extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

// #[proc_macro_derive(KeywordString)]
// pub fn print_keyword_derive(input: TokenStream) -> TokenStream {
//     let input = parse_macro_input!(input as DeriveInput);
//
//     let struct_name = &input.ident; // The name of the struct
//
//     // Constructing the display implementation
//     let output = quote! {
//         impl ToKeywordString for #struct_name {
//             fn to_keyword_string() -> String {
//                 let test_struct = #struct_name::default();
//                 let result = serde_json::to_string(&test_struct).unwrap();
//                 result
//                     .replace("\"", "")
//                     .replace("0", "")
//                     .replace(".", "")
//                     .replace(":", "")
//                     .replace(",", ", ")
//                 // format!("{}{{{}}}", stringify!(#struct_name), field_names.join(", "))
//             }
//         }
//     };
//
//     TokenStream::from(output)
// }
//
#[proc_macro_derive(KeywordString)]
pub fn print_keyword_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident; // The name of the struct

    // Process fields to generate a list of field names
    let fields_tokens = if let Data::Struct(data) = input.data {
        match data.fields {
            Fields::Named(fields) => fields
                .named
                .into_iter()
                .map(|f| {
                    let field_name = f.ident.expect("Expected named field").to_string();
                    quote! { format!("{}",#field_name)}
                    // quote! { format!("{}: {{}}", stringify!(#field_name), <_ as ToKeywordString>::to_keyword_string()) }
                })
                .collect::<Vec<_>>(),
            _ => panic!("KeywordString only supports structs with named fields"),
        }
    } else {
        panic!("KeywordString can only be applied to structs");
    };

    // Constructing the display implementation
    let output = quote! {
        impl ToKeywordString for #struct_name {
            fn to_keyword_string() -> String {
                let field_names = vec![#(#fields_tokens),*];
                format!("{{{}}}", field_names.join(", "))
                // format!("{}{{{}}}", stringify!(#struct_name), field_names.join(", "))
            }
        }
    };

    TokenStream::from(output)
}
