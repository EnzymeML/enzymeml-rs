//! Extraction Macro Module
//!
//! This module provides a powerful macro `extract_all` for flexible field extraction from structs.
//!
//! # Key Features
//!
//! - Extract single fields from structs
//! - Access nested fields through dot notation
//! - Extract elements from vectors using wildcard `[*]` or specific indices
//!
//! # Macro Behavior
//!
//! The macro supports multiple extraction patterns:
//! - Simple field access
//! - Nested field traversal
//! - Vector element extraction
//! - Recursive nested vector element extraction

#[macro_export]
/// `extract_all` is a macro that allows for the extraction of nested fields from a struct.
/// It supports accessing single fields, nested fields, and elements within vectors.
macro_rules! extract_all {
    // Base case for accessing a single field
    // $struct: The struct from which to extract the field
    // $field: The field to extract from the struct
    ($struct:expr, $field:ident) => {
        std::iter::once(&$struct.$field)
    };

    // Case where the next part of the path is a vector with wildcard
    // $struct: The struct from which to extract the field
    // $vec_path: The vector from which to extract all elements
    // $($rest:tt)*: The rest of the path to follow after extracting the elements
    ($struct:expr, $vec_path:ident [*] ) => {
        $struct.$vec_path.unwrap_or_default().iter()
    };

    // Case where the next part of the path is a vector with wildcard
    // $struct: The struct from which to extract the field
    // $vec_path: The vector from which to extract all elements
    // $($rest:tt)*: The rest of the path to follow after extracting the elements
    ($struct:expr, $vec_path:ident [*] . $($rest:tt)*) => {
        $struct.$vec_path.iter().flat_map(|item| extract_all!(item, $($rest)*))
    };

    // Case where the next part of the path is a vector with an explicit index
    // $struct: The struct from which to extract the field
    // $vec_path: The vector from which to extract the element
    // $idx: The index of the element to extract from the vector
    // $($rest:tt)*: The rest of the path to follow after extracting the element
    ($struct:expr, $vec_path:ident [$idx:expr] . $($rest:tt)*) => {
        extract_all!($struct.$vec_path[$idx], $($rest)*)
    };

    // General case for accessing nested fields
    // $struct: The struct from which to extract the field
    // $field: The field to extract from the struct
    // $($rest:tt)*: The rest of the path to follow after extracting the field
    ($struct:expr, $field:ident . $($rest:tt)*) => {
        extract_all!($struct.$field, $($rest)*)
    };
}
