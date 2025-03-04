//! Deserialization Macro Module
//!
//! This module provides powerful macros for flexible deserialization of complex data structures.
//!
//! # Key Features
//!
//! - Unwrap lists during deserialization
//! - Unwrap enum variants during deserialization
//! - Flexible handling of optional and default fields
//!
//! # Macro Behavior
//!
//! - Handles default and optional fields
//! - Provides type-safe deserialization
//! - Supports complex nested structures

#[macro_export]
macro_rules! unwrap_list {
    ($struct_name:ident, $field_name:ident, $field_type:ty, $fn_name:ident) => {
        fn $fn_name<'de, D>(deserializer: D) -> Result<Vec<$field_type>, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            #[allow(non_camel_case_types)]
            #[allow(non_snake_case)]
            #[derive(serde::Deserialize)]
            struct $struct_name {
                #[serde(default)]
                $field_name: Vec<$field_type>,
            }

            Ok($struct_name::deserialize(deserializer)?.$field_name)
        }
    };
}

#[macro_export]
macro_rules! unwrap_enum {
    ($struct_name:ident, $field_name:ident, $field_enum:ty, $fn_name:ident) => {
        fn $fn_name<'de, D>(deserializer: D) -> Result<$field_enum, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            #[allow(non_camel_case_types)]
            #[allow(non_snake_case)]
            #[derive(serde::Deserialize)]
            struct $struct_name {
                #[serde(rename = "$value")]
                any_name: $field_enum,
            }

            Ok($struct_name::deserialize(deserializer)?.any_name)
        }
    };
}
