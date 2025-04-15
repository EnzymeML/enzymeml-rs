//! Schema module for validating EnzymeML documents.
//!
//! This module provides functionality to validate EnzymeML documents against
//! the EnzymeML schema. It includes functions to check the schema compliance
//! of EnzymeML documents and to generate schema violations.

use std::{error::Error, fmt};

use colored::Colorize;
use jsonschema::validator_for;
use schemars::schema_for;
use serde_json::Value;
#[cfg(feature = "wasm")]
use tsify_next::Tsify;

use crate::prelude::EnzymeMLDocument;

/// Report containing validation results
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ValidationReport {
    /// Whether the document is valid
    pub valid: bool,
    /// List of validation errors if any
    pub errors: Vec<ValidationError>,
}

/// Individual validation error details
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ValidationError {
    /// JSON path where the error occurred
    pub location: String,
    /// Description of the validation error
    pub message: String,
}

impl fmt::Display for ValidationError {
    /// Formats the validation error for display
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n\t└── {}",
            self.location.bold(),
            self.message.bold().red()
        )
    }
}

/// Validates an EnzymeML document against its JSON schema
///
/// # Arguments
/// * `content` - JSON string containing the EnzymeML document
///
/// # Returns
/// * `Result<ValidationReport, Box<dyn Error>>` - Validation report or error if validation fails
pub fn validate_json(content: &str) -> Result<ValidationReport, Box<dyn Error>> {
    // Parse the JSON content to JSON Value
    let json: Value = serde_json::from_str(content)?;
    let schema = serde_json::to_value(schema_for!(EnzymeMLDocument))?;
    let validator = validator_for(&schema).expect("Error compiling schema");

    if validator.is_valid(&json) {
        Ok(ValidationReport {
            valid: true,
            errors: vec![],
        })
    } else {
        let mut validation_errors = vec![];
        for error in validator.iter_errors(&json) {
            validation_errors.push(ValidationError {
                location: error.instance_path.to_string(),
                message: error.to_string().replace('"', "'"),
            });
        }

        Ok(ValidationReport {
            valid: false,
            errors: validation_errors,
        })
    }
}
