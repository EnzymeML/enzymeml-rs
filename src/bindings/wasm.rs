//! WASM bindings for EnzymeML validation functionality
//!
//! This module provides WebAssembly bindings for validating EnzymeML documents,
//! including JSON schema validation and consistency checking.
//!
//! If you want to generate the WASM bindings, please use wasm-pack to build the project.
//!
//! ```bash
//! wasm-pack build --target web --out-name enzymeml-validator --features wasm --no-default-features
//! ```
//!
//! It is vital to turn off the default features, otherwise the build will fail due to
//! non-WASM compatible dependencies such as [evalexpr-jit].
//!
//! The generated JavaScript bindings can be found in the `pkg` directory.

use std::error::Error;

use jsonschema::validator_for;
use schemars::schema_for;
use serde_json::Value;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use crate::{
    prelude::EnzymeMLDocument,
    validation::validator::{self, Report},
};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

/// Validates an EnzymeML document against its JSON schema
///
/// # Arguments
/// * `content` - JSON string containing the EnzymeML document
///
/// # Returns
/// * `JsValue` - Validation report as a JavaScript value, containing validation status and any errors
#[wasm_bindgen]
pub fn validate_by_schema(content: &str) -> ValidationReport {
    validate_json(content).unwrap()
}

/// Checks the internal consistency of an EnzymeML document
///
/// Verifies that all references between document elements are valid and required fields are present.
///
/// # Arguments
/// * `content` - JSON string containing the EnzymeML document
///
/// # Returns
/// * `JsValue` - Consistency check report as a JavaScript value
#[wasm_bindgen]
pub fn check_consistency(content: &str) -> Report {
    let enzmldoc: EnzymeMLDocument = serde_json::from_str(content).unwrap();
    validator::check_consistency(&enzmldoc)
}

/// Report containing validation results
#[derive(Tsify, serde::Serialize, serde::Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ValidationReport {
    /// Whether the document is valid
    pub valid: bool,
    /// List of validation errors if any
    pub errors: Vec<ValidationError>,
}

/// Individual validation error details
#[derive(Tsify, serde::Serialize, serde::Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ValidationError {
    /// JSON path where the error occurred
    pub location: String,
    /// Description of the validation error
    pub message: String,
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

#[cfg(test)]
pub mod tests {
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    #[wasm_bindgen_test(unsupported=test)]
    fn test_validate_json() {
        let content = include_str!("../../tests/data/enzmldoc.json");
        let report = validate_by_schema(content);
        assert!(report.valid);
        assert_eq!(report.errors.len(), 0);
    }

    #[wasm_bindgen_test(unsupported=test)]
    fn test_validate_json_invalid() {
        let content = include_str!("../../tests/data/enzymeml_inconsistent_invalid.json");
        let report = validate_by_schema(content);
        assert!(!report.valid);
        assert_eq!(report.errors.len(), 2);
    }

    #[wasm_bindgen_test(unsupported=test)]
    fn test_check_consistency() {
        let content = include_str!("../../tests/data/enzmldoc.json");
        let report = check_consistency(content);
        assert!(report.is_valid);
        assert_eq!(report.errors.len(), 3);
    }
}
