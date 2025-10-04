//! Consistency module for checking consistency of EnzymeML documents.
//!
//! This module provides functionality to validate EnzymeML documents by checking:
//! - Measurements data consistency
//! - Parameter definitions
//! - Equation validity
//! - Reaction species references
//!
//! The main entry point is the `check_consistency` function which runs all validation
//! checks and returns a `Report` with the results.

use std::fmt;

use crate::extract_all;
use crate::prelude::EnzymeMLDocument;
use crate::validation::equations::check_equations;
use crate::validation::measurements::check_measurements;
use crate::validation::parameters::check_parameters;
use crate::validation::reactions::check_reactions;

use colored::Colorize;
#[cfg(feature = "wasm")]
use tsify_next::Tsify;

/// The `check_consistency` function is used to check the consistency of an `EnzymeMLDocument`.
/// It returns a `Report` containing the results of the checks.
///
/// # Arguments
///
/// * `enzmldoc` - A reference to the `EnzymeMLDocument` to be checked.
///
/// # Returns
///
/// Returns a `Report` containing the results of the consistency checks.
pub fn check_consistency(enzmldoc: &EnzymeMLDocument) -> Report {
    let mut report = Report {
        errors: Vec::new(),
        is_valid: true,
    };

    check_measurements(enzmldoc, &mut report);
    check_parameters(enzmldoc, &mut report);
    check_equations(enzmldoc, &mut report);
    check_reactions(enzmldoc, &mut report);

    report
}

/// The `Report` struct is used to store the results of the validation checks.
///
/// Contains a boolean indicating overall validity and a vector of individual validation results.
/// The document is considered invalid if any validation results have Error severity.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct Report {
    /// Whether the document is valid overall. False if any errors were found.
    pub is_valid: bool,
    /// Vector of individual validation results found during checks.
    pub errors: Vec<ValidationResult>,
}

impl Report {
    /// Creates a new `Report`.
    ///
    /// # Returns
    ///
    /// Returns a new `Report` with the default values.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    /// Adds a validation result to the report.
    ///
    /// # Arguments
    ///
    /// * `result` - The `ValidationResult` to be added.
    ///
    /// If the result has Error severity, marks the overall report as invalid.
    pub fn add_result(&mut self, result: ValidationResult) {
        self.errors.push(result.clone());
        if result.severity == Severity::Error {
            self.is_valid = false;
        }
    }
}

/// The `ValidationResult` struct represents a single validation issue found during checking.
///
/// Contains the location where the issue was found, a descriptive message, and the severity level.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ValidationResult {
    /// JSON pointer path to the location of the validation issue
    location: String,
    /// Human readable description of the validation issue
    message: String,
    /// Severity level of the validation issue
    severity: Severity,
    /// The identifier of the object, if any
    identifier: Option<String>,
}

impl ValidationResult {
    /// Creates a new `ValidationResult`.
    ///
    /// # Arguments
    ///
    /// * `location` - The location of the validation issue as a JSON pointer path.
    /// * `message` - A message describing the validation issue.
    /// * `severity` - The severity of the validation issue.
    ///
    /// # Returns
    ///
    /// Returns a new `ValidationResult` with the provided details.
    pub fn new(
        location: String,
        message: String,
        severity: Severity,
        identifier: Option<String>,
    ) -> Self {
        Self {
            location,
            message,
            severity,
            identifier,
        }
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self.severity {
            Severity::Error => self.message.bold().red(),
            Severity::Warning => self.message.bold().yellow(),
            Severity::Info => self.message.bold().green(),
        };

        let severity = match self.severity {
            Severity::Error => "Error".bold().red(),
            Severity::Warning => "Warning".bold().yellow(),
            Severity::Info => "Info".bold().green(),
        };

        write!(
            f,
            "[{}] {}:\n\t└── {}",
            self.location.bold(),
            severity,
            message
        )
    }
}

/// Severity levels for validation issues.
///
/// Used to indicate how serious a validation issue is:
/// - Error: The document is invalid and should not be used
/// - Warning: The document may have issues but is still valid
/// - Info: Informational message about potential improvements
#[derive(Debug, Clone, PartialEq, Copy, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Severity {
    /// Critical issue that makes the document invalid
    Error,
    /// Non-critical issue that should be reviewed
    Warning,
    /// Informational message about potential improvements
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "Error"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Info => write!(f, "Info"),
        }
    }
}

/// Retrieves the species IDs from an `EnzymeMLDocument`.
///
/// # Arguments
///
/// * `enzmldoc` - A reference to the `EnzymeMLDocument`.
///
/// # Returns
///
/// Returns a vector of references to species IDs, including:
/// - Small molecules
/// - Proteins  
/// - Complexes
pub fn get_species_ids(enzmldoc: &EnzymeMLDocument) -> Vec<String> {
    let small_mols = extract_all!(enzmldoc, small_molecules[*].id);
    let proteins = extract_all!(enzmldoc, proteins[*].id);
    let complexes = extract_all!(enzmldoc, complexes[*].id);

    small_mols
        .into_iter()
        .chain(proteins)
        .chain(complexes)
        .map(|id| id.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_valid_enzmldoc() {
        let path = PathBuf::from("tests/data/enzmldoc.json");
        let enzmldoc = load_enzmldoc(&path).expect("Failed to load document");
        let report = check_consistency(&enzmldoc);
        assert!(report.is_valid);
    }

    #[test]
    fn test_inconsistent_enzmldoc() {
        let path = PathBuf::from("tests/data/enzymeml_inconsistent.json");
        let enzmldoc = load_enzmldoc(&path).expect("Failed to load document");
        let report = check_consistency(&enzmldoc);
        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_invalid_enzmldoc() {
        let path = PathBuf::from("tests/data/enzymeml_inconsistent_invalid.json");
        load_enzmldoc(&path).expect("Failed to load document");
    }

    #[test]
    fn test_get_species_ids() {
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_complexes(
                ComplexBuilder::default()
                    .id("C1".to_string())
                    .name("C1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build complex"),
            )
            .to_proteins(
                ProteinBuilder::default()
                    .id("P1".to_string())
                    .name("P1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build protein"),
            )
            .build()
            .expect("Failed to build document");
        let species_ids = get_species_ids(&enzmldoc);
        assert_eq!(species_ids.len(), 3);
    }
}
