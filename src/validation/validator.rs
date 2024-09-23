use crate::enzyme_ml::EnzymeMLDocument;
use crate::extract_all;
use crate::validation::equations::check_equations;
use crate::validation::measurements::check_measurements;
use crate::validation::parameters::check_parameters;
use crate::validation::reactions::check_reactions;

/// The `check_consistency` function is used to check the consistency of an `EnzymeMLDocument`.
/// It returns a `Report` containing the results of the checks.
pub fn check_consistency(enzmldoc: &EnzymeMLDocument) -> Report {
    let mut report = Report {
        errors: Vec::new(),
        is_valid: true,
    };

    check_measurements(&enzmldoc, &mut report);
    check_parameters(&enzmldoc, &mut report);
    check_equations(&enzmldoc, &mut report);
    check_reactions(&enzmldoc, &mut report);

    report
}

/// The `Report` struct is used to store the results of the validation checks.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Report {
    is_valid: bool,
    errors: Vec<ValidationResult>,
}

impl Report {
    pub fn add_result(&mut self, result: ValidationResult) {
        self.errors.push(result.clone());
        if result.severity == Severity::Error {
            self.is_valid = false;
        }
    }
}

/// The `ValidationResult` struct is used to store the result
/// of a single validation check.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationResult {
    location: String,
    message: String,
    severity: Severity,
}

impl ValidationResult {
    pub fn new(location: String, message: String, severity: Severity) -> Self {
        Self {
            location,
            message,
            severity,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

pub fn get_species_ids(enzmldoc: &EnzymeMLDocument) -> Vec<&String> {
    let small_mols = extract_all!(enzmldoc, small_molecules[*].id);
    let proteins = extract_all!(enzmldoc, proteins[*].id);
    let complexes = extract_all!(enzmldoc, complexes[*].id);

    small_mols.chain(proteins).chain(complexes).collect()
}
