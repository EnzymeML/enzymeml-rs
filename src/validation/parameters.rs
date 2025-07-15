use crate::prelude::{EnzymeMLDocument, Parameter};
use crate::validation::consistency::{Report, Severity, ValidationResult};

/// Validates parameters in an EnzymeML document by checking units
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document containing parameters to validate
/// * `report` - Validation report to add any validation warnings to
///
/// # Details
/// For each parameter in the document, checks if it has a unit defined.
/// Adds a warning to the report for any parameters missing units.
pub fn check_parameters(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    for (param_idx, parameter) in enzmldoc.parameters.iter().enumerate() {
        check_parameter_units(report, parameter, param_idx);
    }
}

/// Validates that a parameter has units defined
///
/// # Arguments
/// * `report` - Validation report to add any warnings to
/// * `parameter` - The parameter to validate
/// * `param_idx` - Index of this parameter in the document's parameters list
///
/// # Details
/// Checks if the parameter has a unit defined.
/// Adds a warning to the report if no unit is specified.
fn check_parameter_units(report: &mut Report, parameter: &Parameter, param_idx: usize) {
    if parameter.unit.is_none() {
        let result = ValidationResult::new(
            format!("/parameters/{param_idx}"),
            format!(
                "Parameter '{}' has no unit. It is advisable to equip parameters with a unit.",
                parameter.id
            ),
            Severity::Warning,
        );

        report.add_result(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    /// Test that a parameter with no unit is invalid
    #[test]
    fn test_invalid_parameter_no_unit() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_parameters(
                ParameterBuilder::default()
                    .id("P1".to_string())
                    .symbol("k".to_string())
                    .name("P1".to_string())
                    .build()
                    .expect("Failed to build parameter"),
            )
            .build()
            .expect("Failed to build document");

        check_parameters(&enzmldoc, &mut report);
        assert!(report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }
}
