use crate::prelude::{EnzymeMLDocument, Equation, EquationType};
use crate::system::SystemType;
use crate::validation::consistency::{get_species_ids, Report, Severity, ValidationResult};

/// Validates equations in an EnzymeML document by checking that all referenced variables exist
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document containing equations to validate
/// * `report` - Validation report to add any validation errors to
///
/// # Details
/// For each equation in the document, checks that all variables referenced in the equation
/// correspond to species defined in the document. Adds validation errors to the report for
/// any undefined variables.
pub fn check_equations(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    if has_both_system_types(enzmldoc, report) {
        // We can early return here, because there can only be one type of system
        // and we will have already added an error to the report
        return;
    }

    for (eq_idx, equation) in enzmldoc.equations.iter().enumerate() {
        check_equation_variables(report, equation, &all_species, eq_idx);

        if matches!(equation.equation_type, EquationType::Ode) {
            check_ode_species_id(report, equation, &all_species, eq_idx);
        }
    }
}

/// Validates that the document contains only one type of system
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document to validate
/// * `report` - Validation report to add any errors to
///
/// # Returns
/// * `bool` - True if the document contains both ODEs and kinetic laws, false otherwise
pub fn has_both_system_types(enzmldoc: &EnzymeMLDocument, report: &mut Report) -> bool {
    if let SystemType::Both = enzmldoc.determine_system_type() {
        let result = ValidationResult::new(
            "/enzmldoc".to_string(),
            "Document contains both ODEs and kinetic laws. Only one is allowed.".to_string(),
            Severity::Error,
            None,
        );
        report.add_result(result);
        return true;
    }

    false
}

/// Validates the species ID for ODE equations
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `equation` - The equation to validate
/// * `all_species` - List of all species IDs defined in the document
/// * `eq_idx` - Index of this equation in the document's equations list
///
/// # Details
/// Checks if the species ID for ODE equations is defined in the document.
/// If not, adds a validation error to the report.
/// If the species ID is not defined, adds a validation error to the report.
fn check_ode_species_id(
    report: &mut Report,
    equation: &Equation,
    all_species: &[String],
    eq_idx: usize,
) {
    if !all_species.contains(&equation.species_id) {
        let result = ValidationResult::new(
            format!("/equations/{eq_idx}/species_id"),
            format!(
                "Species ID '{}' is not defined in the document.",
                equation.species_id
            ),
            Severity::Error,
            Some(equation.species_id.clone()),
        );

        report.add_result(result);
    }
}
/// Validates variables in a single equation against list of defined species
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `equation` - The equation to validate variables for
/// * `all_species` - List of all species IDs defined in the document
/// * `eq_idx` - Index of this equation in the document's equations list
///
/// # Details
/// Checks each variable in the equation against the list of defined species.
/// If a variable references a species that doesn't exist, adds a validation error
/// to the report with the variable's location and ID.
fn check_equation_variables(
    report: &mut Report,
    equation: &Equation,
    all_species: &[String],
    eq_idx: usize,
) {
    for (var_idx, var) in equation.variables.iter().enumerate() {
        if !all_species.contains(&var.id) {
            let result = ValidationResult::new(
                format!("/equations/{eq_idx}/variables/{var_idx}"),
                format!(
                    "Variable '{}' in equation is not defined in the document.",
                    var.id
                ),
                Severity::Error,
                Some(var.id.clone()),
            );

            report.add_result(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{
        EnzymeMLDocumentBuilder, EquationBuilder, EquationType, SmallMoleculeBuilder,
        VariableBuilder,
    };
    use crate::validation::consistency::Report;

    #[test]
    fn test_valid_equation() {
        let mut report = Report::new();
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
            .to_equations(
                EquationBuilder::default()
                    .species_id("S1".to_string())
                    .equation_type(EquationType::Ode)
                    .equation("S1 * 2".to_string())
                    .to_variables(
                        VariableBuilder::default()
                            .id("S1".to_string())
                            .name("S1".to_string())
                            .symbol("S1".to_string())
                            .build()
                            .expect("Failed to build variable"),
                    )
                    .build()
                    .expect("Failed to build equation"),
            )
            .build()
            .expect("Failed to build document");

        check_equations(&enzmldoc, &mut report);
        if !report.is_valid {
            println!("Report: {report:#?}");
        }

        assert!(report.is_valid);
    }

    #[test]
    fn test_invalid_equation() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test")
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_equations(
                EquationBuilder::default()
                    .species_id("S2".to_string())
                    .equation_type(EquationType::Ode)
                    .equation("S2 * 2".to_string())
                    .to_variables(
                        VariableBuilder::default()
                            .id("S2".to_string())
                            .symbol("S2".to_string())
                            .name("S2".to_string())
                            .build()
                            .expect("Failed to build variable"),
                    )
                    .build()
                    .expect("Failed to build equation"),
            )
            .build()
            .expect("Failed to build document");

        check_equations(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 2);
    }
}
