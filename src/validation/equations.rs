use crate::enzyme_ml::{EnzymeMLDocument, Equation};
use crate::validation::validator::{get_species_ids, Report, Severity, ValidationResult};

pub fn check_equations(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    for (eq_idx, equation) in enzmldoc.equations.iter().enumerate() {
        check_equation_variables(report, &equation, &all_species, eq_idx);
    }
}

fn check_equation_variables(
    report: &mut Report,
    equation: &Equation,
    all_species: &Vec<&String>,
    eq_idx: usize,
) {
    for (var_idx, var) in equation.variables.iter().enumerate() {
        if !all_species.contains(&&var.id) {
            let result = ValidationResult::new(
                format!("/equations/{}/variables/{}", eq_idx, var_idx),
                format!(
                    "Variable '{}' in equation is not defined in the document.",
                    var.id
                ),
                Severity::Error,
            );

            report.add_result(result);
        }
    }
}
