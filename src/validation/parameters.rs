use crate::enzyme_ml::{EnzymeMLDocument, Parameter};
use crate::validation::validator::{Report, Severity, ValidationResult};

pub fn check_parameters(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    for (param_idx, parameter) in enzmldoc.parameters.iter().enumerate() {
        check_parameter_units(report, &parameter, param_idx);
    }
}

fn check_parameter_units(report: &mut Report, parameter: &Parameter, param_idx: usize) {
    if parameter.unit.is_none() {
        let result = ValidationResult::new(
            format!("/parameters/{}", param_idx),
            format!(
                "Parameter '{}' has no unit. It is advisable to equip parameters with a unit.",
                parameter.id
            ),
            Severity::Warning,
        );

        report.add_result(result);
    }
}
