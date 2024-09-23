use crate::enzyme_ml::{EnzymeMLDocument, Reaction};
use crate::validation::validator::{get_species_ids, Report, Severity, ValidationResult};

pub fn check_reactions(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    for (reaction_idx, reaction) in enzmldoc.reactions.iter().enumerate() {
        check_reaction_species(report, reaction, &all_species, reaction_idx);
    }
}

fn check_reaction_species(
    report: &mut Report,
    reaction: &Reaction,
    all_species: &[&String],
    reaction_idx: usize,
) {
    for (elem_idx, reac_elem) in reaction.species.iter().enumerate() {
        if !all_species.contains(&&reac_elem.species_id) {
            let result = ValidationResult::new(
                format!("/reactions/{}/species/{}", reaction_idx, elem_idx),
                format!(
                    "Species '{}' in reaction is not defined in the document.",
                    reac_elem.species_id
                ),
                Severity::Error,
            );

            report.add_result(result);
        }
    }
}
