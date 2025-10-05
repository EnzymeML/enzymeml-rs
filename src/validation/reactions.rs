use crate::prelude::{EnzymeMLDocument, Reaction};
use crate::validation::consistency::{get_species_ids, Report, Severity, ValidationResult};

/// Validates reactions in an EnzymeML document by checking that all referenced species exist
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document containing reactions to validate
/// * `report` - Validation report to add any validation errors to
///
/// # Details
/// For each reaction in the document, checks that all species referenced in the reaction
/// correspond to species defined in the document. Adds validation errors to the report for
/// any undefined species.
pub fn check_reactions(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    for (reaction_idx, reaction) in enzmldoc.reactions.iter().enumerate() {
        check_reaction_species(report, reaction, &all_species, reaction_idx);
        check_reaction_stoichiometry(report, reaction, reaction_idx);
    }
}

/// Validates species in a single reaction against list of defined species
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `reaction` - The reaction to validate species for
/// * `all_species` - List of all species IDs defined in the document
/// * `reaction_idx` - Index of this reaction in the document's reactions list
///
/// # Details
/// Checks each species in the reaction against the list of defined species.
/// If a species referenced in the reaction doesn't exist, adds a validation error
/// to the report with the species' location and ID.
fn check_reaction_species(
    report: &mut Report,
    reaction: &Reaction,
    all_species: &[String],
    reaction_idx: usize,
) {
    let all_elements = reaction.reactants.iter().chain(reaction.products.iter());
    for (elem_idx, reac_elem) in all_elements.enumerate() {
        if !all_species.contains(&reac_elem.species_id) {
            let result = ValidationResult::new(
                format!("/reactions/{reaction_idx}/species/{elem_idx}"),
                format!(
                    "Species '{}' in reaction is not defined in the document.",
                    reac_elem.species_id
                ),
                Severity::Error,
                Some(reaction.id.clone()),
            );

            report.add_result(result);
        }
    }
}

/// Validates that a reaction has at least one reactant and one product
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `reaction` - The reaction to validate
/// * `reaction_idx` - Index of this reaction in the document's reactions list
///
/// # Details
/// Checks that the reaction has at least one reactant and one product.
/// If not, adds a validation error to the report with the reaction's location and ID.
fn check_reaction_stoichiometry(report: &mut Report, reaction: &Reaction, reaction_idx: usize) {
    let has_reactant = !reaction.reactants.is_empty();
    let has_product = !reaction.products.is_empty();

    if !has_reactant || !has_product {
        let result = ValidationResult::new(
            format!("/reactions/{reaction_idx}"),
            format!(
                "Reaction '{}' must have at least one reactant and one product.",
                reaction.id
            ),
            Severity::Error,
            Some(reaction.id.clone()),
        );

        report.add_result(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    /// Tests valid react   ion is valid
    #[test]
    fn test_valid_reaction() {
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
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S2".to_string())
                    .name("S2".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1".to_string())
                    .name("R1".to_string())
                    .reversible(true)
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("S1".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("S2".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .build()
                    .expect("Failed to build reaction"),
            )
            .build()
            .expect("Failed to build document");

        check_reactions(&enzmldoc, &mut report);
        assert!(report.is_valid);
    }

    /// Test that a reaction with undefined species is invalid
    #[test]
    fn test_invalid_reaction_undefined_species() {
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
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1".to_string())
                    .name("R1".to_string())
                    .reversible(true)
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("S2".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("S3".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .build()
                    .expect("Failed to build reaction"),
            )
            .build()
            .expect("Failed to build document");

        check_reactions(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 2);
    }

    /// Test that a reaction with no reactants or products is invalid
    #[test]
    fn test_invalid_reaction_no_reactants_or_products() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1".to_string())
                    .name("R1".to_string())
                    .reversible(true)
                    .build()
                    .expect("Failed to build reaction"),
            )
            .build()
            .expect("Failed to build document");

        check_reactions(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a reaction with no reactants is invalid
    #[test]
    fn test_invalid_reaction_no_reactants() {
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
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1".to_string())
                    .name("R1".to_string())
                    .reversible(true)
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("S1".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .build()
                    .expect("Failed to build reaction"),
            )
            .build()
            .expect("Failed to build document");

        check_reactions(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a reaction with no products is invalid
    #[test]
    fn test_invalid_reaction_no_products() {
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
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1".to_string())
                    .name("R1".to_string())
                    .reversible(true)
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("S1".to_string())
                            .stoichiometry(1.0)
                            .build()
                            .expect("Failed to build species"),
                    )
                    .build()
                    .expect("Failed to build reaction"),
            )
            .build()
            .expect("Failed to build document");

        check_reactions(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }
}
