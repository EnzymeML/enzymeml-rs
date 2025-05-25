use ndarray::Array2;
use std::collections::{HashMap, HashSet};

use crate::prelude::Reaction;

use super::error::SimulationError;

/// Derives the stoichiometry matrix from an EnzymeML document.
///
/// The stoichiometry matrix represents the relationship between reactions and species,
/// where each row corresponds to a species and each column corresponds to a reaction.
/// Negative values represent reactants (consumed in the reaction) and positive values
/// represent products (produced by the reaction). The absolute values represent the
/// stoichiometric coefficients.
///
/// # Arguments
///
/// * `reactions` - The reactions to derive the stoichiometry matrix from
///
/// # Returns
///
/// * `Result<Array2<f64>, SimulationError>` - The stoichiometry matrix
pub(crate) fn derive_stoichiometry_matrix(
    reactions: &[Reaction],
) -> Result<(Array2<f64>, Vec<String>), SimulationError> {
    let n_reactions = reactions.len();

    // Early return for empty reactions
    if n_reactions == 0 {
        return Err(SimulationError::NoReactions);
    }

    // Pre-allocate with estimated capacity to avoid reallocations
    let mut species_set = HashSet::with_capacity(n_reactions * 4);

    // Collect all unique species IDs in a single pass
    for reaction in reactions {
        // Add all reactants and products in one pass per reaction
        species_set.extend(reaction.reactants.iter().map(|r| r.species_id.clone()));
        species_set.extend(reaction.products.iter().map(|p| p.species_id.clone()));
    }

    // Convert to sorted vector for consistent indexing
    let mut reaction_species: Vec<String> = species_set.into_iter().collect();
    reaction_species.sort_unstable();

    let n_species = reaction_species.len();

    // Pre-create lookup map for species indices
    let species_indices: HashMap<&String, usize> = reaction_species
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();

    // Initialize matrix with species as rows and reactions as columns
    let mut stoichiometry_matrix = Array2::zeros((n_species, n_reactions));

    // Fill the matrix
    for (j, reaction) in reactions.iter().enumerate() {
        // Process reactants - use negative stoichiometry values
        for reactant in &reaction.reactants {
            if let Some(&i) = species_indices.get(&reactant.species_id) {
                stoichiometry_matrix[(i, j)] = -reactant.stoichiometry;
            }
        }

        // Process products - use positive stoichiometry values
        for product in &reaction.products {
            if let Some(&i) = species_indices.get(&product.species_id) {
                stoichiometry_matrix[(i, j)] = product.stoichiometry;
            }
        }
    }

    Ok((stoichiometry_matrix, reaction_species))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        io::load_enzmldoc,
        prelude::{
            EnzymeMLDocument, EnzymeMLDocumentBuilder, ReactionBuilder, ReactionElementBuilder,
            SmallMoleculeBuilder,
        },
    };
    use ndarray::arr2;

    /// Creates a simple test EnzymeML document with two reactions
    /// A + B -> C
    /// C -> D + E
    fn create_test_document() -> EnzymeMLDocument {
        EnzymeMLDocumentBuilder::default()
            .name("Test Document")
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("A")
                    .name("A")
                    .constant(false)
                    .build()
                    .unwrap(),
            )
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("B")
                    .name("B")
                    .constant(false)
                    .build()
                    .unwrap(),
            )
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("C")
                    .name("C")
                    .constant(false)
                    .build()
                    .unwrap(),
            )
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("D")
                    .name("D")
                    .constant(false)
                    .build()
                    .unwrap(),
            )
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("E")
                    .name("E")
                    .constant(false)
                    .build()
                    .unwrap(),
            )
            .to_reactions(
                ReactionBuilder::default()
                    .id("R1")
                    .name("A + B -> C")
                    .reversible(false)
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("A")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("B")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("C")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap(),
            )
            .to_reactions(
                ReactionBuilder::default()
                    .id("R2")
                    .name("C -> D + E")
                    .reversible(false)
                    .to_reactants(
                        ReactionElementBuilder::default()
                            .species_id("C")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("D")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .to_products(
                        ReactionElementBuilder::default()
                            .species_id("E")
                            .stoichiometry(1.0)
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn test_derive_stoichiometry_matrix() {
        let doc = create_test_document();
        let (matrix, reaction_species) = derive_stoichiometry_matrix(&doc.reactions).unwrap();

        // Check matrix dimensions (5 species x 2 reactions)
        assert_eq!(matrix.shape(), &[5, 2]);
        assert_eq!(reaction_species, vec!["A", "B", "C", "D", "E"]);

        // Expected matrix with species as rows and reactions as columns:
        // Species/Reactions  R1    R2
        // A                [-1.0,  0.0]
        // B                [-1.0,  0.0]
        // C                [ 1.0, -1.0]
        // D                [ 0.0,  1.0]
        // E                [ 0.0,  1.0]
        let expected = arr2(&[
            [-1.0, 0.0],
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_with_non_unity_stoichiometry() {
        // Create a document with non-unity stoichiometry
        // 2A + B -> C
        // C -> 3D
        let mut doc = create_test_document();
        doc.reactions.clear();

        doc.reactions.push(
            ReactionBuilder::default()
                .id("R1")
                .name("2A + B -> C")
                .reversible(false)
                .to_reactants(
                    ReactionElementBuilder::default()
                        .species_id("A")
                        .stoichiometry(2.0)
                        .build()
                        .unwrap(),
                )
                .to_reactants(
                    ReactionElementBuilder::default()
                        .species_id("B")
                        .stoichiometry(1.0)
                        .build()
                        .unwrap(),
                )
                .to_products(
                    ReactionElementBuilder::default()
                        .species_id("C")
                        .stoichiometry(1.0)
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        doc.reactions.push(
            ReactionBuilder::default()
                .id("R2")
                .name("C -> 3D")
                .reversible(false)
                .to_reactants(
                    ReactionElementBuilder::default()
                        .species_id("C")
                        .stoichiometry(1.0)
                        .build()
                        .unwrap(),
                )
                .to_products(
                    ReactionElementBuilder::default()
                        .species_id("D")
                        .stoichiometry(3.0)
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        let (matrix, reaction_species) = derive_stoichiometry_matrix(&doc.reactions).unwrap();

        // We should have 4 species (A, B, C, D) and 2 reactions
        assert_eq!(matrix.shape(), &[4, 2]);
        assert_eq!(reaction_species, vec!["A", "B", "C", "D"]);

        // Expected matrix with species as rows and reactions as columns:
        // Species/Reactions  R1    R2
        // A                [-2.0,  0.0]
        // B                [-1.0,  0.0]
        // C                [ 1.0, -1.0]
        // D                [ 0.0,  3.0]
        let expected = arr2(&[[-2.0, 0.0], [-1.0, 0.0], [1.0, -1.0], [0.0, 3.0]]);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_empty_reactions() {
        let doc = EnzymeMLDocumentBuilder::default()
            .name("Empty Document")
            .build()
            .unwrap();

        let result = derive_stoichiometry_matrix(&doc.reactions);
        assert!(result.is_err());
    }

    #[test]
    fn test_reversible_reaction() {
        // For stoichiometry matrix generation, reversibility doesn't matter
        // We just care about reactants and products
        let mut doc = create_test_document();
        doc.reactions.clear();

        doc.reactions.push(
            ReactionBuilder::default()
                .id("R1")
                .name("A <-> B")
                .reversible(true) // This shouldn't affect the stoichiometry matrix
                .to_reactants(
                    ReactionElementBuilder::default()
                        .species_id("A")
                        .stoichiometry(1.0)
                        .build()
                        .unwrap(),
                )
                .to_products(
                    ReactionElementBuilder::default()
                        .species_id("B")
                        .stoichiometry(1.0)
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap(),
        );

        let (matrix, reaction_species) = derive_stoichiometry_matrix(&doc.reactions).unwrap();

        // Matrix should be 2 species x 1 reaction
        assert_eq!(matrix.shape(), &[2, 1]);
        assert_eq!(reaction_species, vec!["A", "B"]);

        // Expected matrix with species as rows and reactions as columns:
        // Species/Reactions  R1
        // A                [-1.0]
        // B                [ 1.0]
        let expected = arr2(&[[-1.0], [1.0]]);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_from_file() {
        let doc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let (matrix, reaction_species) = derive_stoichiometry_matrix(&doc.reactions).unwrap();

        // 4 species, 2 reactions
        assert_eq!(matrix.shape(), &[4, 2]);
        assert_eq!(
            reaction_species,
            vec!["abts", "abts_radical", "slac", "slac_inactive"]
        );

        // Check the matrix
        // ABTS -> ABTS Radicals
        // Slac -> Slac-Inactive
        let expected = arr2(&[
            [-1.0, 0.0], // ABTS
            [1.0, 0.0],  // ABTS Radicals
            [0.0, -1.0], // Slac
            [0.0, 1.0],  // Slac-Inactive
        ]);

        assert_eq!(matrix, expected);
    }
}
