//! Species type classification for SBML species based on SBO terms.
//!
//! This module provides functionality to classify SBML species into different
//! biological entity types (small molecules, proteins, complexes) based on their
//! Systems Biology Ontology (SBO) term annotations.

use std::fmt::Display;

use sbml::species::Species;
use variantly::Variantly;

use super::error::SBMLError;

/// SBO term identifier for small molecules (simple chemical entities).
pub(crate) const SMALL_MOLECULE_SBO_TERM: &str = "SBO:0000247";

/// SBO term identifier for proteins (macromolecular complexes).
pub(crate) const PROTEIN_SBO_TERM: &str = "SBO:0000252";

/// SBO term identifier for complexes (non-covalent molecular complexes).
pub(crate) const COMPLEX_SBO_TERM: &str = "SBO:0000296";

/// Represents the different types of biological species that can be found in SBML models.
///
/// This enum is used to classify species based on their SBO (Systems Biology Ontology) terms,
/// which provide standardized annotations for biological entities. The classification helps
/// determine how to process and convert species to their corresponding EnzymeML representations.
///
/// The `Variantly` derive macro provides additional utility methods for type checking
/// (e.g., `is_small_molecule()`, `is_not_protein()`).
#[derive(Debug, Clone, Copy, Variantly)]
pub enum SpeciesType {
    /// Small molecules or simple chemical entities (SBO:0000247)
    SmallMolecule,
    /// Proteins or polypeptide chains (SBO:0000252)
    Protein,
    /// Molecular complexes formed by non-covalent interactions (SBO:0000296)
    Complex,
}

impl Display for SpeciesType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Converts an SBML species to a species type based on its SBO term annotation.
///
/// This implementation examines the SBO term ID of the species and maps it to the
/// corresponding species type. If the SBO term is not recognized, an error is returned.
///
/// # Errors
///
/// Returns `SBMLError::InvalidSBOTerm` if the species has an unrecognized SBO term
/// that doesn't correspond to any of the supported species types.
impl TryFrom<&Species<'_>> for SpeciesType {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        Ok(match species.sbo_term_id().as_str() {
            SMALL_MOLECULE_SBO_TERM => SpeciesType::SmallMolecule,
            PROTEIN_SBO_TERM => SpeciesType::Protein,
            COMPLEX_SBO_TERM => SpeciesType::Complex,
            _ => return Err(SBMLError::InvalidSBOTerm(species.sbo_term_id())),
        })
    }
}
