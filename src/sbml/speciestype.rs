use std::fmt::Display;

use sbml::species::Species;
use variantly::Variantly;

use super::error::SBMLError;

pub(crate) const SMALL_MOLECULE_SBO_TERM: &str = "SBO:0000247";
pub(crate) const PROTEIN_SBO_TERM: &str = "SBO:0000252";
pub(crate) const COMPLEX_SBO_TERM: &str = "SBO:0000296";

#[derive(Debug, Clone, Copy, Variantly)]
pub enum SpeciesType {
    SmallMolecule,
    Protein,
    Complex,
}

impl Display for SpeciesType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

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
