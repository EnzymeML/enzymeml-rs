//! ODE System Generation for EnzymeML Documents
//!
//! This module provides functionality to derive ordinary differential equation (ODE) systems
//! from EnzymeML documents. It handles the conversion of reaction kinetics into a system of
//! differential equations that describe the time evolution of species concentrations.
//!
//! The main components include:
//! - Deriving complete ODE systems from EnzymeML documents
//! - Building mappings between species and their corresponding equations
//! - Creating individual ODE equations for each species
//! - Error handling for various ODE generation scenarios

use std::{collections::HashMap, fmt::Display};

use thiserror::Error;

use crate::{
    equation::{create_equation, EnzymeMLDocState},
    prelude::{EnzymeMLDocument, EquationBuilderError, EquationType},
    validation::consistency::{check_consistency, ValidationResult},
};

/// The type of system in the document
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemType {
    /// ODEs only
    ODE,
    /// Kinetic laws only
    KineticLaw,
    /// Both ODEs and kinetic laws
    Both,
}

impl Display for SystemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemType::ODE => write!(f, "ODEs only"),
            SystemType::KineticLaw => write!(f, "Kinetic laws only"),
            SystemType::Both => write!(f, "Both ODEs and kinetic laws"),
        }
    }
}

impl EnzymeMLDocument {
    /// Determine the type of system in the document
    ///
    /// # Returns
    ///
    /// * `SystemType` - The type of system in the document
    pub fn determine_system_type(&self) -> SystemType {
        let has_odes = self
            .equations
            .iter()
            .any(|eq| matches!(eq.equation_type, EquationType::ODE));
        let has_kinetic_laws = self.reactions.iter().any(|rxn| rxn.kinetic_law.is_some());

        if has_odes && has_kinetic_laws {
            SystemType::Both
        } else if has_odes {
            SystemType::ODE
        } else {
            SystemType::KineticLaw
        }
    }

    /// Derive the system from the document
    ///
    /// # Returns
    ///
    /// * `Result<(), ODEError>` - Ok if successful, or an error
    pub fn derive_system(&mut self) -> Result<(), ODEError> {
        let system_type = self.determine_system_type();

        match system_type {
            SystemType::KineticLaw => derive_system(self),
            SystemType::Both => Err(ODEError::SystemTypeNotSupported(system_type)),
            _ => Ok(()),
        }
    }
}

/// Main function to derive the ODE system from an EnzymeML document
///
/// This function performs the following steps:
/// 1. Validates document consistency
/// 2. Builds a mapping of species to their equations
/// 3. Creates ODE equations for each species in the document
///
/// # Arguments
///
/// * `enzmldoc` - A mutable reference to the EnzymeML document
///
/// # Returns
///
/// * `Result<(), ODEError>` - Ok if successful, or an error describing what went wrong
///
/// # Errors
///
/// Returns an error if:
/// - The document is not consistent
/// - Species equations cannot be built
/// - ODE creation fails for any species
pub fn derive_system(enzmldoc: &mut EnzymeMLDocument) -> Result<(), ODEError> {
    // Early return if document is not consistent
    let report = check_consistency(enzmldoc);
    if !report.is_valid {
        return Err(ODEError::DocumentNotConsistent(report.errors));
    }

    // Build species to equation mapping
    let species_to_equation = build_species_equations(enzmldoc)?;

    // Get all species IDs once
    let species_ids = get_species_ids(enzmldoc);

    // Create ODEs for each species
    for (species_id, equations) in species_to_equation {
        create_species_ode(species_id, equations, &species_ids, enzmldoc)?;
    }

    Ok(())
}

/// Builds a mapping of species IDs to their corresponding velocity equations
///
/// For each reaction in the document, this function extracts the kinetic law
/// and associates it with each participating species, accounting for stoichiometry.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document containing reactions and species
///
/// # Returns
///
/// * `Result<HashMap<String, Vec<(f64, String)>>, ODEError>` - A mapping from species IDs to
///   vectors of (stoichiometry, equation) pairs, or an error
///
/// # Errors
///
/// Returns an error if any reaction is reversible (not currently supported)
pub fn build_species_equations(
    enzmldoc: &EnzymeMLDocument,
) -> Result<HashMap<String, Vec<(f64, String)>>, ODEError> {
    let mut species_to_equation: HashMap<String, Vec<(f64, String)>> = HashMap::with_capacity(
        enzmldoc.small_molecules.len() + enzmldoc.complexes.len() + enzmldoc.proteins.len(),
    );

    for reaction in &enzmldoc.reactions {
        if reaction.reversible {
            return Err(ODEError::ReversibleReactionNotSupported);
        }

        if let Some(kinetic_law) = &reaction.kinetic_law {
            let law_equation = &kinetic_law.equation;
            for species_data in &reaction.species {
                species_to_equation
                    .entry(species_data.species_id.clone())
                    .or_default()
                    .push((species_data.stoichiometry, law_equation.clone()));
            }
        }
    }

    Ok(species_to_equation)
}

/// Creates an ODE equation for a single species
///
/// Combines all the kinetic equations affecting a species into a single ODE,
/// accounting for stoichiometry, and adds it to the EnzymeML document.
///
/// # Arguments
///
/// * `species_id` - The ID of the species for which to create the ODE
/// * `equations` - Vector of (stoichiometry, equation) pairs affecting this species
/// * `species_ids` - List of all species IDs in the document
/// * `enzmldoc` - Mutable reference to the EnzymeML document
///
/// # Returns
///
/// * `Result<(), ODEError>` - Ok if successful, or an error
///
/// # Errors
///
/// Returns an error if equation creation or building fails
fn create_species_ode(
    species_id: String,
    equations: Vec<(f64, String)>,
    species_ids: &[String],
    enzmldoc: &mut EnzymeMLDocument,
) -> Result<(), ODEError> {
    let equation_string = equations
        .iter()
        .map(|(stoichiometry, equation)| match *stoichiometry {
            1.0 => equation.clone(),
            -1.0 => format!("-{}", equation),
            _ => format!("({}) * ({})", stoichiometry, equation),
        })
        .collect::<Vec<_>>()
        .join(" + ");

    let equation = create_equation(
        &equation_string,
        species_ids,
        EquationType::ODE,
        EnzymeMLDocState::Document(enzmldoc),
    )
    .map_err(|e| ODEError::EquationCreationFailed(e.to_string()))?
    .species_id(species_id)
    .build()
    .map_err(ODEError::EquationBuildFailed)?;

    enzmldoc.equations.push(equation);

    Ok(())
}

/// Get all species ids from the document
///
/// Collects IDs from small molecules, complexes, and proteins into a single vector.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document
///
/// # Returns
///
/// * `Vec<String>` - Vector containing all species IDs
fn get_species_ids(enzmldoc: &EnzymeMLDocument) -> Vec<String> {
    let capacity =
        enzmldoc.small_molecules.len() + enzmldoc.complexes.len() + enzmldoc.proteins.len();
    let mut species = Vec::with_capacity(capacity);

    species.extend(enzmldoc.small_molecules.iter().map(|s| s.id.clone()));
    species.extend(enzmldoc.complexes.iter().map(|s| s.id.clone()));
    species.extend(enzmldoc.proteins.iter().map(|s| s.id.clone()));

    species
}

/// Errors that can occur during ODE system generation
#[derive(Error, Debug)]
pub enum ODEError {
    /// Error when a referenced species cannot be found in the document
    #[error("Species not found: {0}")]
    SpeciesNotFound(String),

    /// Error when the document fails consistency validation
    #[error("Document is not consistent: {0:?}")]
    DocumentNotConsistent(Vec<ValidationResult>),

    /// Error when a reversible reaction is encountered (not currently supported)
    #[error("Reversible reactions are not supported yet")]
    ReversibleReactionNotSupported,

    /// Error during equation creation
    #[error("Equation creation failed: {0}")]
    EquationCreationFailed(String),

    /// Error during equation building
    #[error("Equation build failed: {0}")]
    EquationBuildFailed(#[from] EquationBuilderError),

    /// Error when the system type is not supported
    #[error("System type not supported: {0}")]
    SystemTypeNotSupported(SystemType),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_enzmldoc;

    #[test]
    fn test_build_species_equations() {
        let enzmldoc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let result = build_species_equations(&enzmldoc);
        assert!(result.is_ok());

        let species_to_equation = result.unwrap();
        assert!(!species_to_equation.is_empty());
    }

    #[test]
    fn test_create_species_ode() {
        let mut enzmldoc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let species_ids = get_species_ids(&enzmldoc);
        let equations = vec![(1.0, "k * S".to_string())];

        let result = create_species_ode("S".to_string(), equations, &species_ids, &mut enzmldoc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_derive_system() {
        let mut enzmldoc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let result = derive_system(&mut enzmldoc);

        if let Err(e) = result {
            panic!("Error: {}", e);
        }
    }
}
