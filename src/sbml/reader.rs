//! SBML to EnzymeML Document Reader
//!
//! This module provides comprehensive functionality for reading and parsing SBML documents
//! into EnzymeML format, serving as the primary entry point for importing enzymatic data
//! from SBML/OMEX archives. It handles the complete conversion process from low-level SBML
//! structures to high-level EnzymeML entities while preserving experimental data and
//! metadata through annotation processing.
//!
//! ## Module Role in SBML Processing Pipeline
//!
//! This reader module works in conjunction with other SBML processing components:
//! - **Reader** (this module): Converts SBML → EnzymeML, extracting entities and annotations
//! - **Writer** (@writer.rs): Converts EnzymeML → SBML, generating compliant SBML documents  
//! - **Version** (@version.rs): Manages version detection and compatibility across EnzymeML formats
//!
//! ## Conversion Process
//!
//! The reading process follows a structured pipeline:
//!
//! 1. **Archive Processing**: OMEX archives are opened and SBML documents extracted
//! 2. **Entity Extraction**: SBML model components (species, reactions, parameters) are
//!    identified and converted to their EnzymeML equivalents
//! 3. **Annotation Processing**: EnzymeML-specific annotations are parsed and applied to
//!    enhance entities with experimental metadata
//! 4. **Data Integration**: External measurement data from the archive is linked to
//!    model entities through annotation references
//! 5. **Document Assembly**: All components are combined into a complete EnzymeML document
//!
//! ## Supported SBML Elements
//!
//! The reader processes the following SBML model components:
//! - **Compartments** → Vessels (reaction environments with volume and units)
//! - **Species** → Small molecules, proteins, or complexes (based on type annotations)
//! - **Reactions** → Enzymatic reactions (with stoichiometry and kinetic laws)
//! - **Parameters** → Model parameters (global and local with statistical metadata)
//! - **Rules** → Mathematical equations (rate laws and assignments)
//!
//! ## Version Compatibility
//!
//! The reader automatically detects EnzymeML version information through XML namespace
//! analysis and applies appropriate annotation parsing logic. This ensures compatibility
//! with both legacy EnzymeML v1 and current v2 formats while maintaining data integrity
//! during the conversion process.
//!
//! ## Error Handling
//!
//! Conversion errors are propagated through the `SBMLError` type, providing detailed
//! information about parsing failures, missing required elements, or incompatible
//! data structures. The reader employs defensive parsing strategies to handle
//! incomplete or malformed SBML documents gracefully.

use std::path::PathBuf;

use sbml::{
    modref::ModifierSpeciesReference,
    prelude::{
        CombineArchive, Compartment, KineticLaw, LocalParameter, Parameter as SBMLParameter,
        Reaction as SBMLReaction, SpeciesReference,
    },
    rule::{Rule, RuleType},
    species::Species,
    SBMLDocument,
};

use crate::{
    io::save_enzmldoc,
    prelude::{
        Complex, EnzymeMLDocument, Equation, EquationType, ModifierElement, ModifierRole,
        Parameter, Protein, Reaction, ReactionElement, SmallMolecule, UnitDefinition, Vessel,
    },
    sbml::{
        annotations::{ComplexAnnot, DataAnnot, ParameterAnnot, ProteinAnnot, SmallMoleculeAnnot},
        error::SBMLError,
        speciestype::SpeciesType,
        utils::{read_omex_file, read_sbml_file},
        version::EnzymeMLAnnotation,
    },
};

/// Reads an OMEX archive from file and converts it to an EnzymeML document.
///
/// This function serves as the primary entry point for importing OMEX archives containing
/// SBML models with EnzymeML annotations. It handles file I/O, archive validation, and
/// delegates to the main parsing logic.
///
/// # Arguments
/// * `path` - Path to the OMEX archive file
///
/// # Returns
/// * `Result<EnzymeMLDocument, SBMLError>` - Complete EnzymeML document or parsing error
pub fn from_omex(path: &PathBuf) -> Result<EnzymeMLDocument, SBMLError> {
    let archive = read_omex_file(path)?;
    parse_omex(archive)
}

/// Parses an OMEX archive and converts its contents to an EnzymeML document.
///
/// This function orchestrates the complete conversion process from SBML to EnzymeML format.
/// It extracts the SBML document from the archive, converts model entities, processes
/// annotations, and integrates external measurement data to produce a comprehensive
/// EnzymeML document.
///
/// The parsing process follows these steps:
/// 1. Extract SBML document from the OMEX archive
/// 2. Convert basic model structure (entities, reactions, parameters)
/// 3. Process data annotations to extract measurement information
/// 4. Link measurement data with model entities
/// 5. Assemble complete EnzymeML document with all metadata
///
/// # Arguments
/// * `archive` - OMEX combine archive containing SBML and data files
///
/// # Returns
/// * `Result<EnzymeMLDocument, SBMLError>` - Complete EnzymeML document with integrated data
///
/// # Errors
/// Returns `SBMLError` if:
/// - SBML document cannot be extracted from archive
/// - Model structure is invalid or incomplete
/// - Annotation processing fails
/// - Measurement data cannot be linked to model entities
pub fn parse_omex(mut archive: CombineArchive) -> Result<EnzymeMLDocument, SBMLError> {
    let sbml = read_sbml_file(&mut archive)?;

    // Parse the SBML document to an EnzymeML document
    let mut enzmldoc = EnzymeMLDocument::try_from(&sbml)?;

    if let Ok(data_annot) = DataAnnot::try_from(&sbml) {
        enzmldoc.measurements = data_annot.extract_measurements(&sbml, &mut archive)?;
    }

    save_enzmldoc("test.json", &enzmldoc).expect("Save has failed!");

    Ok(enzmldoc)
}

/// Converts an SBML document to an EnzymeML document.
///
/// This implementation performs the core model conversion from SBML format to EnzymeML
/// representation. It systematically processes all model components and creates
/// corresponding EnzymeML entities while preserving structural relationships and
/// experimental metadata through annotation processing.
///
/// The conversion process handles:
/// - **Model Structure**: Compartments, species, reactions, and parameters
/// - **Mathematical Relationships**: Rules and kinetic laws
/// - **Experimental Context**: Annotations containing measurement metadata
/// - **Version Compatibility**: Automatic detection and handling of EnzymeML versions
///
/// # Process Overview
/// 1. Extract vessels from SBML compartments
/// 2. Classify and convert species based on type annotations
/// 3. Process reactions with stoichiometry and kinetic laws
/// 4. Extract global and local parameters with statistical metadata
/// 5. Convert mathematical rules to equations
/// 6. Assemble complete document structure
impl TryFrom<&SBMLDocument> for EnzymeMLDocument {
    type Error = SBMLError;

    fn try_from(document: &SBMLDocument) -> Result<Self, Self::Error> {
        let model = document.model().ok_or(SBMLError::MissingModel)?;

        // Extract vessels from compartments
        let vessels = model
            .list_of_compartments()
            .iter()
            .filter_map(|compartment| Vessel::try_from(compartment.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract different types of species
        let small_molecules = model
            .list_of_species()
            .iter()
            .filter_map(|species| SmallMolecule::try_from(species.as_ref()).ok())
            .collect::<Vec<_>>();

        let proteins = model
            .list_of_species()
            .iter()
            .filter_map(|species| Protein::try_from(species.as_ref()).ok())
            .collect::<Vec<_>>();

        let complexes = model
            .list_of_species()
            .iter()
            .filter_map(|species| Complex::try_from(species.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract reactions
        let reactions = model
            .list_of_reactions()
            .iter()
            .filter_map(|reaction| Reaction::try_from(reaction.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract equations from rate rules
        let rate_rules = model
            .list_of_rate_rules()
            .iter()
            .filter_map(|rate_rule| Equation::try_from(rate_rule.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract assignments from rules
        let assignment_rules = model
            .list_of_assignment_rules()
            .iter()
            .filter_map(|assignment_rule| Equation::try_from(assignment_rule.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract global parameters
        let mut parameters = model
            .list_of_parameters()
            .iter()
            .filter_map(|parameter| Parameter::try_from(parameter.as_ref()).ok())
            .collect::<Vec<_>>();

        // Extract local parameters from reaction kinetic laws
        for reaction in model.list_of_reactions() {
            if let Some(kinetic_law) = reaction.kinetic_law() {
                for parameter in kinetic_law.local_parameters() {
                    parameters.push(Parameter::try_from(parameter.as_ref())?);
                }
            }
        }

        Ok(EnzymeMLDocument {
            name: model.name(),
            vessels,
            small_molecules,
            proteins,
            complexes,
            reactions,
            parameters,
            measurements: vec![], // Will be populated separately
            version: "2.0".to_string(),
            equations: rate_rules.into_iter().chain(assignment_rules).collect(),
            ..Default::default()
        })
    }
}

/// Converts an SBML compartment to an EnzymeML vessel.
///
/// Vessels represent reaction environments with defined volumes and units, providing
/// spatial context for enzymatic reactions. This conversion preserves compartment
/// properties while ensuring compatibility with EnzymeML vessel semantics.
///
/// # SBML Mapping
/// - `compartment.id` → `vessel.id`
/// - `compartment.name` → `vessel.name` (fallback to id if missing)
/// - `compartment.size` → `vessel.volume` (default 1.0 if unspecified)
/// - `compartment.units` → `vessel.unit` (required for EnzymeML)
/// - `compartment.constant` → `vessel.constant` (default true)
impl TryFrom<&Compartment<'_>> for Vessel {
    type Error = SBMLError;

    fn try_from(compartment: &Compartment<'_>) -> Result<Self, Self::Error> {
        let unit = compartment
            .unit_definition()
            .map(|unit| UnitDefinition::try_from(unit.as_ref()))
            .transpose()?
            .ok_or(SBMLError::MissingUnitDefinition)?;

        Ok(Vessel {
            id: compartment.id(),
            name: compartment.name().unwrap_or(compartment.id()),
            volume: compartment.size().unwrap_or(1.0),
            unit,
            constant: compartment.constant().unwrap_or(true),
        })
    }
}

/// Converts an SBML species to an EnzymeML small molecule.
///
/// This conversion is conditional on species type validation through annotation analysis.
/// Only species identified as small molecules are processed, with chemical identifiers
/// and structural information extracted from EnzymeML annotations.
///
/// # Annotation Processing
/// - Extracts chemical identifiers (InChI, InChIKey, SMILES)
/// - Preserves database references and synonymous names
/// - Applies version-specific annotation logic automatically
///
/// # Requirements
/// - Species must be classified as a small molecule in annotations
/// - Chemical identification data should be present in annotations
impl TryFrom<&Species<'_>> for SmallMolecule {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_small_molecule() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = SmallMoleculeAnnot::extract(species, &species.id())
            .unwrap_or_else(|_| Box::new(SmallMoleculeAnnot::default()));

        let mut smallmol = SmallMolecule {
            id: species.id(),
            name: species.name().unwrap_or(species.id().to_string()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            canonical_smiles: None,
            inchi: None,
            inchikey: None,
            synonymous_names: vec![],
            references: vec![],
        };

        annotation.apply(&mut smallmol);

        Ok(smallmol)
    }
}

/// Converts an SBML species to an EnzymeML protein.
///
/// This conversion extracts enzymatic information including sequences, EC numbers,
/// and taxonomic data from EnzymeML annotations. Only species classified as proteins
/// are processed to ensure type safety and data consistency.
///
/// # Annotation Processing
/// - Extracts amino acid sequences and enzymatic classifications
/// - Preserves organism information and taxonomic identifiers
/// - Handles database references for protein identification
///
/// # Requirements
/// - Species must be classified as a protein in annotations
/// - Enzymatic metadata should be present for complete conversion
impl TryFrom<&Species<'_>> for Protein {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_protein() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = ProteinAnnot::extract(species, &species.id())
            .unwrap_or_else(|_| Box::new(ProteinAnnot::default()));

        let mut protein = Protein {
            id: species.id(),
            name: species.name().unwrap_or(species.id()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            sequence: None,
            references: vec![],
            ecnumber: None,
            organism: None,
            organism_tax_id: None,
        };

        annotation.apply(&mut protein);

        Ok(protein)
    }
}

/// Converts an SBML species to an EnzymeML complex.
///
/// This conversion processes multi-component molecular assemblies, extracting
/// participant information and interaction details from EnzymeML annotations.
/// Complex formation and stoichiometry data are preserved during conversion.
///
/// # Annotation Processing
/// - Identifies complex participants and their roles
/// - Preserves assembly information and interaction details
/// - Handles multi-component system relationships
///
/// # Requirements
/// - Species must be classified as a complex in annotations
/// - Participant information should be available for complete conversion
impl TryFrom<&Species<'_>> for Complex {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_complex() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = ComplexAnnot::extract(species, &species.id())
            .unwrap_or_else(|_| Box::new(ComplexAnnot::default()));

        let mut complex = Complex {
            id: species.id(),
            name: species.name().unwrap_or(species.id().to_string()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            participants: vec![],
        };

        annotation.apply(&mut complex);

        Ok(complex)
    }
}

/// Converts an SBML kinetic law to an EnzymeML equation.
///
/// Kinetic laws represent mathematical expressions describing reaction rates.
/// This conversion preserves the mathematical formula while adapting it to
/// EnzymeML equation semantics with proper variable identification.
///
/// # Formula Processing
/// - Preserves original mathematical expression syntax
/// - Classifies equation as rate law type
/// - Maintains parameter and variable relationships
impl TryFrom<&KineticLaw<'_>> for Equation {
    type Error = SBMLError;

    fn try_from(kinetic_law: &KineticLaw<'_>) -> Result<Self, Self::Error> {
        Ok(Equation {
            species_id: "v".to_string(),
            equation: kinetic_law.formula(),
            equation_type: EquationType::RateLaw,
            variables: vec![],
        })
    }
}

/// Converts an SBML reaction to an EnzymeML reaction.
///
/// This conversion processes complete reaction systems including reactants,
/// products, modifiers, and associated kinetic laws. Stoichiometric relationships
/// and catalytic roles are preserved during the conversion process.
///
/// # Component Processing
/// - **Reactants**: Species consumed in the reaction with stoichiometry
/// - **Products**: Species produced in the reaction with stoichiometry  
/// - **Modifiers**: Catalytic species affecting reaction rate (enzymes, cofactors)
/// - **Kinetic Laws**: Mathematical descriptions of reaction rates
///
/// # Relationship Preservation
/// - Maintains stoichiometric coefficients for mass balance
/// - Preserves modifier roles for catalytic analysis
/// - Links kinetic laws to reaction mechanisms
impl TryFrom<&SBMLReaction<'_>> for Reaction {
    type Error = SBMLError;

    fn try_from(reaction: &SBMLReaction<'_>) -> Result<Self, Self::Error> {
        let reactants = reaction
            .reactants()
            .borrow()
            .iter()
            .filter_map(|species_reference| {
                ReactionElement::try_from(species_reference.as_ref()).ok()
            })
            .collect::<Vec<_>>();

        let products = reaction
            .products()
            .borrow()
            .iter()
            .filter_map(|species_reference| {
                ReactionElement::try_from(species_reference.as_ref()).ok()
            })
            .collect::<Vec<_>>();

        let modifiers = reaction
            .modifiers()
            .borrow()
            .iter()
            .filter_map(|modifier_species_reference| {
                ModifierElement::try_from(modifier_species_reference.as_ref()).ok()
            })
            .collect::<Vec<_>>();

        let kinetic_law = if let Some(kinetic_law) = reaction.kinetic_law() {
            Some(Equation::try_from(kinetic_law.as_ref())?)
        } else {
            None
        };

        Ok(Reaction {
            id: reaction.id(),
            name: reaction.name().unwrap_or(reaction.id()),
            reversible: reaction.reversible().unwrap_or(false),
            reactants,
            products,
            modifiers,
            kinetic_law,
        })
    }
}

/// Converts an SBML species reference to an EnzymeML reaction element.
///
/// Species references define the quantitative participation of species in reactions
/// through stoichiometric coefficients. This conversion preserves mass balance
/// information essential for kinetic analysis.
///
/// # Stoichiometry Handling
/// - Preserves exact stoichiometric coefficients
/// - Maintains species identification and linkage
/// - Supports fractional stoichiometry for complex mechanisms
impl TryFrom<&SpeciesReference<'_>> for ReactionElement {
    type Error = SBMLError;

    fn try_from(species_reference: &SpeciesReference<'_>) -> Result<Self, Self::Error> {
        Ok(ReactionElement {
            species_id: species_reference.species(),
            stoichiometry: species_reference.stoichiometry(),
        })
    }
}

/// Converts an SBML modifier species reference to an EnzymeML modifier element.
///
/// Modifiers represent catalytic species that influence reaction rates without
/// being consumed or produced. This conversion classifies modifiers by their
/// catalytic role in the enzymatic system.
///
/// # Role Classification
/// - Default classification as biocatalyst (enzyme)
/// - Support for cofactors and other modifier types
/// - Preservation of catalytic relationship information
impl TryFrom<&ModifierSpeciesReference<'_>> for ModifierElement {
    type Error = SBMLError;

    fn try_from(
        modifier_species_reference: &ModifierSpeciesReference<'_>,
    ) -> Result<Self, Self::Error> {
        Ok(ModifierElement {
            species_id: modifier_species_reference.species(),
            role: ModifierRole::Biocatalyst,
        })
    }
}

/// Converts an SBML global parameter to an EnzymeML parameter.
///
/// Global parameters represent model-wide constants or variables used across
/// multiple reactions or equations. This conversion extracts statistical metadata
/// and optimization constraints from EnzymeML annotations.
///
/// # Annotation Processing
/// - Extracts statistical bounds and uncertainty estimates
/// - Preserves optimization constraints for parameter fitting
/// - Maintains unit definitions and dimensional analysis
///
/// # Statistical Enhancement
/// - Initial values for parameter estimation
/// - Upper and lower bounds for optimization
/// - Standard errors and uncertainty quantification
impl TryFrom<&SBMLParameter<'_>> for Parameter {
    type Error = SBMLError;

    fn try_from(parameter: &SBMLParameter<'_>) -> Result<Self, Self::Error> {
        let annotation = ParameterAnnot::extract(parameter, &parameter.id())
            .unwrap_or_else(|_| Box::new(ParameterAnnot::default()));
        let unit = parameter
            .unit_definition()
            .map(|unit| UnitDefinition::try_from(unit.as_ref()))
            .transpose()?;

        let mut parameter = Parameter {
            id: parameter.id(),
            symbol: parameter.name().unwrap_or(parameter.id()),
            name: parameter.name().unwrap_or(parameter.id()),
            value: parameter.value(),
            unit,
            initial_value: None,
            upper_bound: None,
            lower_bound: None,
            stderr: None,
            constant: parameter.constant(),
        };

        annotation.apply(&mut parameter);

        Ok(parameter)
    }
}

/// Converts an SBML local parameter to an EnzymeML parameter.
///
/// Local parameters are reaction-specific constants defined within kinetic laws.
/// This conversion handles parameter scoping and naming while preserving
/// statistical metadata from annotations.
///
/// # Scoping Considerations
/// - Parameters are specific to individual reactions
/// - Naming fallback logic (name → id if name is empty)
/// - Statistical metadata extraction from annotations
///
/// # Enhanced Metadata
/// - Statistical bounds for parameter optimization
/// - Uncertainty estimates for model confidence
/// - Unit preservation for dimensional consistency
impl TryFrom<&LocalParameter<'_>> for Parameter {
    type Error = SBMLError;

    fn try_from(local_parameter: &LocalParameter<'_>) -> Result<Self, Self::Error> {
        let annotation = ParameterAnnot::extract(local_parameter, &local_parameter.id())
            .unwrap_or_else(|_| Box::new(ParameterAnnot::default()));

        let unit = local_parameter
            .unit_definition()
            .map(|unit| UnitDefinition::try_from(unit.as_ref()))
            .transpose()?;

        // Use name if available and non-empty, otherwise use id
        let name = if !local_parameter.name().is_empty() {
            local_parameter.name()
        } else {
            local_parameter.id()
        };

        let mut parameter = Parameter {
            id: local_parameter.id(),
            symbol: local_parameter.id(),
            name,
            value: local_parameter.value(),
            unit,
            stderr: None,
            constant: local_parameter.constant(),
            ..Default::default()
        };

        annotation.apply(&mut parameter);

        Ok(parameter)
    }
}

/// Converts an SBML rule to an EnzymeML equation.
///
/// Rules define mathematical relationships between model variables, including
/// rate laws and assignment rules. This conversion classifies rule types and
/// preserves mathematical expressions for kinetic analysis.
///
/// # Rule Type Classification
/// - **Rate Rules**: Differential equations describing species dynamics
/// - **Assignment Rules**: Algebraic equations for derived quantities
///
/// # Mathematical Preservation
/// - Maintains original formula syntax and structure
/// - Preserves variable relationships and dependencies
/// - Supports complex mathematical expressions
impl TryFrom<&Rule<'_>> for Equation {
    type Error = SBMLError;

    fn try_from(rate_rule: &Rule<'_>) -> Result<Self, Self::Error> {
        let equation_type = match rate_rule.rule_type().map_err(SBMLError::UnknownRuleType)? {
            RuleType::RateRule => EquationType::Ode,
            RuleType::AssignmentRule => EquationType::Assignment,
        };

        Ok(Equation {
            species_id: rate_rule.variable(),
            equation: rate_rule.formula(),
            equation_type,
            variables: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::sbml::utils::read_omex_file;

    use super::*;

    #[test]
    fn test_parse_omex_v2() {
        let path = PathBuf::from("tests/data/enzymeml_v2.omex");
        let archive = read_omex_file(&path).unwrap();
        let enzmldoc = parse_omex(archive).unwrap();
        insta::assert_debug_snapshot!(enzmldoc);
    }

    #[test]
    fn test_parse_omex_v1() {
        let path = PathBuf::from("tests/data/enzymeml_v1.omex");
        let archive = read_omex_file(&path).unwrap();
        let enzmldoc = parse_omex(archive).unwrap();
        insta::assert_debug_snapshot!(enzmldoc);
    }
}
