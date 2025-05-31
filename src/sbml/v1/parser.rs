//! SBML v1 Parser Module
//!
//! This module provides functionality to parse SBML (Systems Biology Markup Language) documents
//! and convert them to EnzymeML format. It handles the extraction of biological entities such as
//! species, reactions, parameters, and measurements from SBML documents contained within COMBINE archives.
//!
//! # Overview
//!
//! The parser supports conversion of:
//! - SBML species to EnzymeML small molecules, proteins, and complexes
//! - SBML reactions to EnzymeML reactions with kinetic laws
//! - SBML parameters to EnzymeML parameters
//! - SBML compartments to EnzymeML vessels
//! - Measurement data from attached TSV files

use polars::{io::SerReader, prelude::CsvReadOptions, series::Series};
use sbml::{
    modref::ModifierSpeciesReference,
    parameter::Parameter as SBMLParameter,
    prelude::{
        CombineArchive, Compartment, KineticLaw, LocalParameter, Reaction as SBMLReaction,
        SpeciesReference,
    },
    rule::{Rule, RuleType},
    species::Species,
    Annotation, SBMLDocument,
};

use crate::{
    prelude::{
        Complex, DataTypes, EnzymeMLDocument, Equation, EquationType, Measurement, MeasurementData,
        ModifierElement, ModifierRole, Parameter, Protein, Reaction, ReactionElement,
        SmallMolecule, UnitDefinition, Vessel,
    },
    sbml::{error::SBMLError, read::read_sbml_file, speciestype::SpeciesType},
};

use super::schema::{
    ColumnAnnot, ComplexAnnot, DataAnnot, FileAnnot, FormatAnnot, InitConcAnnot, MeasurementAnnot,
    ParameterAnnot, ProteinAnnot, ReactantAnnot,
};

/// Parses an SBML document from a COMBINE archive and converts it to EnzymeML format.
///
/// This is the main entry point for SBML to EnzymeML conversion. It extracts the SBML document
/// from the archive, parses all biological entities, extracts measurement data from attached
/// files, and saves the result as an EnzymeML JSON file.
///
/// # Arguments
///
/// * `archive` - A COMBINE archive containing the SBML document and associated data files
///
/// # Panics
///
/// This function will panic if:
/// - The SBML document cannot be parsed from the archive
/// - Data annotations cannot be extracted from the SBML document
/// - The SBML document cannot be converted to EnzymeML format
/// - Measurements cannot be extracted from the data files
/// - The resulting EnzymeML document cannot be saved
pub fn parse_v1_omex(mut archive: CombineArchive) -> Result<EnzymeMLDocument, SBMLError> {
    let sbml = read_sbml_file(&mut archive).expect("Could not parse SBML document");
    let data_annotations = extract_data_annotations(&sbml)
        .expect("Could not extract data annotations from SBML document");

    // Parse the SBML document to an EnzymeML document
    let mut enzmldoc = EnzymeMLDocument::try_from(&sbml)
        .expect("Could not convert SBML document to EnzymeML format");

    enzmldoc.measurements = extract_measurements(&sbml, &data_annotations, &mut archive)
        .expect("Could not extract measurements from data files");

    Ok(enzmldoc)
}

// ================================================================================================
// DATA EXTRACTION FUNCTIONS
// ================================================================================================

/// Extracts measurement data from SBML annotations and associated data files.
///
/// This function processes measurement annotations found in the SBML document and extracts
/// the corresponding time-series data from TSV files contained in the COMBINE archive.
/// Each measurement contains initial concentrations and time-course data for various species.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing measurement annotations
/// * `data_annotations` - Parsed data annotations describing file mappings
/// * `archive` - The COMBINE archive containing data files
///
/// # Returns
///
/// A vector of `Measurement` objects containing time-series data for each species.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - File or format annotations are missing
/// - Data files cannot be read from the archive
/// - CSV parsing fails
/// - Time column is missing from the data
/// - Unit definitions cannot be resolved
fn extract_measurements(
    sbml: &SBMLDocument,
    data_annotations: &DataAnnot,
    archive: &mut CombineArchive,
) -> Result<Vec<Measurement>, SBMLError> {
    let mut measurements = Vec::new();

    for meas in &data_annotations.measurements.measurement {
        let measurement = process_single_measurement(sbml, meas, data_annotations, archive)?;
        measurements.push(measurement);
    }

    Ok(measurements)
}

/// Processes a single measurement annotation and extracts its data.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `meas` - The measurement annotation to process
/// * `data_annotations` - The complete data annotations structure
/// * `archive` - The COMBINE archive containing data files
///
/// # Returns
///
/// A `Measurement` object containing the processed data.
///
/// # Errors
///
/// Returns `SBMLError` if the measurement cannot be processed.
fn process_single_measurement(
    sbml: &SBMLDocument,
    meas: &MeasurementAnnot,
    data_annotations: &DataAnnot,
    archive: &mut CombineArchive,
) -> Result<Measurement, SBMLError> {
    let (file_annot, format_annot) = extract_meas_annotations(meas, data_annotations)?;
    let df = load_measurement_dataframe(file_annot, archive)?;

    let mut measurement = Measurement {
        id: meas.id.clone(),
        name: meas.name.clone(),
        ..Default::default()
    };

    let (time_data, time_unit) = extract_time_information(sbml, format_annot, &df)?;

    for init_conc in meas.init_concs.iter() {
        let meas_data =
            create_measurement_data(sbml, init_conc, format_annot, &df, &time_data, &time_unit)?;
        measurement.species_data.push(meas_data);
    }

    Ok(measurement)
}

/// Loads and parses a CSV file from the archive into a Polars DataFrame.
///
/// # Arguments
///
/// * `file_annot` - The file annotation containing the file location
/// * `archive` - The COMBINE archive to extract from
///
/// # Returns
///
/// A Polars DataFrame containing the parsed CSV data.
///
/// # Errors
///
/// Returns `SBMLError` if the file cannot be loaded or parsed.
fn load_measurement_dataframe(
    file_annot: &FileAnnot,
    archive: &mut CombineArchive,
) -> Result<polars::prelude::DataFrame, SBMLError> {
    let file_content = get_file_content(file_annot, archive)?;

    let cursor = std::io::Cursor::new(file_content.as_bytes());
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(cursor)
        .finish()?;

    Ok(df)
}

/// Extracts time column data and unit information from the format annotation.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `format_annot` - The format annotation containing column definitions
/// * `df` - The DataFrame containing the measurement data
///
/// # Returns
///
/// A tuple containing the time data vector and time unit definition.
///
/// # Errors
///
/// Returns `SBMLError` if the time column is missing or units cannot be resolved.
fn extract_time_information(
    sbml: &SBMLDocument,
    format_annot: &FormatAnnot,
    df: &polars::prelude::DataFrame,
) -> Result<(Vec<f64>, UnitDefinition), SBMLError> {
    let time_col = format_annot
        .columns
        .iter()
        .find(|col| col.column_type.is_time())
        .ok_or(SBMLError::MissingTimeColumn(format_annot.id.clone()))?;

    let time_col_index = time_col.index;
    let time_data = convert_column_to_vec(&df[time_col_index]);
    let time_unit = extract_unit(sbml, &format_annot.columns[time_col_index].unit)?;

    Ok((time_data, time_unit))
}

/// Creates a MeasurementData object for a single species.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `init_conc` - The initial concentration annotation
/// * `format_annot` - The format annotation containing column definitions
/// * `df` - The DataFrame containing the measurement data
/// * `time_data` - The time data vector
/// * `time_unit` - The time unit definition
///
/// # Returns
///
/// A `MeasurementData` object containing the species data.
///
/// # Errors
///
/// Returns `SBMLError` if units cannot be resolved.
fn create_measurement_data(
    sbml: &SBMLDocument,
    init_conc: &InitConcAnnot,
    format_annot: &FormatAnnot,
    df: &polars::prelude::DataFrame,
    time_data: &[f64],
    time_unit: &UnitDefinition,
) -> Result<MeasurementData, SBMLError> {
    let species_id = init_conc
        .protein
        .as_ref()
        .or(init_conc.reactant.as_ref())
        .ok_or(SBMLError::MissingSpeciesId(format_annot.id.clone()))?;

    let data_type = if let Some(column) = get_column(format_annot, species_id) {
        (&column.column_type).into()
    } else {
        Some(DataTypes::Concentration)
    };

    let initial = init_conc.value;
    let data_unit = if let Some(unit) = &init_conc.unit {
        Some(extract_unit(sbml, unit)?)
    } else {
        None
    };

    let mut meas_data = MeasurementData {
        species_id: species_id.clone(),
        prepared: None,
        initial: Some(initial),
        data_unit,
        time_unit: Some(time_unit.clone()),
        data: vec![],
        time: vec![],
        data_type,
        is_simulated: None,
    };

    // Extract time-series data if available
    if let Some(column) = get_column(format_annot, species_id) {
        let column_data = convert_column_to_vec(&df[column.index]);
        meas_data.data = column_data;
        meas_data.time = time_data.to_vec();
    }

    Ok(meas_data)
}

/// Finds a column annotation for a specific species in the format annotation.
///
/// # Arguments
///
/// * `format_annot` - The format annotation containing column definitions
/// * `column_name` - The species ID to search for
///
/// # Returns
///
/// An optional reference to the matching column annotation.
fn get_column<'a>(format_annot: &'a FormatAnnot, column_name: &str) -> Option<&'a ColumnAnnot> {
    format_annot
        .columns
        .iter()
        .filter(|col| col.species_id.is_some())
        .find(|col| col.species_id.as_ref().unwrap() == column_name)
}

/// Converts a Polars Series column to a vector of f64 values.
///
/// # Arguments
///
/// * `column` - The Polars Series to convert
///
/// # Returns
///
/// A vector of f64 values extracted from the series.
///
/// # Panics
///
/// Panics if the column cannot be converted to f64 or contains null values.
fn convert_column_to_vec(column: &Series) -> Vec<f64> {
    column.f64().unwrap().into_no_null_iter().collect()
}

/// Extracts a unit definition from the SBML document by unit ID.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `unit` - The unit ID to look up
///
/// # Returns
///
/// The corresponding `UnitDefinition` object.
///
/// # Errors
///
/// Returns `SBMLError::MissingUnit` if the unit definition is not found.
fn extract_unit(sbml: &SBMLDocument, unit: &str) -> Result<UnitDefinition, SBMLError> {
    if let Some(model) = sbml.model() {
        if let Some(unit_def) = model.get_unit_definition(unit) {
            return UnitDefinition::try_from(unit_def.as_ref());
        }
    }
    Err(SBMLError::MissingUnit(unit.to_string()))
}

/// Extracts file and format annotations for a specific measurement.
///
/// # Arguments
///
/// * `meas` - The measurement annotation
/// * `data_annotations` - The complete data annotations structure
///
/// # Returns
///
/// A tuple containing references to the file and format annotations.
///
/// # Errors
///
/// Returns `SBMLError` if the referenced file or format annotations are not found.
fn extract_meas_annotations<'a>(
    meas: &'a MeasurementAnnot,
    data_annotations: &'a DataAnnot,
) -> Result<(&'a FileAnnot, &'a FormatAnnot), SBMLError> {
    let file_annot = data_annotations
        .files
        .file
        .iter()
        .find(|file| file.id == meas.file)
        .ok_or(SBMLError::MissingFile(meas.file.clone()))?;

    let format_annot = data_annotations
        .formats
        .format
        .iter()
        .find(|format| format.id == file_annot.format)
        .ok_or(SBMLError::MissingFormat(file_annot.format.clone()))?;

    Ok((file_annot, format_annot))
}

/// Retrieves file content from the COMBINE archive.
///
/// # Arguments
///
/// * `file` - The file annotation containing the file location
/// * `archive` - The COMBINE archive to extract from
///
/// # Returns
///
/// The file content as a string.
///
/// # Errors
///
/// Returns `SBMLError` if the file cannot be found or read from the archive.
fn get_file_content(file: &FileAnnot, archive: &mut CombineArchive) -> Result<String, SBMLError> {
    let file_content = archive
        .entry(&file.location)
        .map_err(SBMLError::CombineArchiveError)?;

    let file_content = file_content
        .as_string()
        .map_err(SBMLError::SBMLReaderError)?;

    Ok(file_content)
}

/// Extracts data annotations from the SBML document's reaction list.
///
/// Data annotations contain metadata about measurement files and their format,
/// typically stored as annotations on the ListOfReactions element.
///
/// # Arguments
///
/// * `sbml` - The SBML document to extract annotations from
///
/// # Returns
///
/// The parsed data annotations structure.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - The SBML model is missing
/// - The annotations cannot be deserialized
fn extract_data_annotations(sbml: &SBMLDocument) -> Result<DataAnnot, SBMLError> {
    let model = sbml.model().ok_or(SBMLError::MissingModel)?;
    let data_annotations = model
        .get_reactions_annotation_serde::<DataAnnot>()
        .map_err(SBMLError::DeserializeError)?;

    Ok(data_annotations)
}

// ================================================================================================
// TYPE CONVERSIONS
// ================================================================================================

/// Converts an SBML document to an EnzymeML document.
///
/// This implementation extracts all biological entities from the SBML model and converts
/// them to their corresponding EnzymeML representations. It processes compartments, species,
/// reactions, parameters, and rules.
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
        let equations = model
            .list_of_rate_rules()
            .iter()
            .filter_map(|rate_rule| Equation::try_from(rate_rule.as_ref()).ok())
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
            version: "v1".to_string(),
            equations,
            ..Default::default()
        })
    }
}

/// Converts an SBML compartment to an EnzymeML vessel.
///
/// Vessels represent reaction compartments with defined volumes and units.
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
/// Only processes species that are identified as small molecules based on their annotations.
impl TryFrom<&Species<'_>> for SmallMolecule {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_small_molecule() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = species
            .get_annotation_serde::<ReactantAnnot>()
            .unwrap_or_default();

        Ok(SmallMolecule {
            id: species.id(),
            name: species.name().unwrap_or(species.id().to_string()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            canonical_smiles: annotation.smiles,
            inchi: annotation.inchi,
            inchikey: None,
            synonymous_names: vec![],
            references: vec![],
        })
    }
}

/// Converts an SBML species to an EnzymeML protein.
///
/// Only processes species that are identified as proteins based on their annotations.
impl TryFrom<&Species<'_>> for Protein {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_protein() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = species
            .get_annotation_serde::<ProteinAnnot>()
            .unwrap_or_default();

        Ok(Protein {
            id: species.id(),
            name: species.name().unwrap_or(species.id()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            sequence: annotation.sequence,
            references: vec![],
            ecnumber: annotation.ecnumber,
            organism: annotation.organism,
            organism_tax_id: annotation.organism_tax_id,
        })
    }
}

/// Converts an SBML species to an EnzymeML complex.
///
/// Only processes species that are identified as complexes based on their annotations.
impl TryFrom<&Species<'_>> for Complex {
    type Error = SBMLError;

    fn try_from(species: &Species<'_>) -> Result<Self, Self::Error> {
        let species_type = SpeciesType::try_from(species)?;
        if species_type.is_not_complex() {
            return Err(SBMLError::InvalidSpeciesType(species_type));
        }

        let annotation = species
            .get_annotation_serde::<ComplexAnnot>()
            .unwrap_or_default();

        Ok(Complex {
            id: species.id(),
            name: species.name().unwrap_or(species.id().to_string()),
            constant: species.constant(),
            vessel_id: species.compartment(),
            participants: annotation.participants,
        })
    }
}

/// Converts an SBML kinetic law to an EnzymeML equation.
///
/// Kinetic laws represent rate equations for reactions.
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
/// Processes reactants, products, modifiers, and kinetic laws.
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
/// Species references define the stoichiometry of reactants and products in reactions.
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
/// Modifiers are species that affect reaction rates but are not consumed or produced.
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
/// Global parameters are model-wide constants or variables.
impl TryFrom<&SBMLParameter<'_>> for Parameter {
    type Error = SBMLError;

    fn try_from(parameter: &SBMLParameter<'_>) -> Result<Self, Self::Error> {
        let annotation = parameter
            .get_annotation_serde::<ParameterAnnot>()
            .unwrap_or_default();

        let unit = parameter
            .unit_definition()
            .map(|unit| UnitDefinition::try_from(unit.as_ref()))
            .transpose()?;

        Ok(Parameter {
            id: parameter.id(),
            symbol: parameter.name().unwrap_or(parameter.id()),
            name: parameter.name().unwrap_or(parameter.id()),
            value: parameter.value(),
            unit,
            initial_value: annotation.initial,
            upper_bound: annotation.upper,
            lower_bound: annotation.lower,
            stderr: None,
            constant: parameter.constant(),
        })
    }
}

/// Converts an SBML local parameter to an EnzymeML parameter.
///
/// Local parameters are specific to individual reactions and their kinetic laws.
impl TryFrom<&LocalParameter<'_>> for Parameter {
    type Error = SBMLError;

    fn try_from(local_parameter: &LocalParameter<'_>) -> Result<Self, Self::Error> {
        let annotation: ParameterAnnot = local_parameter
            .get_annotation_serde::<ParameterAnnot>()
            .unwrap_or_default();

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

        Ok(Parameter {
            id: local_parameter.id(),
            symbol: local_parameter.id(),
            name,
            value: local_parameter.value(),
            unit,
            initial_value: annotation.initial,
            upper_bound: annotation.upper,
            lower_bound: annotation.lower,
            stderr: None,
            constant: local_parameter.constant(),
        })
    }
}

/// Converts an SBML rule to an EnzymeML equation.
///
/// Rules define mathematical relationships between model variables.
impl TryFrom<&Rule<'_>> for Equation {
    type Error = SBMLError;

    fn try_from(rate_rule: &Rule<'_>) -> Result<Self, Self::Error> {
        let equation_type = match rate_rule.rule_type().map_err(SBMLError::UnknownRuleType)? {
            RuleType::RateRule => EquationType::RateLaw,
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
    use super::*;
    use crate::sbml::read::read_omex_file;
    use std::path::PathBuf;

    /// Test the complete SBML to EnzymeML conversion process.
    ///
    /// This test verifies that a sample OMEX file can be successfully parsed
    /// and converted to EnzymeML format without errors.
    #[test]
    fn test_sbml_to_enzymeml() {
        let path = PathBuf::from("tests/data/enzymeml_v1.omex");
        let archive = read_omex_file(&path).unwrap();
        parse_v1_omex(archive).unwrap();
    }
}
