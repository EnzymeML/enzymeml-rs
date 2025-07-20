//! SBML v1 Data Extraction Module
//!
//! This module provides functionality to extract measurement data from SBML (Systems Biology Markup Language)
//! documents with EnzymeML v1 annotations. It processes data annotations found in SBML documents and extracts
//! corresponding time-series measurement data from TSV files contained within COMBINE archives.
//!
//! # Overview
//!
//! The extraction process handles:
//! - Parsing data annotations from SBML reaction elements
//! - Loading measurement data from TSV files in COMBINE archives
//! - Converting time-series data to EnzymeML measurement format
//! - Resolving unit definitions and species mappings
//! - Processing initial concentrations and time-course data
//!
//! # Data Structure
//!
//! The v1 annotation format organizes data through:
//! - `DataAnnot`: Root annotation containing files, formats, and measurements
//! - `FileAnnot`: References to data files within the archive
//! - `FormatAnnot`: Column definitions and data structure descriptions
//! - `MeasurementAnnot`: Individual measurement metadata and species mappings
//! - `InitConcAnnot`: Initial concentration values for species

use polars::{io::SerReader, prelude::CsvReadOptions};
use sbml::{prelude::CombineArchive, SBMLDocument};

use crate::{
    prelude::{DataTypes, Measurement, MeasurementData, UnitDefinition},
    sbml::{error::SBMLError, utils::convert_column_to_vec},
};

use super::schema::{
    ColumnAnnot, DataAnnot, FileAnnot, FormatAnnot, InitConcAnnot, MeasurementAnnot,
};

impl TryFrom<&SBMLDocument> for DataAnnot {
    type Error = SBMLError;

    /// Extracts data annotations from an SBML document's reaction elements.
    ///
    /// In EnzymeML v1, data annotations are stored as XML annotations within SBML reaction
    /// elements. This method deserializes these annotations into structured data that can
    /// be used to locate and process measurement files.
    ///
    /// # Arguments
    ///
    /// * `sbml` - The SBML document containing EnzymeML v1 data annotations
    ///
    /// # Returns
    ///
    /// A `DataAnnot` structure containing all file, format, and measurement definitions.
    ///
    /// # Errors
    ///
    /// Returns `SBMLError` if:
    /// - The SBML document lacks a model element
    /// - Data annotations cannot be deserialized from reaction elements
    fn try_from(sbml: &SBMLDocument) -> Result<Self, Self::Error> {
        let model = sbml.model().ok_or(SBMLError::MissingModel)?;
        let data_annotations = model
            .get_reactions_annotation_serde::<DataAnnot>()
            .map_err(SBMLError::DeserializeError)?;
        Ok(data_annotations)
    }
}

/// Extracts all measurement data from SBML v1 annotations and associated data files.
///
/// This is the main entry point for processing EnzymeML v1 measurement data. It iterates
/// through all measurement annotations, loads the corresponding data files from the COMBINE
/// archive, and converts them into EnzymeML measurement objects with proper time-series data.
///
/// Each measurement typically contains:
/// - Initial concentrations for multiple species
/// - Time-course data showing concentration changes over time
/// - Unit definitions for both time and concentration measurements
/// - Metadata such as measurement ID and name
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions and model structure
/// * `data_annotations` - Parsed v1 data annotations describing file mappings and formats
/// * `archive` - The COMBINE archive containing TSV data files
///
/// # Returns
///
/// A vector of `Measurement` objects, each containing time-series data for multiple species.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - Referenced data files cannot be found in the archive
/// - TSV files cannot be parsed or have invalid format
/// - Unit definitions referenced in annotations are missing from the SBML model
/// - Time column data is missing or malformed
pub(crate) fn extract_measurements(
    sbml: &SBMLDocument,
    annot: &DataAnnot,
    archive: &mut CombineArchive,
) -> Result<Vec<Measurement>, SBMLError> {
    let mut measurements = Vec::new();

    for meas in &annot.measurements.measurement {
        let measurement = process_single_measurement(sbml, meas, annot, archive)?;
        measurements.push(measurement);
    }

    Ok(measurements)
}

/// Processes a single measurement annotation to extract its complete dataset.
///
/// This function handles the complete workflow for a single measurement:
/// 1. Resolves file and format annotations
/// 2. Loads the corresponding TSV data file
/// 3. Extracts time column information
/// 4. Processes each species' initial concentration and time-series data
/// 5. Assembles everything into a complete Measurement object
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `meas` - The specific measurement annotation to process
/// * `data_annotations` - The complete data annotations structure for reference lookups
/// * `archive` - The COMBINE archive containing data files
///
/// # Returns
///
/// A complete `Measurement` object with all species data and metadata.
///
/// # Errors
///
/// Returns `SBMLError` if any step in the processing pipeline fails.
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

/// Loads a TSV data file from the COMBINE archive into a Polars DataFrame.
///
/// This function extracts the specified file from the archive and parses it as a CSV/TSV
/// file without headers (as is typical for EnzymeML v1 data files). The resulting DataFrame
/// provides efficient access to columnar data for further processing.
///
/// # Arguments
///
/// * `file_annot` - The file annotation containing the archive location path
/// * `archive` - The COMBINE archive to extract the file from
///
/// # Returns
///
/// A Polars DataFrame containing the parsed tabular data with numeric columns.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - The file cannot be found in the archive
/// - The file content cannot be read as valid UTF-8
/// - The CSV/TSV parsing fails due to malformed data
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

/// Extracts time column data and resolves time unit definitions.
///
/// In EnzymeML v1, time data is stored in a dedicated column identified by its column type.
/// This function locates the time column, extracts its numeric data, and resolves the
/// associated unit definition from the SBML model.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `format_annot` - The format annotation containing column type definitions
/// * `df` - The DataFrame containing the measurement data
///
/// # Returns
///
/// A tuple containing:
/// - Vector of time values as f64
/// - Resolved UnitDefinition for the time axis
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - No time column is defined in the format annotation
/// - The time unit cannot be resolved from the SBML model
/// - Time column data cannot be converted to numeric values
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

/// Creates a MeasurementData object for a single species from initial concentration data.
///
/// This function processes an initial concentration annotation to create a complete
/// MeasurementData object. It handles both initial concentration values and optional
/// time-series data if the species has a corresponding data column.
///
/// The function determines the appropriate data type (concentration, activity, etc.)
/// based on column annotations and sets up proper unit definitions for both the
/// measurement values and time axis.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `init_conc` - The initial concentration annotation for this species
/// * `format_annot` - The format annotation containing column definitions
/// * `df` - The DataFrame containing measurement data
/// * `time_data` - Pre-extracted time values for the measurement
/// * `time_unit` - Pre-resolved time unit definition
///
/// # Returns
///
/// A complete `MeasurementData` object with initial values and optional time-series data.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - The species ID cannot be determined from the annotation
/// - Unit definitions cannot be resolved
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

/// Finds a column annotation for a specific species in the format definition.
///
/// This helper function searches through column annotations to find one that corresponds
/// to a specific species ID. This is used to determine if time-series data is available
/// for a species and to identify the correct column index and data type.
///
/// # Arguments
///
/// * `format_annot` - The format annotation containing all column definitions
/// * `column_name` - The species ID to search for
///
/// # Returns
///
/// An optional reference to the matching column annotation, or None if not found.
fn get_column<'a>(format_annot: &'a FormatAnnot, column_name: &str) -> Option<&'a ColumnAnnot> {
    format_annot
        .columns
        .iter()
        .filter(|col| col.species_id.is_some())
        .find(|col| col.species_id.as_ref().unwrap() == column_name)
}

/// Resolves a unit definition from the SBML model by unit ID.
///
/// This function looks up a unit definition in the SBML model's unit definition list
/// and converts it to an EnzymeML UnitDefinition object. Unit definitions are essential
/// for proper interpretation of measurement data and ensuring dimensional consistency.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing unit definitions
/// * `unit` - The unit ID string to look up
///
/// # Returns
///
/// The corresponding `UnitDefinition` object with proper unit scaling and base units.
///
/// # Errors
///
/// Returns `SBMLError::MissingUnit` if:
/// - The SBML document has no model
/// - The specified unit ID is not found in the model's unit definitions
/// - The unit definition cannot be converted to EnzymeML format
fn extract_unit(sbml: &SBMLDocument, unit: &str) -> Result<UnitDefinition, SBMLError> {
    if let Some(model) = sbml.model() {
        if let Some(unit_def) = model.get_unit_definition(unit) {
            return UnitDefinition::try_from(unit_def.as_ref());
        }
    }
    Err(SBMLError::MissingUnit(unit.to_string()))
}

/// Resolves file and format annotation references for a measurement.
///
/// Each measurement annotation contains references to file and format IDs that must be
/// resolved against the complete data annotations structure. This function performs
/// these lookups and returns the referenced annotation objects.
///
/// # Arguments
///
/// * `meas` - The measurement annotation containing file and format references
/// * `data_annotations` - The complete data annotations structure containing all definitions
///
/// # Returns
///
/// A tuple containing references to the resolved file and format annotations.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - The referenced file ID cannot be found in the data annotations
/// - The referenced format ID cannot be found in the data annotations
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

/// Retrieves and reads file content from the COMBINE archive.
///
/// This function extracts a file from the COMBINE archive using the location path
/// specified in the file annotation and returns its content as a UTF-8 string.
/// The content is typically TSV data that will be parsed into a DataFrame.
///
/// # Arguments
///
/// * `file` - The file annotation containing the archive location path
/// * `archive` - The COMBINE archive to extract the file from
///
/// # Returns
///
/// The complete file content as a UTF-8 string.
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - The file cannot be found at the specified location in the archive
/// - The file content cannot be read or decoded as valid UTF-8
fn get_file_content(file: &FileAnnot, archive: &mut CombineArchive) -> Result<String, SBMLError> {
    let file_content = archive
        .entry(&file.location)
        .map_err(SBMLError::CombineArchiveError)?;

    let file_content = file_content
        .as_string()
        .map_err(SBMLError::SBMLReaderError)?;

    Ok(file_content)
}
