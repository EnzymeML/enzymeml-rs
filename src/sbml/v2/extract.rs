//! SBML v2 extraction module
//!
//! This module provides functionality to extract EnzymeML documents from SBML v2 format
//! files and COMBINE archives containing measurement data.
//!
//! The v2 extraction system reconstructs complete EnzymeML measurement objects from SBML
//! annotations and associated CSV data files. Unlike the v1 format, v2 uses a simplified
//! tab-separated data structure with direct species column mapping and measurement grouping
//! by ID within single files.
//!
//! # Core Functionality
//!
//! The module handles the extraction of SBML data structures from the v2 annotation format,
//! including:
//! - Reading measurement data from CSV files in COMBINE archives
//! - Converting SBML v2 annotation structures to EnzymeML entities
//! - Reconstructing measurement time-series data with proper unit associations
//! - Processing experimental conditions and metadata
//! - Validating data integrity and unit consistency
//!
//! # Data Flow
//!
//! The extraction process follows this structure:
//! 1. **Annotation Parsing**: SBML model annotations are deserialized into v2 schema structures
//! 2. **Data File Reading**: CSV measurement data is loaded from COMBINE archive references
//! 3. **Entity Reconstruction**: Annotation data is converted back to EnzymeML measurement objects
//! 4. **Data Association**: Time-series data is matched with species and unit definitions
//! 5. **Validation**: Unit definitions and data consistency are verified
//!
//! # File Format
//!
//! The v2 format expects tab-separated CSV files with specific column structures:
//! - `id` column for measurement identification and grouping
//! - `time` column for temporal data points with consistent units
//! - Species columns named by species ID containing concentration/activity data
//! - Consistent measurement grouping by ID for multi-species datasets
//! - Optional experimental condition metadata embedded in annotations

use polars::{
    frame::DataFrame,
    io::SerReader,
    prelude::{CsvParseOptions, CsvReadOptions},
    series::{ChunkCompare, Series},
};
use sbml::{model::Model, prelude::CombineArchive, Annotation, SBMLDocument};

use crate::{
    prelude::{Measurement, MeasurementData, UnitDefinition},
    sbml::{
        error::SBMLError,
        units::get_unit_definition,
        utils::convert_column_to_vec,
        v2::schema::{DataAnnot, MeasurementAnnot, SpeciesDataAnnot},
    },
};

impl TryFrom<&SBMLDocument> for DataAnnot {
    type Error = SBMLError;

    /// Extract data annotations from an SBML document
    ///
    /// This implementation extracts measurement metadata and file references from the SBML model's
    /// annotation section. The data annotation provides the structural information needed to
    /// reconstruct measurement data from external CSV files referenced in COMBINE archives.
    ///
    /// The extraction process locates the model within the SBML document and deserializes
    /// the data annotation structure, which contains measurement definitions, species mappings,
    /// experimental conditions, and file references pointing to the actual time-series data.
    ///
    /// # Returns
    ///
    /// Returns the deserialized `DataAnnot` structure containing all measurement metadata,
    /// or an empty data annotation if none exists in the model.
    ///
    /// # Errors
    ///
    /// Returns `SBMLError::MissingModel` if the SBML document does not contain a model,
    /// or annotation parsing errors if the data annotation format is invalid.
    fn try_from(sbml: &SBMLDocument) -> Result<Self, Self::Error> {
        let model = sbml.model().ok_or(SBMLError::MissingModel)?;
        let data_annotations = model.get_annotation_serde::<DataAnnot>()?;

        Ok(data_annotations)
    }
}

/// Extract measurements from an SBML model using data annotations and a combine archive
///
/// This function reconstructs complete `Measurement` objects from SBML v2 data annotations
/// by reading the referenced CSV file from the COMBINE archive and associating the time-series
/// data with the appropriate species and unit definitions from the SBML model.
///
/// The extraction process involves parsing the tab-separated data file, filtering measurements
/// by ID, and creating measurement objects with properly typed and unit-annotated data series.
/// Each measurement contains complete metadata including experimental conditions, species
/// data with proper unit associations, and validated time-series information.
///
/// # Arguments
///
/// * `sbml` - The SBML document containing the model with unit definitions and validation context
/// * `data_annot` - Data annotations containing measurement metadata and file references
/// * `archive` - Mutable reference to the combine archive containing the data files
///
/// # Returns
///
/// Returns a vector of fully reconstructed `Measurement` objects with time-series data,
/// experimental conditions, and proper unit associations ready for analysis and modeling.
///
/// # Errors
///
/// This function will return an error if:
/// - The referenced data file cannot be found in the archive
/// - The CSV file cannot be parsed or has an invalid format
/// - Unit definitions referenced in the annotations cannot be found in the model
/// - Required columns (id, time) are missing from the data file
/// - Data validation fails during reconstruction
pub(crate) fn extract_measurements(
    sbml: &SBMLDocument,
    data_annot: &DataAnnot,
    archive: &mut CombineArchive,
) -> Result<Vec<Measurement>, SBMLError> {
    let model = sbml.model().ok_or(SBMLError::MissingModel)?;
    let df = extract_measurement_dataframe(&data_annot.file, archive)?;
    let measurements = data_annot
        .measurements
        .iter()
        .map(|meas_annot| extract_measurement(meas_annot, &model, &df))
        .collect::<Result<Vec<Measurement>, SBMLError>>()?;

    Ok(measurements)
}

/// Extract a DataFrame from a CSV file in the combine archive
///
/// This function reads a tab-separated CSV file from the combine archive and parses it
/// into a Polars DataFrame for measurement data processing. The function handles the
/// file extraction from the archive and configures the CSV parser for tab-separated
/// values as expected by the v2 format specification.
///
/// The parsing configuration is optimized for the v2 data format which uses tab separation
/// for improved data integrity and compatibility with common scientific data formats.
/// This approach reduces parsing ambiguity compared to comma-separated formats.
///
/// # Arguments
///
/// * `file` - The filename/path of the CSV file within the archive
/// * `archive` - Mutable reference to the combine archive for file access
///
/// # Returns
///
/// Returns a parsed `DataFrame` containing the measurement data with proper column
/// types and structure for further processing and data extraction operations.
///
/// # Errors
///
/// This function will return an error if:
/// - The file cannot be found in the archive or access permissions fail
/// - The file content cannot be parsed as a valid CSV format
/// - The CSV structure doesn't match expected column requirements
/// - Memory allocation fails during DataFrame construction
fn extract_measurement_dataframe(
    file: &str,
    archive: &mut CombineArchive,
) -> Result<DataFrame, SBMLError> {
    let file_content = archive.entry(file)?;
    let cursor = std::io::Cursor::new(file_content.as_bytes());
    let df = CsvReadOptions::default()
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .into_reader_with_file_handle(cursor)
        .finish()?;

    Ok(df)
}

/// Extract a single measurement from measurement annotations and DataFrame
///
/// This function processes a measurement annotation to create a complete `Measurement` object
/// by combining metadata from the annotation with actual time-series data from the DataFrame.
/// The process includes resolving unit definitions, extracting experimental conditions,
/// and associating species data with the correct time points.
///
/// The extraction ensures data integrity by validating unit references against the SBML model
/// and maintaining consistency between annotation metadata and actual data values. All
/// experimental conditions are properly processed and associated with their respective
/// unit definitions for accurate data interpretation.
///
/// # Arguments
///
/// * `meas_annot` - The measurement annotation containing metadata and species definitions
/// * `model` - The SBML model for unit definition lookups and validation
/// * `df` - The DataFrame containing the raw measurement data for processing
///
/// # Returns
///
/// Returns a fully reconstructed `Measurement` object with time-series data, experimental
/// conditions, and proper unit associations for all species measurements with validated
/// data integrity and consistent metadata.
///
/// # Errors
///
/// This function will return an error if:
/// - Unit definitions referenced in annotations cannot be found in the model
/// - The DataFrame cannot be filtered for the specific measurement ID
/// - Required columns (time, species data) are missing from the DataFrame
/// - Data type conversion fails during processing operations
/// - Unit validation fails for temperature or other experimental conditions
fn extract_measurement(
    meas_annot: &MeasurementAnnot,
    model: &Model,
    df: &DataFrame,
) -> Result<Measurement, SBMLError> {
    let mut measurement = Measurement::from(meas_annot);

    let time_unit = match meas_annot.time_unit.clone() {
        Some(unit_id) => Some(get_unit_definition(model, &unit_id)?),
        None => None,
    };

    if let Some(unit) = meas_annot
        .conditions
        .as_ref()
        .and_then(|c| c.temperature.as_ref())
        .and_then(|t| t.unit.as_ref())
    {
        measurement.temperature_unit = Some(get_unit_definition(model, unit)?);
    }

    let filtered_df = filter_dataframe(df, &measurement.id)?;
    let time_column = filtered_df.column("time")?;
    let time_values = convert_column_to_vec(time_column);

    measurement.species_data = meas_annot
        .species_data
        .iter()
        .map(|species_data| {
            extract_meas_data(species_data, model, &time_unit, &time_values, &filtered_df)
        })
        .collect::<Result<Vec<MeasurementData>, SBMLError>>()?;

    Ok(measurement)
}

/// Extract measurement data for a specific species from the DataFrame
///
/// This function creates a `MeasurementData` object for an individual species by extracting
/// the corresponding data column from the DataFrame and associating it with proper units
/// and metadata. The function handles missing species columns gracefully by creating
/// empty measurement data objects when species are defined but data is not available.
///
/// The extraction process maintains data integrity by ensuring proper unit associations
/// and handling edge cases where species are defined in annotations but may not have
/// corresponding data columns in the CSV file. This approach supports flexible data
/// structures while maintaining consistency in the resulting measurement objects.
///
/// # Arguments
///
/// * `species_data` - The species data annotation containing metadata and unit references
/// * `model` - The SBML model for unit definition lookups and validation
/// * `time_unit` - Optional time unit definition shared across all species in the measurement
/// * `time_values` - Vector of time values for the measurement time points
/// * `filtered_df` - DataFrame filtered for the specific measurement containing species columns
///
/// # Returns
///
/// Returns a `MeasurementData` object with time-series data, unit definitions, and species
/// metadata properly associated for downstream analysis and model reconstruction with
/// validated unit consistency and data integrity.
///
/// # Errors
///
/// This function will return an error if:
/// - The unit definition for the species data cannot be found in the model
/// - Unit reference parsing fails during definition lookup operations
/// - Critical data validation fails during object construction
fn extract_meas_data(
    species_data: &SpeciesDataAnnot,
    model: &Model,
    time_unit: &Option<UnitDefinition>,
    time_values: &[f64],
    filtered_df: &DataFrame,
) -> Result<MeasurementData, SBMLError> {
    let mut meas_data = MeasurementData::from(species_data);

    meas_data.time_unit = time_unit.clone();
    meas_data.data_unit = get_unit_definition(model, &species_data.unit)?.into();

    if let Ok(column) = filtered_df.column(&species_data.species_id) {
        meas_data.data = convert_column_to_vec(column);
        meas_data.time = time_values.to_owned();
    }

    Ok(meas_data)
}

/// Filter a DataFrame to include only rows matching a specific measurement ID
///
/// This function creates a boolean mask to filter the DataFrame based on the "id" column,
/// isolating rows that belong to a specific measurement. This is essential for processing
/// datasets that contain multiple measurements in a single CSV file, allowing each
/// measurement to be processed independently with its own time series and species data.
///
/// The filtering operation is optimized for large datasets and maintains data integrity
/// by preserving all columns while efficiently reducing the row count to only relevant
/// measurement data. This approach supports the v2 format's consolidated file structure
/// while enabling individual measurement processing.
///
/// # Arguments
///
/// * `df` - The DataFrame containing multiple measurements to filter
/// * `id` - The measurement ID to filter by for data isolation
///
/// # Returns
///
/// Returns a filtered `DataFrame` containing only rows matching the specified measurement ID,
/// preserving all columns but reducing rows to the relevant measurement data with
/// maintained column structure and data types.
///
/// # Errors
///
/// This function will return an error if:
/// - The "id" column is missing from the DataFrame structure
/// - The filtering operation fails due to data type mismatches
/// - Memory allocation fails during DataFrame operations
/// - The measurement ID format is incompatible with the data structure
fn filter_dataframe(df: &DataFrame, id: &str) -> Result<DataFrame, SBMLError> {
    let series: Series = vec![id; df.height()].into_iter().collect();
    let mask = df.column("id")?.equal(&series)?;
    df.filter(&mask).map_err(SBMLError::PolarsError)
}

impl From<&MeasurementAnnot> for Measurement {
    /// Convert a measurement annotation into a `Measurement` object
    ///
    /// This implementation extracts basic measurement information from the v2 annotation format,
    /// including measurement identification, naming, and experimental conditions such as pH
    /// and temperature. The conversion creates the measurement container structure that will
    /// be populated with species data during the extraction process.
    ///
    /// The conversion handles optional fields gracefully, providing sensible defaults where
    /// necessary while preserving all available metadata from the annotation structure.
    /// This ensures compatibility with various annotation completeness levels.
    ///
    /// # Fields Mapped
    ///
    /// - `id` - Unique measurement identifier for data association
    /// - `name` - Human-readable measurement name (defaults to ID if not provided)
    /// - `ph` - Experimental pH condition value with proper nesting handling
    /// - `temperature` - Experimental temperature condition value with unit support
    /// - `species_data` - Initialized as empty vector, populated separately during extraction
    /// - `temperature_unit` - Initialized as None, resolved during extraction process
    /// - `group_id` - Initialized as None, not supported in v2 format
    fn from(measurement_annot: &MeasurementAnnot) -> Self {
        let conditions = measurement_annot.conditions.clone().unwrap_or_default();
        Measurement {
            id: measurement_annot.id.clone(),
            name: measurement_annot
                .name
                .clone()
                .unwrap_or(measurement_annot.id.clone()),
            species_data: vec![],
            group_id: None,
            ph: conditions.ph.and_then(|ph| ph.value),
            temperature: conditions.temperature.and_then(|temp| temp.value),
            temperature_unit: None,
        }
    }
}

impl From<&SpeciesDataAnnot> for MeasurementData {
    /// Convert a species data annotation into a `MeasurementData` object
    ///
    /// This implementation extracts species-specific measurement information from the v2
    /// annotation format, including species identification, initial concentration values,
    /// and data type specifications. The conversion creates the data container structure
    /// that will be populated with actual time-series data during the extraction process.
    ///
    /// The conversion maintains consistency with EnzymeML data model requirements while
    /// preserving all metadata from the annotation structure. This ensures proper data
    /// association and maintains traceability between annotation and measurement data.
    ///
    /// # Fields Mapped
    ///
    /// - `species_id` - Reference to the species this data belongs to for proper association
    /// - `initial` - Initial concentration or activity value from experimental setup
    /// - `prepared` - Prepared concentration (copies initial value for consistency)
    /// - `data_type` - Type of measurement data (concentration, activity, etc.)
    /// - `data` - Initialized as empty vector, populated from DataFrame during extraction
    /// - `time` - Initialized as empty vector, populated from DataFrame during extraction
    /// - `data_unit` - Initialized as None, resolved during extraction with unit validation
    /// - `time_unit` - Initialized as None, resolved during extraction with unit validation
    /// - `is_simulated` - Initialized as None, not typically specified in v2 annotations
    fn from(species_data: &SpeciesDataAnnot) -> Self {
        MeasurementData {
            species_id: species_data.species_id.clone(),
            prepared: species_data.initial,
            initial: species_data.initial,
            data_unit: None,
            data: vec![],
            time: vec![],
            time_unit: None,
            data_type: Some(species_data.data_type.clone()),
            is_simulated: None,
        }
    }
}
