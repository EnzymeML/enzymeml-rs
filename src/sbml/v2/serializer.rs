//! SBML v2 serialization module
//!
//! This module provides functionality to serialize EnzymeML documents to SBML v2 format
//! files and COMBINE archives containing measurement data.
//!
//! The v2 serialization system converts complete EnzymeML data structures into SBML
//! annotations and associated CSV data files. Unlike the v1 format, v2 uses a simplified
//! tab-separated data structure with direct species column mapping and measurement grouping
//! by ID within single files.
//!
//! # Core Functionality
//!
//! The module handles the conversion of EnzymeML data structures to the v2 annotation format,
//! including:
//! - Writing measurement data to CSV files in COMBINE archives
//! - Converting EnzymeML entities to SBML v2 annotation structures
//! - Serializing time-series data with proper unit associations
//! - Processing experimental conditions and metadata
//! - Maintaining data integrity and unit consistency
//!
//! # Data Flow
//!
//! The serialization process follows this structure:
//! 1. **Entity Conversion**: EnzymeML objects are converted to v2 schema structures
//! 2. **Data Preparation**: Time-series data is prepared for tabular export
//! 3. **File Generation**: CSV measurement data is written to COMBINE archive
//! 4. **Annotation Creation**: SBML annotations are created with metadata and file references
//! 5. **Validation**: Unit definitions and data consistency are preserved
//!
//! # File Format
//!
//! The v2 format generates tab-separated CSV files with specific column structures:
//! - `id` column for measurement identification and grouping
//! - `time` column for temporal data points with consistent units
//! - Species columns named by species ID containing concentration/activity data
//! - Consistent measurement grouping by ID for multi-species datasets
//! - Comprehensive experimental condition metadata embedded in annotations
//!
//! This module provides `From` and `TryFrom` trait implementations that convert
//! EnzymeML data structures into their corresponding SBML v2 annotation formats.
//! These conversions are used when serializing EnzymeML documents to SBML format.

use polars::{
    io::SerWriter,
    prelude::{concat, CsvWriter, IntoLazy, NamedFrom, UnionArgs},
    series::Series,
};
use sbml::{combine::KnownFormats, prelude::CombineArchive};

use crate::{
    prelude::{
        Complex, EnzymeMLDocument, Measurement, MeasurementData, Parameter, Protein, SmallMolecule,
        Variable,
    },
    sbml::{
        error::SBMLError,
        v2::{
            self,
            schema::{ConditionsAnnot, PHAnnot, TemperatureAnnot, ENZYMEML_V2_NS},
        },
    },
};

/// Converts an EnzymeML document to a COMBINE archive in SBML v2 format
///
/// This function creates measurement data files within a COMBINE archive by processing
/// all measurements in the EnzymeML document and generating a consolidated tab-separated
/// file containing time-series data for all species across all measurements.
///
/// The function processes measurements by converting each to a lazy dataframe, concatenating
/// them into a unified dataset, and writing the result as a tab-separated file to the
/// archive. This approach ensures efficient memory usage and maintains data integrity
/// across the serialization process.
///
/// # Data Structure
///
/// The generated CSV file contains:
/// - `id` column identifying each measurement
/// - `time` column with temporal data points
/// - Species columns named by species ID with concentration/activity values
/// - Proper tab separation for compatibility with v2 format expectations
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document containing measurements to serialize
/// * `archive` - Mutable reference to the COMBINE archive for file storage
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or serialization error
///
/// # Errors
///
/// This function will return an error if:
/// - DataFrame conversion fails for any measurement
/// - Concatenation of measurement data fails due to schema mismatches
/// - CSV writing operations fail due to I/O or formatting issues
/// - Archive entry creation fails due to storage constraints
pub(crate) fn write_measurement_data(
    enzmldoc: &EnzymeMLDocument,
    archive: &mut CombineArchive,
) -> Result<(), SBMLError> {
    let dfs = enzmldoc
        .measurements
        .iter()
        .map(measurement_to_lazy_dataframe)
        .collect::<Result<Vec<_>, SBMLError>>()?;

    let mut df = concat(dfs, UnionArgs::default())?.collect()?;
    let mut string_buffer = Vec::new();

    CsvWriter::new(&mut string_buffer)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df)?;

    archive.add_entry(
        "data.tsv",
        KnownFormats::TSV,
        false,
        string_buffer.as_slice(),
    )?;

    Ok(())
}

/// Converts a measurement to a lazy dataframe with measurement ID as the first column
///
/// This function transforms a measurement object into a lazy dataframe suitable for
/// concatenation with other measurements. It adds an `id` column containing the
/// measurement identifier repeated for each row, enabling proper measurement grouping
/// in the consolidated data file.
///
/// The lazy evaluation approach optimizes memory usage during the conversion process
/// and allows for efficient concatenation operations when combining multiple
/// measurements into a single dataset.
///
/// # Arguments
/// * `measurement` - The measurement to convert to dataframe format
///
/// # Returns
/// * `Result<polars::prelude::LazyFrame, SBMLError>` - The lazy dataframe with ID column or an error
///
/// # Errors
///
/// This function will return an error if:
/// - DataFrame creation from measurement data fails
/// - Column addition operations fail due to shape mismatches
/// - Memory allocation fails during dataframe construction
fn measurement_to_lazy_dataframe(
    measurement: &Measurement,
) -> Result<polars::prelude::LazyFrame, SBMLError> {
    let mut df = measurement.to_dataframe(false);
    let id_series = Series::new("id", vec![measurement.id.clone().as_str(); df.height()]);

    df.with_column(id_series)?;
    Ok(df.lazy())
}

/// Converts a `SmallMolecule` to its SBML v2 annotation representation
///
/// This implementation maps chemical identifiers from the EnzymeML small molecule
/// format to the corresponding SBML v2 annotation fields. The conversion preserves
/// all available chemical identifier information including InChI, InChIKey, and
/// canonical SMILES representations for comprehensive chemical database linking.
///
/// The conversion maintains chemical accuracy and enables cross-referencing with
/// external chemical databases while preserving structural information in multiple
/// standardized formats for maximum interoperability.
impl From<&SmallMolecule> for v2::schema::SmallMoleculeAnnot {
    fn from(small_molecule: &SmallMolecule) -> Self {
        Self {
            xmlns: ENZYMEML_V2_NS.to_string(),
            inchikey: small_molecule.inchikey.clone(),
            inchi: small_molecule.inchi.clone(),
            canonical_smiles: small_molecule.canonical_smiles.clone(),
            synonyms: Some(small_molecule.synonymous_names.clone()),
        }
    }
}

/// Converts a `Protein` to its SBML v2 annotation representation
///
/// This implementation maps protein metadata including enzymatic classification,
/// organism information, and sequence data to the SBML v2 annotation format. The
/// conversion preserves all biological context information necessary for proper
/// protein identification and functional annotation.
///
/// The mapping ensures compatibility with protein databases and maintains enzymatic
/// classification information for kinetic modeling and biochemical analysis purposes.
impl From<&Protein> for v2::schema::ProteinAnnot {
    fn from(protein: &Protein) -> Self {
        Self {
            xmlns: ENZYMEML_V2_NS.to_string(),
            ecnumber: protein.ecnumber.clone(),
            organism: protein.organism.clone(),
            organism_tax_id: protein.organism_tax_id.clone(),
            sequence: protein.sequence.clone(),
        }
    }
}

/// Converts a `Complex` to its SBML v2 annotation representation
///
/// This implementation maps complex participant information to the SBML v2
/// annotation format for molecular complexes. The conversion preserves the
/// structural composition of multi-component molecular assemblies and maintains
/// participant relationships for complex modeling applications.
///
/// The mapping enables proper representation of protein-protein and protein-substrate
/// interactions within the SBML annotation framework while preserving complex topology.
impl From<&Complex> for v2::schema::ComplexAnnot {
    fn from(complex: &Complex) -> Self {
        Self {
            xmlns: ENZYMEML_V2_NS.to_string(),
            participants: complex.participants.clone(),
        }
    }
}

/// Converts a `Parameter` to its SBML v2 annotation representation
///
/// This implementation maps parameter values, bounds, and statistical information
/// to the SBML v2 annotation format. The conversion preserves all parameter metadata
/// including optimization constraints, uncertainty estimates, and unit associations
/// necessary for kinetic model fitting and parameter estimation.
///
/// The mapping maintains statistical rigor by preserving confidence bounds and
/// uncertainty information while ensuring proper unit reference handling for
/// dimensional analysis and model validation.
impl From<&Parameter> for v2::schema::ParameterAnnot {
    fn from(parameter: &Parameter) -> Self {
        let unit_id = parameter.unit.as_ref().and_then(|unit| unit.id.clone());

        Self {
            xmlns: ENZYMEML_V2_NS.to_string(),
            initial: parameter.initial_value,
            lower_bound: parameter.lower_bound,
            upper_bound: parameter.upper_bound,
            stderr: parameter.stderr,
            unit: unit_id,
        }
    }
}

/// Converts a `Variable` to its SBML v2 annotation representation
///
/// This implementation maps variable metadata including identifier, name, and
/// symbol to the SBML v2 annotation format. The conversion preserves mathematical
/// variable definitions used in kinetic equations and model expressions while
/// maintaining proper symbol mapping for equation resolution.
///
/// The mapping ensures compatibility with mathematical modeling frameworks and
/// preserves variable semantics for kinetic model construction and analysis.
impl From<&Variable> for v2::schema::VariableAnnot {
    fn from(variable: &Variable) -> Self {
        Self {
            id: Some(variable.id.clone()),
            name: Some(variable.name.clone()),
            symbol: Some(variable.symbol.clone()),
        }
    }
}

/// Converts a slice of measurements to SBML v2 data annotation representation
///
/// This implementation creates a data annotation structure that references the
/// consolidated measurement data file and contains metadata for all measurements
/// in the collection. The conversion establishes the link between SBML annotations
/// and external data files within COMBINE archives.
///
/// The data annotation serves as the primary interface for accessing measurement
/// data during SBML processing and maintains referential integrity between
/// annotation metadata and actual time-series data storage.
impl TryFrom<&[Measurement]> for v2::schema::DataAnnot {
    type Error = SBMLError;
    fn try_from(measurements: &[Measurement]) -> Result<Self, Self::Error> {
        Ok(Self {
            xmlns: ENZYMEML_V2_NS.to_string(),
            file: "data.tsv".to_string(),
            measurements: measurements
                .iter()
                .map(|measurement| measurement.into())
                .collect(),
        })
    }
}

/// Converts a `Measurement` to its SBML v2 annotation representation
///
/// This implementation extracts time unit information from species data, converts
/// experimental conditions, and maps all species measurement data to the SBML v2
/// annotation format. The conversion preserves experimental context including pH,
/// temperature, and unit associations necessary for proper data interpretation.
///
/// The mapping process extracts time unit information from the first available
/// species data entry and converts experimental conditions to their annotation
/// representations while maintaining data relationships and unit consistency.
impl From<&Measurement> for v2::schema::MeasurementAnnot {
    fn from(measurement: &Measurement) -> Self {
        let time_unit = measurement
            .species_data
            .iter()
            .filter_map(|meas_data| meas_data.time_unit.clone().and_then(|unit| unit.id))
            .next();

        let condition: ConditionsAnnot = measurement.into();

        Self {
            id: measurement.id.clone(),
            name: Some(measurement.name.clone()),
            time_unit,
            conditions: Some(condition),
            species_data: measurement
                .species_data
                .iter()
                .map(|meas_data| {
                    meas_data
                        .try_into()
                        .expect("Failed to convert measurement data")
                })
                .collect(),
        }
    }
}

/// Converts `MeasurementData` to its SBML v2 species data annotation representation
///
/// This implementation extracts unit information and maps measurement data to the
/// SBML v2 format. The conversion ensures proper unit association and validates
/// that required unit information is available for data interpretation and analysis.
///
/// The mapping preserves data type information, initial conditions, and unit
/// references necessary for proper measurement data reconstruction during
/// extraction operations and maintains data integrity throughout the conversion process.
///
/// # Errors
///
/// Returns `SBMLError::MissingUnit` if the measurement data lacks required unit
/// information or if unit ID resolution fails during the conversion process.
impl TryFrom<&MeasurementData> for v2::schema::SpeciesDataAnnot {
    type Error = SBMLError;
    fn try_from(measurement_data: &MeasurementData) -> Result<Self, Self::Error> {
        let unit_id = measurement_data
            .data_unit
            .clone()
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?
            .id
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?;

        Ok(Self {
            species_id: measurement_data.species_id.clone(),
            initial: measurement_data.initial,
            data_type: measurement_data
                .data_type
                .clone()
                .unwrap_or(crate::prelude::DataTypes::Concentration),
            unit: unit_id,
        })
    }
}

/// Converts a `Measurement` to its SBML v2 conditions annotation representation
///
/// This implementation extracts pH and temperature conditions from the measurement
/// and maps them to the SBML v2 annotation format, including unit information for
/// temperature measurements. The conversion preserves experimental conditions
/// necessary for proper data interpretation and experimental reproducibility.
///
/// The mapping process handles optional condition values gracefully while
/// maintaining unit associations for temperature measurements and preserving
/// experimental context for kinetic analysis and model validation.
impl From<&Measurement> for v2::schema::ConditionsAnnot {
    fn from(measurement: &Measurement) -> Self {
        let ph = measurement.ph.map(|ph| PHAnnot { value: Some(ph) });

        let temp_unit = measurement
            .temperature_unit
            .clone()
            .and_then(|unit| unit.id);
        let temperature = measurement.temperature.map(|value| TemperatureAnnot {
            value: Some(value),
            unit: temp_unit,
        });

        Self { ph, temperature }
    }
}
