//! SBML v1 serialization module
//!
//! This module provides functionality to convert EnzymeML documents to SBML v1 format
//! and create COMBINE archives containing the SBML model and associated data.
//!
//! The module handles the serialization of EnzymeML data structures into the appropriate
//! SBML v1 annotation format, including:
//! - Writing measurement data as CSV files to COMBINE archives
//! - Converting EnzymeML entities (small molecules, proteins, complexes, parameters) to their corresponding annotation structures
//! - Transforming measurement data into the v1 data annotation format with proper file references and column definitions
//!
//! # Data Flow
//!
//! The serialization process follows this structure:
//! 1. **Measurement Data Export**: Individual measurements are converted to CSV files and added to the COMBINE archive
//! 2. **Entity Annotations**: Core EnzymeML entities are converted to their SBML annotation equivalents
//! 3. **Data Structure Annotations**: Measurement metadata is organized into the hierarchical v1 data annotation format
//!
//! # Annotation Structure
//!
//! The v1 format organizes data annotations in a nested structure:
//! - `DataAnnot`: Root container with files, formats, and measurements
//! - `FilesWrapper`: Contains file references pointing to CSV data files
//! - `FormatsWrapper`: Defines column structures and data organization
//! - `MeasurementsWrapper`: Contains measurement metadata and initial concentrations

use polars::{frame::DataFrame, io::SerWriter, prelude::CsvWriter};
use sbml::{combine::KnownFormats, prelude::CombineArchive};

use crate::{
    prelude::{
        Complex, DataTypes, EnzymeMLDocument, Measurement, MeasurementData, Parameter, Protein,
        SmallMolecule,
    },
    sbml::{error::SBMLError, units::replace_slashes},
};

use super::schema::{
    ColumnAnnot, ColumnType, ComplexAnnot, DataAnnot, FileAnnot, FilesWrapper, FormatAnnot,
    FormatsWrapper, InitConcAnnot, MeasurementAnnot, MeasurementsWrapper, ParameterAnnot,
    ProteinAnnot, ReactantAnnot, ENZYMEML_V1_NS,
};

/// Writes measurement data from an EnzymeML document to a COMBINE archive as CSV files
///
/// This function processes each measurement in the EnzymeML document and exports the
/// time-series data as individual CSV files within the archive. The CSV files are stored
/// in the `data/` directory with filenames matching the measurement IDs.
///
/// The exported CSV format follows the v1 specification:
/// - No headers are included in the CSV files
/// - Time column is always first
/// - Species data columns follow in measurement order
/// - File paths follow the pattern `data/{measurement_id}.csv`
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document containing measurement data to export
/// * `archive` - Mutable reference to the COMBINE archive where CSV files will be added
///
/// # Returns
///
/// * `Result<(), SBMLError>` - Success or an error during CSV generation or archive writing
///
/// # Errors
///
/// Returns `SBMLError` if:
/// - DataFrame conversion fails for any measurement
/// - CSV serialization encounters formatting issues
/// - Archive entry addition fails due to I/O errors
pub(crate) fn write_measurement_data(
    enzmldoc: &EnzymeMLDocument,
    archive: &mut CombineArchive,
) -> Result<(), SBMLError> {
    for measurement in enzmldoc.measurements.iter() {
        let mut df: DataFrame = measurement.to_dataframe(false);
        let mut string_buffer = Vec::new();
        CsvWriter::new(&mut string_buffer)
            .include_header(false)
            .finish(&mut df)?;
        archive.add_entry(
            format!("data/{}.csv", measurement.id),
            KnownFormats::CSV,
            false,
            string_buffer.as_slice(),
        )?;
    }

    Ok(())
}

// Type conversions for annotations

/// Converts a SmallMolecule to a ReactantAnnot for SBML v1 annotation
///
/// This implementation extracts chemical identifiers from the EnzymeML small molecule
/// representation and maps them to the v1 reactant annotation format. The conversion
/// focuses on structural identifiers commonly used in biochemical databases.
///
/// # Fields Mapped
///
/// - `inchi` - International Chemical Identifier for structural representation
/// - `smiles` - Canonical SMILES notation for chemical structure
/// - `chebi_id` - Set to None as this is typically handled separately in v1 format
impl From<&SmallMolecule> for ReactantAnnot {
    fn from(small_molecule: &SmallMolecule) -> Self {
        ReactantAnnot {
            // xmlns: ENZYMEML_V1_NS.to_string(),
            inchi: small_molecule.inchi.clone(),
            smiles: small_molecule.canonical_smiles.clone(),
            chebi_id: None,
        }
    }
}

/// Converts a Protein to a ProteinAnnot for SBML v1 annotation
///
/// This implementation transforms EnzymeML protein data into the v1 protein annotation
/// format, preserving sequence information, enzyme classification, and taxonomic data.
/// The conversion ensures compatibility with standard protein databases and classification systems.
///
/// # Fields Mapped
///
/// - `sequence` - Primary amino acid sequence
/// - `ecnumber` - Enzyme Commission number for functional classification
/// - `organism` - Source organism name
/// - `organism_tax_id` - NCBI Taxonomy ID for species identification
/// - `uniprotid` - Set to None, handled separately in v1 format
impl From<&Protein> for ProteinAnnot {
    fn from(protein: &Protein) -> Self {
        ProteinAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            sequence: protein.sequence.clone(),
            ecnumber: protein.ecnumber.clone(),
            uniprotid: None,
            organism: protein.organism.clone(),
            organism_tax_id: protein.organism_tax_id.clone(),
            additional_fields: std::collections::HashMap::new(),
        }
    }
}

/// Converts a Complex to a ComplexAnnot for SBML v1 annotation
///
/// This implementation handles the conversion of macromolecular complexes to the v1
/// annotation format. The conversion preserves participant information that defines
/// the composition and stoichiometry of the complex.
///
/// # Fields Mapped
///
/// - `participants` - List of component species and their roles in the complex
impl From<&Complex> for ComplexAnnot {
    fn from(complex: &Complex) -> Self {
        ComplexAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            participants: complex.participants.clone(),
        }
    }
}

/// Converts a Parameter to a ParameterAnnot for SBML v1 annotation
///
/// This implementation transforms EnzymeML parameter definitions into the v1 parameter
/// annotation format, preserving bounds and initial values used for parameter estimation
/// and model simulation. Unit information is processed to ensure compatibility with
/// SBML unit definitions.
///
/// # Fields Mapped
///
/// - `initial` - Initial value for parameter estimation
/// - `upper` - Upper bound constraint for optimization
/// - `lower` - Lower bound constraint for optimization
/// - `unit` - Unit definition ID, extracted from the parameter's unit reference
impl From<&Parameter> for ParameterAnnot {
    fn from(parameter: &Parameter) -> Self {
        ParameterAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            initial: parameter.initial_value,
            upper: parameter.upper_bound,
            lower: parameter.lower_bound,
            unit: parameter.unit.as_ref().and_then(|unit| unit.id.clone()),
        }
    }
}

/// Converts a slice of Measurements to a DataAnnot for SBML v1 annotation
///
/// This implementation creates the complete data annotation structure required for v1 format,
/// organizing measurement information into the hierarchical structure of files, formats,
/// and measurement metadata. The conversion process ensures that all measurement data
/// can be properly referenced and reconstructed from the SBML annotation.
///
/// # Structure Created
///
/// - `FormatsWrapper` - Defines column structures for each measurement's data format
/// - `MeasurementsWrapper` - Contains metadata and initial concentrations for each measurement
/// - `FilesWrapper` - Provides file references linking to CSV data files in the archive
///
/// # Errors
///
/// Returns `SBMLError` if any individual measurement conversion fails, typically due to:
/// - Missing required metadata (time units, data units)
/// - Invalid data structures in measurement data
/// - Inconsistent measurement format definitions
impl TryFrom<&[Measurement]> for DataAnnot {
    type Error = SBMLError;

    fn try_from(measurements: &[Measurement]) -> Result<Self, Self::Error> {
        let formats_wrapper = FormatsWrapper {
            format: measurements
                .iter()
                .map(FormatAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        };

        let measurements_wrapper = MeasurementsWrapper {
            measurement: measurements
                .iter()
                .map(MeasurementAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        };

        let files_wrapper = FilesWrapper {
            file: measurements.iter().map(FileAnnot::from).collect(),
        };

        Ok(DataAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            formats: formats_wrapper,
            measurements: measurements_wrapper,
            files: files_wrapper,
        })
    }
}

/// Converts a Measurement to a FileAnnot for SBML v1 annotation
///
/// This implementation creates file reference annotations that link measurement metadata
/// to the actual CSV data files in the COMBINE archive. The file annotation provides
/// the necessary information for data parsers to locate and process measurement data.
///
/// # File Naming Convention
///
/// - `id` - Generated as `file_{measurement_id}` for unique identification
/// - `location` - Points to `data/{measurement_id}.csv` in the archive
/// - `format` - References the format ID matching the measurement ID
impl From<&Measurement> for FileAnnot {
    fn from(measurement: &Measurement) -> Self {
        FileAnnot {
            id: format!("file_{}", measurement.id),
            location: format!("data/{}.csv", measurement.id),
            format: measurement.id.clone(),
        }
    }
}

/// Converts a Measurement to a MeasurementAnnot for SBML v1 annotation
///
/// This implementation creates measurement metadata annotations that describe the
/// experimental conditions and initial concentrations for each measurement. The
/// conversion processes all species data to extract initial concentration information
/// required for model reconstruction.
///
/// # Fields Generated
///
/// - `id` - Measurement identifier for cross-referencing
/// - `name` - Human-readable measurement name
/// - `file` - Reference to the associated file annotation
/// - `init_concs` - Initial concentration data for all species in the measurement
///
/// # Errors
///
/// Returns `SBMLError` if initial concentration conversion fails for any species,
/// typically due to missing initial values or unit definitions.
impl TryFrom<&Measurement> for MeasurementAnnot {
    type Error = SBMLError;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        Ok(MeasurementAnnot {
            id: measurement.id.clone(),
            name: measurement.name.clone(),
            file: format!("file_{}", measurement.id),
            init_concs: measurement
                .species_data
                .iter()
                .map(InitConcAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// Converts MeasurementData to an InitConcAnnot for SBML v1 annotation
///
/// This implementation extracts initial concentration information from measurement data
/// and formats it according to v1 annotation requirements. The conversion handles
/// unit processing to ensure compatibility with SBML unit definitions.
///
/// # Data Processing
///
/// - Validates that initial concentration values are present
/// - Processes unit definitions with slash replacement for SBML compatibility
/// - Maps species references to the appropriate annotation fields
///
/// # Errors
///
/// Returns `SBMLError::MissingInitialValue` if the measurement data lacks
/// required initial concentration values for proper model reconstruction.
impl TryFrom<&MeasurementData> for InitConcAnnot {
    type Error = SBMLError;

    fn try_from(measurement_data: &MeasurementData) -> Result<Self, Self::Error> {
        let initial = measurement_data
            .initial
            .ok_or(SBMLError::MissingInitialValue(
                measurement_data.species_id.clone(),
            ))?;

        let unit_id = measurement_data
            .data_unit
            .as_ref()
            .and_then(|unit| unit.id.clone());

        Ok(InitConcAnnot {
            protein: None,
            reactant: Some(measurement_data.species_id.clone()),
            value: initial,
            unit: unit_id.map(|id| replace_slashes(&id)),
        })
    }
}

/// Converts a Measurement to a FormatAnnot for SBML v1 annotation
///
/// This implementation creates column format definitions that describe the structure
/// and organization of measurement data in CSV files. The format annotation enables
/// proper parsing and interpretation of time-series data during model reconstruction.
///
/// # Column Organization
///
/// - Time column is always first (index 0) with appropriate time units
/// - Species data columns follow sequentially with proper indexing
/// - Empty time series are filtered out to avoid malformed data structures
/// - Each column includes species mapping, data type, and unit information
///
/// # Data Type Mapping
///
/// The conversion processes measurement data types and maps them to v1 column types,
/// ensuring proper interpretation of concentration, activity, or other measurement types.
///
/// # Errors
///
/// Returns `SBMLError::MissingTimeUnit` if time unit information cannot be extracted
/// from any species data in the measurement, which is required for proper data interpretation.
impl TryFrom<&Measurement> for FormatAnnot {
    type Error = SBMLError;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        // This is very ugly, but it works :')
        let time_unit = measurement
            .species_data
            .iter()
            .find(|data| data.time_unit.is_some())
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .time_unit
            .as_ref()
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .id
            .as_ref()
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .clone();

        let mut columns = vec![ColumnAnnot {
            species_id: None,
            column_type: ColumnType::Time,
            unit: time_unit,
            index: 0,
            replica: None,
            is_calculated: false,
        }];

        let mut index = 1;
        for data in measurement.species_data.iter() {
            if data.time.is_empty() {
                // If the time column is empty, we don't need to add it to the format
                continue;
            }

            let mut column = ColumnAnnot::try_from(data)?;
            column.index = index;
            index += 1;
            columns.push(column);
        }

        Ok(FormatAnnot {
            id: measurement.id.clone(),
            columns,
        })
    }
}

/// Converts MeasurementData to a ColumnAnnot for SBML v1 annotation
///
/// This implementation creates individual column definitions that describe how species
/// data is organized within CSV files. The column annotation includes all metadata
/// necessary for proper data interpretation and species mapping during model reconstruction.
///
/// # Column Definition
///
/// - Maps data types to appropriate v1 column types (concentration, activity, etc.)
/// - Processes unit definitions with SBML-compatible formatting
/// - Associates columns with specific species for proper data attribution
/// - Sets up indexing structure (indices are assigned during format creation)
///
/// # Unit Processing
///
/// Unit IDs undergo slash replacement to ensure compatibility with SBML unit definition
/// naming conventions, preventing parsing errors in SBML readers.
///
/// # Errors
///
/// Returns `SBMLError::MissingUnit` if the measurement data lacks required unit
/// definitions, which are essential for proper data interpretation and model validation.
impl TryFrom<&MeasurementData> for ColumnAnnot {
    type Error = SBMLError;

    fn try_from(measurement_data: &MeasurementData) -> Result<Self, Self::Error> {
        let column_type: ColumnType = measurement_data
            .data_type
            .as_ref()
            .unwrap_or(&DataTypes::Concentration)
            .into();

        let unit_id = &measurement_data
            .data_unit
            .as_ref()
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?;

        let unit_id = unit_id
            .id
            .as_ref()
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?;

        Ok(ColumnAnnot {
            species_id: Some(measurement_data.species_id.clone()),
            column_type,
            unit: replace_slashes(unit_id),
            index: 0, // Indices will be set later
            replica: None,
            is_calculated: false,
        })
    }
}
