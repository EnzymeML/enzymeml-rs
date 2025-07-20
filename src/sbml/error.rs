use std::string::FromUtf8Error;

use polars::error::PolarsError;
use sbml::combine::error::CombineArchiveError;
use thiserror::Error;

use crate::sbml::speciestype::SpeciesType;

/// Errors that can occur during SBML parsing, serialization, and conversion
#[derive(Debug, Error)]
pub enum SBMLError {
    /// Error when reading an SBML file fails
    #[error("Failed to read SBML file: {0}")]
    ReadError(#[from] std::io::Error),

    /// Error when an invalid SBOTerm is encountered
    #[error("Invalid SBOTerm: {0}")]
    InvalidSBOTerm(String),

    /// Error when an invalid species type is encountered during conversion
    #[error("Invalid species type: {0}")]
    InvalidSpeciesType(SpeciesType),

    /// Error when the SBML document doesn't contain a model
    #[error("Cannot convert SBML document to EnzymeMLDocument: missing model")]
    MissingModel,

    /// Error when an invalid unit kind is encountered
    #[error("Invalid unit kind: {0}")]
    InvalidUnitKind(String),

    /// Error when a required unit definition is missing
    #[error("Cannot convert SBML document to EnzymeMLDocument: missing unit definition")]
    MissingUnitDefinition,

    /// Error when an unknown rule type is encountered
    #[error("Unknown rule type: {0}")]
    UnknownRuleType(Box<dyn std::error::Error>),

    /// Error when opening a COMBINE archive fails
    #[error("Failed to open COMBINE archive: {0}")]
    CombineArchiveError(#[from] CombineArchiveError),

    /// Error when reading an SBML file as UTF-8 fails
    #[error("Failed to read SBML file: {0}")]
    SBMLReaderError(#[from] FromUtf8Error),

    /// Error when a required file is missing from the archive
    #[error("Missing file: {0}")]
    MissingFile(String),

    /// Error when a required format is missing
    #[error("Missing format: {0}")]
    MissingFormat(String),

    /// Error when parsing a CSV file fails
    #[error("Failed to parse CSV: {0}")]
    PolarsError(#[from] PolarsError),

    /// Error when the time column is missing in a format
    #[error("Missing time column in format {0}")]
    MissingTimeColumn(String),

    /// Error when a species ID is missing for a column
    #[error("Missing species id in format {0} for column {1}")]
    MissingColumnSpeciesId(String, usize),

    /// Error when a species ID is missing in a format
    #[error("Missing species id in format {0}")]
    MissingSpeciesId(String),

    /// Error when an initial value is missing for a species
    #[error("Missing initial value for species {0}")]
    MissingInitialValue(String),

    /// Error when a unit with a specific SID is missing
    #[error("Missing unit with SID: {0}")]
    MissingUnit(String),

    /// Error when a unit definition ID is missing
    #[error("Missing unit definition id")]
    MissingUnitDefinitionId(String),

    /// Error when a required column is missing in a format
    #[error("Missing column {0} in format {1}")]
    MissingColumn(String, String),

    /// Error when a time unit is missing in a measurement
    #[error("Missing time unit in measurement {0}")]
    MissingTimeUnit(String),

    /// Error when a time column is missing in a measurement's DataFrame
    #[error("Cannot convert measurement to DataFrame: missing time column in measurement {0}")]
    MissingTimeColumnInDataFrame(String),

    /// Error when a data annotation is missing
    #[error("Missing data annotation")]
    MissingDataAnnotation,

    /// Error when serializing an annotation fails
    #[error("Failed to serialize annotation: {0}")]
    SerializeError(#[from] quick_xml::SeError),

    /// Error when deserializing an annotation fails
    #[error("Failed to deserialize annotation: {0}")]
    DeserializeError(#[from] quick_xml::DeError),

    /// No existing annotation found
    #[error("No existing annotation found for {0}")]
    NoExistingAnnotation(String),
}
