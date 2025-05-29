use std::string::FromUtf8Error;

use polars::error::PolarsError;
use quick_xml::DeError;
use sbml::combine::error::CombineArchiveError;
use thiserror::Error;

use crate::sbml::speciestype::SpeciesType;

#[derive(Debug, Error)]
pub enum SBMLError {
    #[error("Failed to read SBML file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Invalid SBOTerm: {0}")]
    InvalidSBOTerm(String),
    #[error("Failed to deserialize annotation: {0}")]
    DeserializeError(#[from] DeError),
    #[error("Invalid species type: {0}")]
    InvalidSpeciesType(SpeciesType),
    #[error("Cannot convert SBML document to EnzymeMLDocument: missing model")]
    MissingModel,
    #[error("Invalid unit kind: {0}")]
    InvalidUnitKind(String),
    #[error("Cannot convert SBML document to EnzymeMLDocument: missing unit definition")]
    MissingUnitDefinition,
    #[error("Unknown rule type: {0}")]
    UnknownRuleType(Box<dyn std::error::Error>),
    #[error("Failed to open COMBINE archive: {0}")]
    CombineArchiveError(#[from] CombineArchiveError),
    #[error("Failed to read SBML file: {0}")]
    SBMLReaderError(#[from] FromUtf8Error),
    #[error("Missing file: {0}")]
    MissingFile(String),
    #[error("Missing format: {0}")]
    MissingFormat(String),
    #[error("Failed to parse CSV: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("Missing time column in format {0}")]
    MissingTimeColumn(String),
    #[error("Missing species id in format {0} for column {1}")]
    MissingColumnSpeciesId(String, usize),
    #[error("Missing species id in format {0}")]
    MissingSpeciesId(String),
    #[error("Missing initial value for species {0}")]
    MissingInitialValue(String),
    #[error("Missing unit with SID: {0}")]
    MissingUnit(String),
}
