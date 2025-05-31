//! SBML Reading Module
//!
//! This module provides functions for reading SBML (Systems Biology Markup Language) documents
//! from COMBINE archives. It handles the extraction of SBML models from OMEX files and parsing
//! them into usable SBML document objects.

use std::path::PathBuf;

use sbml::{combine::KnownFormats, prelude::CombineArchive, reader::SBMLReader, SBMLDocument};

use crate::prelude::EnzymeMLDocument;

use super::{error::SBMLError, v1::parser::parse_v1_omex};

pub fn from_omex(path: &PathBuf) -> Result<EnzymeMLDocument, SBMLError> {
    let archive = read_omex_file(path)?;
    parse_v1_omex(archive)
}

/// Reads a COMBINE archive from a file path.
///
/// This function opens a COMBINE archive (OMEX file) from the specified path and returns
/// a handle to the archive for further processing.
///
/// # Arguments
///
/// * `path` - Path to the OMEX file to be opened
///
/// # Returns
///
/// A Result containing either the opened CombineArchive or an SBMLError
pub(crate) fn read_omex_file(path: &PathBuf) -> Result<CombineArchive, SBMLError> {
    CombineArchive::open(path).map_err(SBMLError::CombineArchiveError)
}

/// Extracts and parses an SBML document from a COMBINE archive.
///
/// This function searches for an SBML file in the provided archive, extracts it,
/// and parses it into an SBMLDocument object that can be used for further processing.
///
/// # Arguments
///
/// * `archive` - A mutable reference to a CombineArchive from which to extract the SBML file
///
/// # Returns
///
/// A Result containing either the parsed SBMLDocument or an SBMLError
pub(crate) fn read_sbml_file(archive: &mut CombineArchive) -> Result<SBMLDocument, SBMLError> {
    let sbml_file = archive.entry_by_format(KnownFormats::SBML)?;
    let xml = sbml_file.as_string().map_err(SBMLError::SBMLReaderError)?;
    Ok(SBMLReader::from_xml_string(&xml))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_sbml_file() {
        let path = PathBuf::from("tests/data/enzymeml_v1.omex");
        let mut archive = read_omex_file(&path).unwrap();
        let sbml = read_sbml_file(&mut archive).unwrap();

        println!("{}", sbml.to_xml_string());
    }
}
