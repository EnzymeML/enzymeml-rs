use std::path::PathBuf;

use thiserror::Error;

use crate::prelude::EnzymeMLDocument;

/// Loads and parses an EnzymeML document from a JSON file.
///
/// This function attempts to read an EnzymeML document stored in JSON format from the specified file path.
/// It handles both file system operations and JSON parsing, returning either the parsed document or an error.
///
/// # Arguments
///
/// * `path` - A string slice containing the path to the JSON file containing the EnzymeML document
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Ok(EnzymeMLDocument)` - The successfully parsed EnzymeML document
/// * `Err(IOError)` - An error that occurred during file reading or JSON parsing
///
/// # Errors
///
/// This function will return an error if:
/// * The file cannot be found or opened (`IOError::FileNotFound`)
/// * The file contents cannot be parsed as valid JSON (`IOError::JsonParseError`)
/// * The JSON structure does not match the expected EnzymeML document format
pub fn load_enzmldoc(path: impl Into<PathBuf>) -> Result<EnzymeMLDocument, IOError> {
    let path = path.into();
    let file = std::fs::File::open(path).map_err(IOError::FileNotFound)?;
    serde_json::from_reader(file).map_err(IOError::JsonParseError)
}

/// Saves an EnzymeML document to a JSON file.
///
/// This function attempts to save an EnzymeML document to a JSON file at the specified path.
/// It handles both file system operations and JSON serialization, returning either the saved document or an error.
///
/// # Arguments
///
/// * `path` - A string slice containing the path to the JSON file to save the EnzymeML document to
/// * `doc` - A reference to the EnzymeML document to save
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `Ok(())` - The successfully saved EnzymeML document
/// * `Err(IOError)` - An error that occurred during file writing or JSON serialization
pub fn save_enzmldoc(path: impl Into<PathBuf>, doc: &EnzymeMLDocument) -> Result<(), IOError> {
    let path = path.into();
    let file = std::fs::File::create(path).map_err(IOError::FileNotFound)?;
    serde_json::to_writer_pretty(file, doc).map_err(IOError::JsonParseError)
}

/// Represents errors that can occur during EnzymeML document I/O operations.
///
/// This enum encapsulates the various error conditions that may arise when reading
/// and parsing EnzymeML documents from files.
#[derive(Error, Debug)]
pub enum IOError {
    /// Indicates that the specified file could not be found or opened.
    ///
    /// This variant wraps the underlying std::io::Error that provides more details
    /// about the specific file system error that occurred.
    #[error("File not found: {0}")]
    FileNotFound(#[from] std::io::Error),

    /// Indicates that the file contents could not be parsed as valid JSON.
    ///
    /// This variant wraps the underlying serde_json::Error that provides more details
    /// about the specific JSON parsing error that occurred.
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),
}
