//! EnzymeML Suite integration module
//!
//! This module provides functionality to interact with the EnzymeML Suite API,
//! allowing users to fetch EnzymeML documents from a running Suite instance.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{io::IOError, prelude::EnzymeMLDocument};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SuiteResponse<T> {
    data: T,
    status: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocumentResponse {
    title: String,
    content: EnzymeMLDocument,
}

/// Fetches an EnzymeML document from the EnzymeML Suite
///
/// This function connects to a local EnzymeML Suite instance and retrieves
/// the specified document by its ID. If no ID is provided, it fetches the
/// current document. If no base URL is provided, it defaults to the local
/// Suite instance running on port 13452.
///
/// # Arguments
///
/// * `id` - The unique identifier of the document to fetch from the Suite.
///          If `None` or empty, defaults to ":current" to fetch the current document.
/// * `base_url` - The base URL of the Suite instance. If `None`, defaults to
///                "http://127.0.0.1:13452".
///
/// # Returns
///
/// * `Ok(EnzymeMLDocument)` - The successfully fetched and parsed EnzymeML document
/// * `Err(SuiteError)` - An error occurred during the fetch or parsing process
///
/// # Errors
///
/// This function will return an error if:
/// - The Suite API is not accessible at the specified URL
/// - The HTTP request fails (network issues, timeouts, etc.)
/// - The document ID is not found in the Suite
/// - The response cannot be parsed as valid JSON
/// - The response indicates an error status (non-200)
/// - The response cannot be parsed as a valid EnzymeML document
pub fn fetch_document_from_suite(
    id: impl Into<Option<String>>,
    base_url: impl Into<Option<String>>,
) -> Result<EnzymeMLDocument, SuiteError> {
    let doc_id = id.into().unwrap_or_else(|| ":current".to_string());
    let base_url = base_url
        .into()
        .unwrap_or_else(|| format!("http://127.0.0.1:13452"));
    let url = format!("{base_url}/docs/{doc_id}");

    let response = reqwest::blocking::get(&url).map_err(SuiteError::RequestError)?;
    let body = response.text().map_err(SuiteError::RequestError)?;
    let suite_response: SuiteResponse<DocumentResponse> =
        serde_json::from_str(&body).map_err(SuiteError::JSONError)?;

    if suite_response.status != 200 {
        return Err(SuiteError::InvalidDocument(suite_response.data.title));
    }

    Ok(suite_response.data.content)
}

/// Errors that can occur when interacting with the EnzymeML Suite
#[derive(Error, Debug)]
pub enum SuiteError {
    /// An I/O error occurred while processing the document
    #[error("IO error: {0}")]
    IOError(IOError),
    /// A JSON parsing error occurred while deserializing the Suite response
    #[error("JSON error: {0}")]
    JSONError(serde_json::Error),
    /// The HTTP request to the Suite API failed
    #[error("Request failed: {0}")]
    RequestError(reqwest::Error),
    /// The document retrieved from the Suite is invalid or the request returned an error status
    #[error("Invalid document: {0}")]
    InvalidDocument(String),
}

#[cfg(test)]
mod tests {
    use httpmock::MockServer;

    use super::*;

    #[test]
    fn test_fetch_document_from_suite() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path(format!("/docs/:current").as_str());
            then.status(200).body(
                serde_json::to_string(&mock_suite_response())
                    .expect("Failed to serialize mock suite response"),
            );
        });

        let base_url = format!("http://{}", mock.server_address().to_string());
        let document =
            fetch_document_from_suite(None, base_url).expect("Failed to fetch document from suite");
        assert_eq!(document.name, "Test Document");

        mock.assert();
    }

    #[test]
    fn test_fetch_document_from_suite_invalid_address() {
        let base_url = format!("http://invalid");
        fetch_document_from_suite(None, base_url)
            .expect_err("Should have failed to fetch document from suite");
    }

    fn mock_suite_response() -> SuiteResponse<DocumentResponse> {
        SuiteResponse {
            data: DocumentResponse {
                title: "Test Document".to_string(),
                content: EnzymeMLDocument {
                    name: "Test Document".to_string(),
                    ..Default::default()
                },
            },
            status: 200,
        }
    }
}
