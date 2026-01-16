//! EnzymeML Suite integration module
//!
//! This module provides functionality to interact with the EnzymeML Suite API,
//! allowing users to fetch EnzymeML documents from a running Suite instance.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

use crate::{io::IOError, prelude::EnzymeMLDocument};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:13452";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseEntry {
    pub id: u64,
    pub title: String,
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
/// * `id` - The unique identifier of the document to fetch from the Suite. If `None` or empty, defaults to ":current" to fetch the current document.
/// * `base_url` - The base URL of the Suite instance. If `None`, defaults to "http://127.0.0.1:13452".
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
    id: impl Into<Option<u64>>,
    base_url: impl Into<Option<String>>,
) -> Result<EnzymeMLDocument, SuiteError> {
    let doc_id = id
        .into()
        .map(|id| id.to_string())
        .unwrap_or_else(|| ":current".to_string());
    let base_url_str = base_url
        .into()
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

    let base = Url::parse(&base_url_str).map_err(|e| {
        SuiteError::InvalidDocument(format!("Invalid base URL '{}': {}", base_url_str, e))
    })?;

    let url = base
        .join(&format!("docs/{}", doc_id))
        .map_err(|e| SuiteError::InvalidDocument(format!("Failed to join path: {}", e)))?;

    let response = reqwest::blocking::get(url.as_str()).map_err(SuiteError::RequestError)?;
    let body = response.text().map_err(SuiteError::RequestError)?;
    let suite_response: SuiteResponse<DocumentResponse> =
        serde_json::from_str(&body).map_err(SuiteError::JSONError)?;

    if suite_response.status != 200 {
        return Err(SuiteError::InvalidDocument(suite_response.data.title));
    }

    Ok(suite_response.data.content)
}

/// Lists all documents from the EnzymeML Suite
///
/// This function connects to a local EnzymeML Suite instance and lists all documents in the database. If no base URL is provided, it defaults to the local Suite instance running on port 13452.
///
/// # Arguments
///
/// * `base_url` - The base URL of the Suite instance. If `None`, defaults to "http://127.0.0.1:13452".
///
/// # Returns
///
/// * `Ok(Vec<DatabaseEntry>)` - The list of documents in the database
/// * `Err(SuiteError)` - An error occurred during the list operation
///
/// # Errors
///
/// This function will return an error if:
/// - The Suite API is not accessible at the specified URL
/// - The HTTP request fails (network issues, timeouts, etc.)
/// - The response cannot be parsed as valid JSON
/// - The response indicates an error status (non-200)
/// - The response cannot be parsed as a valid list of database entries
pub fn list_documents_from_suite(
    base_url: impl Into<Option<String>>,
) -> Result<Vec<DatabaseEntry>, SuiteError> {
    let base_url_str = base_url
        .into()
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

    let base = Url::parse(&base_url_str).map_err(|e| {
        SuiteError::InvalidDocument(format!("Invalid base URL '{}': {}", base_url_str, e))
    })?;

    let url = base
        .join("docs")
        .map_err(|e| SuiteError::InvalidDocument(format!("Failed to join path: {}", e)))?;

    let response = reqwest::blocking::get(url.as_str()).map_err(SuiteError::RequestError)?;
    let body = response.text().map_err(SuiteError::RequestError)?;
    let suite_response: SuiteResponse<Vec<DatabaseEntry>> =
        serde_json::from_str(&body).map_err(SuiteError::JSONError)?;

    Ok(suite_response
        .data
        .into_iter()
        .map(|entry| DatabaseEntry {
            id: entry.id,
            title: entry.title,
        })
        .collect())
}

/// Pushes an EnzymeML document to the EnzymeML Suite
///
/// This function connects to a local EnzymeML Suite instance and pushes
/// the specified document to the current document. If no base URL is provided,
/// it defaults to the local Suite instance running on port 13452.
///
/// # Arguments
///
/// * `document` - The EnzymeML document to push to the Suite.
/// * `base_url` - The base URL of the Suite instance. If `None`, defaults to "http://127.0.0.1:13452".
///
/// # Returns
///
/// * `Ok(())` - The document was successfully pushed to the Suite
pub fn push_document_to_suite(
    document: &EnzymeMLDocument,
    base_url: impl Into<Option<String>>,
) -> Result<(), SuiteError> {
    let base_url_str = base_url
        .into()
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

    let base = Url::parse(&base_url_str).map_err(|e| {
        SuiteError::InvalidDocument(format!("Invalid base URL '{}': {}", base_url_str, e))
    })?;

    let url = base
        .join("docs/:current")
        .map_err(|e| SuiteError::InvalidDocument(format!("Failed to join path: {}", e)))?;

    // Send PUT request to the Suite API to update the current document
    let client = reqwest::blocking::Client::new();
    let response = client
        .put(url.as_str())
        .json(document)
        .send()
        .map_err(SuiteError::RequestError)?;

    if response.status() != 200 {
        return Err(SuiteError::InvalidDocument(
            response.text().map_err(SuiteError::RequestError)?,
        ));
    }

    Ok(())
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

    use crate::prelude::Protein;

    use super::*;

    #[test]
    fn test_fetch_document_from_suite() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/docs/:current".to_string().as_str());
            then.status(200).body(
                serde_json::to_string(&mock_suite_response())
                    .expect("Failed to serialize mock suite response"),
            );
        });

        let base_url = format!("http://{}", mock.server_address());
        let document =
            fetch_document_from_suite(None, base_url).expect("Failed to fetch document from suite");
        assert_eq!(document.name, "Test Document");

        mock.assert();
    }

    #[test]
    fn test_fetch_document_from_suite_invalid_address() {
        let base_url = "http://invalid".to_string();
        fetch_document_from_suite(None, base_url)
            .expect_err("Should have failed to fetch document from suite");
    }

    #[test]
    fn test_push_document_to_suite() {
        let server = MockServer::start();
        let test_document = EnzymeMLDocument {
            name: "Test Push Document".to_string(),
            ..Default::default()
        };
        let expected_body =
            serde_json::to_string(&test_document).expect("Failed to serialize test document");

        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/docs/:current")
                .body(&expected_body);
            then.status(200);
        });

        let base_url = format!("http://{}", mock.server_address());
        push_document_to_suite(&test_document, base_url).expect("Failed to push document to suite");

        mock.assert();
    }

    #[test]
    fn test_live_push_document_to_suite() {
        let document = EnzymeMLDocument {
            name: "Test Push Document".to_string(),
            proteins: vec![Protein {
                id: "P00001".to_string(),
                name: "Protein 1".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        push_document_to_suite(&document, None).expect("Failed to push document to suite");
    }

    #[test]
    fn test_fetch_with_trailing_slash_base_url() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/docs/:current");
            then.status(200).body(
                serde_json::to_string(&mock_suite_response())
                    .expect("Failed to serialize mock suite response"),
            );
        });

        let base_url = format!("http://{}/", mock.server_address());
        let document =
            fetch_document_from_suite(None, base_url).expect("Failed to fetch document from suite");
        assert_eq!(document.name, "Test Document");

        mock.assert();
    }

    #[test]
    fn test_push_with_trailing_slash_base_url() {
        let server = MockServer::start();
        let test_document = EnzymeMLDocument {
            name: "Test Push Document".to_string(),
            ..Default::default()
        };
        let expected_body =
            serde_json::to_string(&test_document).expect("Failed to serialize test document");

        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/docs/:current")
                .body(&expected_body);
            then.status(200);
        });

        let base_url = format!("http://{}/", mock.server_address());
        push_document_to_suite(&test_document, base_url).expect("Failed to push document to suite");

        mock.assert();
    }

    #[test]
    fn test_list_documents_from_suite() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/docs");
            then.status(200).body(
                serde_json::to_string(&mock_suite_response_list())
                    .expect("Failed to serialize mock suite response"),
            );
        });

        let base_url = format!("http://{}", mock.server_address());
        let documents =
            list_documents_from_suite(base_url).expect("Failed to list documents from suite");
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].id, 1);
        assert_eq!(documents[0].title, "Test Document");

        mock.assert();
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

    fn mock_suite_response_list() -> SuiteResponse<Vec<DatabaseEntry>> {
        SuiteResponse {
            data: vec![DatabaseEntry {
                id: 1,
                title: "Test Document".to_string(),
            }],
            status: 200,
        }
    }
}
