//! LLM integration module for EnzymeML document generation
//!
//! This module provides functionality to interact with Large Language Models (LLMs)
//! to generate EnzymeML documents from natural language descriptions. It currently
//! supports OpenAI's GPT models through their API.
//!
//! # Key Components
//!
//! - `query_llm`: Main function to query the LLM and generate EnzymeML documents
//! - `PromptInput`: Enum for handling different types of input prompts
//! - `LLMError`: Error types specific to LLM operations
use std::path::PathBuf;

use mdmodels::{llm::extraction::query_openai, prelude::DataModel};
use serde_json::Value;
use thiserror::Error;

// Include the EnzymeML specification from markdown file
const SPECS: &str = include_str!("../specs/specifications/v2.md");

/// Default system prompt that instructs the LLM on its role and expected output format
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful scientific assistant that is capable to identify scientific facts and data from a given text. You are also capable of extracting information from a given text and returning it in a structured format. Please return the information in a JSON format. Think step by step and work precisely.";

/// Query an LLM to generate an EnzymeML document from a natural language description
///
/// # Arguments
///
/// * `prompt` - The main input prompt describing the enzyme kinetics experiment
/// * `system_prompt` - Optional custom system prompt to override the default
/// * `llm_model` - Optional specific LLM model to use (defaults to "gpt-4o")
/// * `api_key` - Optional API key for the LLM service (falls back to OPENAI_API_KEY env var)
///
/// # Returns
///
/// Returns a Result containing either the generated EnzymeML document or an error
///
/// # Errors
///
/// Can return various errors including:
/// - Environment variable not found
/// - File reading errors
/// - LLM service errors
/// - Data model parsing errors
/// - JSON serialization errors
pub fn query_llm(
    prompt: impl Into<PromptInput>,
    system_prompt: Option<impl Into<PromptInput>>,
    llm_model: Option<String>,
    api_key: Option<String>,
) -> Result<Value, LLMError> {
    let llm_model = llm_model.unwrap_or_else(|| "gpt-4o".to_string());
    let api_key = match api_key {
        Some(key) => key,
        None => std::env::var("OPENAI_API_KEY").map_err(LLMError::EnvError)?,
    };

    let prompt: String = prompt.into().try_into()?;
    let system: String = if let Some(system_prompt) = system_prompt {
        system_prompt.into().try_into()?
    } else {
        DEFAULT_SYSTEM_PROMPT.to_string()
    };

    let model = DataModel::from_markdown_string(SPECS)
        .map_err(|e| LLMError::DataModelError(e.to_string()))?;

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(query_openai(
            prompt.as_str(),
            system.as_str(),
            &model,
            "EnzymeMLDocument",
            &llm_model,
            false,
            Some(api_key),
        ))
        .map_err(LLMError::LLMServiceError)
}

/// Represents different types of input prompts that can be provided to the LLM
///
/// This enum allows for flexible input handling, accepting either direct strings
/// or file paths containing the prompt text.
#[derive(Debug)]
pub enum PromptInput {
    /// Path to a file containing the prompt text
    File(PathBuf),
    /// Direct string containing the prompt text
    String(String),
}

impl TryInto<String> for PromptInput {
    type Error = LLMError;

    fn try_into(self) -> Result<String, Self::Error> {
        match self {
            PromptInput::String(s) => Ok(s),
            PromptInput::File(path) => {
                Ok(std::fs::read_to_string(path).map_err(LLMError::FileError)?)
            }
        }
    }
}

impl From<String> for PromptInput {
    fn from(s: String) -> Self {
        PromptInput::String(s)
    }
}

impl From<&str> for PromptInput {
    fn from(s: &str) -> Self {
        PromptInput::String(s.to_string())
    }
}

impl From<PathBuf> for PromptInput {
    fn from(path: PathBuf) -> Self {
        PromptInput::File(path)
    }
}

/// Errors that can occur during LLM operations
///
/// This enum encompasses all possible error types that might occur when
/// interacting with the LLM service and processing its responses.
#[derive(Debug, Error)]
pub enum LLMError {
    /// Error when reading prompt from file
    #[error("File not found: {0}")]
    FileError(#[from] std::io::Error),
    /// Error when accessing environment variables
    #[error("Environment variable not found: {0}")]
    EnvError(#[from] std::env::VarError),
    /// Error from the LLM service itself
    #[error("LLM service error: {0}")]
    LLMServiceError(#[from] Box<dyn std::error::Error>),
    /// Error in the underlying data model
    #[error("LLM model error: {0}")]
    DataModelError(String),
    /// Error during JSON serialization/deserialization
    #[error("Serde error: {0}")]
    SerdeError(#[from] serde_json::Error),
}
