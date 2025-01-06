//! This module provides functionality for handling initial parameter values in optimization.

use std::collections::HashMap;

use crate::prelude::EnzymeMLDocument;
use ndarray::Array1;

use super::error::OptimizeError;

/// Type alias for an array of initial parameter values
pub type Initials = Array1<f64>;

/// Prepares a sorted array of initial parameter values from a HashMap of parameter IDs and values.
///
/// This function takes a HashMap mapping parameter IDs to their initial values and returns them as
/// a sorted Array1. The sorting is done by parameter ID to ensure consistent ordering.
///
/// # Arguments
/// * `initials` - HashMap mapping parameter IDs to their initial values
///
/// # Returns
/// * `Initials` - Sorted array of initial values
pub fn prepare_initials(initials: HashMap<String, f64>) -> Initials {
    let mut param_ids = initials.keys().collect::<Vec<_>>();
    param_ids.sort();

    Array1::from_vec(
        param_ids
            .into_iter()
            .map(|id| initials.get(id).unwrap())
            .cloned()
            .collect(),
    )
}

impl TryFrom<&EnzymeMLDocument> for Initials {
    type Error = OptimizeError;

    /// Attempts to extract initial parameter values from an EnzymeMLDocument.
    ///
    /// This function collects all initial parameter values from the document and returns them
    /// as a sorted array. If any parameters are missing initial values, an error is returned.
    ///
    /// # Arguments
    /// * `doc` - Reference to the EnzymeMLDocument to extract initial values from
    ///
    /// # Returns
    /// * `Result<Initials, InitialsError>` - Array of initial values if successful, error if any are missing
    ///
    /// # Errors
    /// Returns `InitialsError::MissingInitialValues` if any parameters lack initial values
    fn try_from(doc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        let mut parameters = HashMap::new();
        let mut missing_parameters = Vec::new();
        for param in doc.parameters.iter() {
            if let Some(initial_value) = param.initial_value {
                parameters.insert(param.name.clone(), initial_value);
            } else {
                missing_parameters.push(param.name.clone());
            }
        }

        if !missing_parameters.is_empty() {
            return Err(OptimizeError::MissingInitialValues {
                missing: missing_parameters,
            });
        }

        let mut param_ids = parameters.keys().collect::<Vec<_>>();
        param_ids.sort();

        let param_vec = Array1::from_vec(
            param_ids
                .into_iter()
                .map(|id| parameters.get(id).unwrap())
                .cloned()
                .collect(),
        );

        Ok(param_vec)
    }
}
