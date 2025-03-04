//! Initial Conditions Module for ODE Simulations
//!
//! This module provides functionality for extracting and managing initial conditions
//! in ODE simulations, particularly from measurement data.
//!
//! # Key Components
//!
//! - [`InitialCondition`]: A type alias for a HashMap mapping species IDs to their initial values
//! - [`extract_initial_conditions()`]: Function to extract initial conditions from measurements
//! - `From<&Measurement>` implementation: Converts measurement data to initial conditions
//!
//! # Purpose
//!
//! The module helps in:
//! - Converting measurement data into initial conditions for ODE systems
//! - Providing a flexible way to specify initial species concentrations
//! - Supporting multiple initial condition sets from different measurements
//!
//! # Usage
//!
//! Initial conditions can be extracted from measurement data and used to set up
//! the starting state of an ODE simulation.

use crate::prelude::{EnzymeMLDocument, Measurement};
use std::collections::HashMap;

pub type InitialCondition = HashMap<String, f64>;

/// Extracts initial conditions from a slice of measurements.
///
/// This function takes a slice of `Measurement` objects and converts each measurement
/// into its corresponding initial condition represented as a `HashMap`. The resulting
/// vector contains the initial conditions for all provided measurements.
///
/// # Arguments
///
/// * `measurements` - A slice of `Measurement` objects from which to extract initial conditions.
///
/// # Returns
///
/// A vector of `InitialConditionType`, where each element is a `HashMap` mapping species IDs
/// to their initial values.
pub fn extract_initial_conditions(measurements: &[Measurement]) -> Vec<InitialCondition> {
    measurements.iter().map(|m| m.into()).collect()
}

impl From<&Measurement> for InitialCondition {
    /// Converts a `Measurement` into an initial condition represented as a `HashMap`.
    ///
    /// This function extracts the species IDs and their corresponding initial values
    /// from the provided `Measurement` object and returns them as a `HashMap`.
    ///
    /// # Arguments
    ///
    /// * `measurement` - A reference to the `Measurement` object from which to extract
    ///   initial conditions.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping species IDs to their initial values.
    fn from(measurement: &Measurement) -> Self {
        HashMap::from_iter(
            measurement
                .species_data
                .iter()
                .map(|species| (species.species_id.clone(), species.initial)),
        )
    }
}

impl From<&EnzymeMLDocument> for Vec<InitialCondition> {
    /// Converts an `EnzymeMLDocument` into a vector of initial conditions.
    ///
    /// This function extracts all measurements from the provided `EnzymeMLDocument`
    /// and converts each measurement into its corresponding initial condition.
    ///
    /// # Arguments
    ///
    /// * `doc` - A reference to the `EnzymeMLDocument` from which to extract
    ///   initial conditions.
    ///
    /// # Returns
    ///
    /// A vector of `InitialCondition`, where each element is a `HashMap` mapping
    /// species IDs to their initial values.
    fn from(doc: &EnzymeMLDocument) -> Self {
        doc.get_measurements().iter().map(|m| m.into()).collect()
    }
}
