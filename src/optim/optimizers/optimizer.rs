//! Traits and conversion logic for optimization.
//!
//! This module provides core traits and conversion functionality for optimization,
//! including:
//!
//! - The `Optimizer` trait defining the interface for optimization algorithms
//! - Initial guess handling via `InitialGuesses` and conversion from EnzymeML documents

use ndarray::Array1;

use crate::{optim::problem::Problem, prelude::EnzymeMLDocument};

use crate::optim::error::OptimizeError;

/// Trait defining the interface for optimization algorithms.
pub trait Optimizer {
    /// Optimizes the given problem to find optimal parameters.
    ///
    /// # Arguments
    /// * `problem` - The optimization problem to solve
    /// * `initial_guess` - Initial parameter values to start optimization from
    ///
    /// # Returns
    /// * `Result<Array1<f64>, OptimizeError>` - The optimal parameters or an error
    fn optimize<T>(
        &self,
        problem: &Problem,
        initial_guess: Option<T>,
    ) -> Result<Array1<f64>, OptimizeError>
    where
        T: Into<InitialGuesses>;
}

/// Wrapper type for initial parameter guesses used in optimization.
pub struct InitialGuesses(pub Array1<f64>);

impl InitialGuesses {
    pub fn get_values(self) -> Array1<f64> {
        self.0
    }
}

impl TryInto<InitialGuesses> for &EnzymeMLDocument {
    type Error = OptimizeError;

    /// Attempts to extract initial parameter guesses from an EnzymeML document.
    ///
    /// # Errors
    /// Returns `OptimizeError::MissingInitialGuesses` if any parameters lack initial values.
    fn try_into(self) -> Result<InitialGuesses, Self::Error> {
        // First collect parameters with missing initial values
        let missing = self
            .parameters
            .iter()
            .filter(|p| p.initial_value.is_none())
            .map(|p| p.name.clone())
            .collect::<Vec<_>>();

        // Return early with error if any parameters are missing initial values
        if !missing.is_empty() {
            return Err(OptimizeError::MissingInitialGuesses { missing });
        }

        // Extract initial values, using unwrap() since we verified they exist
        let values = self
            .parameters
            .iter()
            .filter_map(|p| p.initial_value)
            .collect::<Vec<_>>();

        // Convert to ndarray and wrap in InitialGuesses
        Ok(InitialGuesses(Array1::from_vec(values)))
    }
}

impl From<Array1<f64>> for InitialGuesses {
    fn from(value: Array1<f64>) -> Self {
        InitialGuesses(value)
    }
}
