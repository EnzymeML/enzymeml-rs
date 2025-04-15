//! Traits and conversion logic for optimization.
//!
//! This module provides core traits and conversion functionality for optimization,
//! including:
//!
//! - The `Optimizer` trait defining the interface for optimization algorithms
//! - Initial guess handling via `InitialGuesses` and conversion from EnzymeML documents

use ndarray::Array1;
use peroxide::fuga::ODEIntegrator;

use crate::optim::report::OptimizationReport;
use crate::prelude::ObjectiveFunction;
use crate::{optim::problem::Problem, prelude::EnzymeMLDocument};

use crate::optim::error::OptimizeError;

/// Trait defining the interface for optimization algorithms.
pub trait Optimizer<S: ODEIntegrator + Copy, L: ObjectiveFunction> {
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
        problem: &Problem<S, L>,
        initial_guess: Option<T>,
    ) -> Result<OptimizationReport, OptimizeError>
    where
        T: Into<InitialGuesses>;
}

/// Wrapper type for initial parameter guesses used in optimization.
#[derive(Debug, Clone)]
pub struct InitialGuesses(pub Array1<f64>);

impl InitialGuesses {
    /// Get the values of the initial guesses.
    ///
    /// # Returns
    /// * `Array1<f64>` - The values of the initial guesses
    pub fn get_values(self) -> Array1<f64> {
        self.0
    }

    /// Get the values of the initial guesses.
    ///
    /// # Returns
    /// * `&Array1<f64>` - The values of the initial guesses
    pub fn get_values_ref(&self) -> &Array1<f64> {
        &self.0
    }

    /// Set the value of the initial guess at a specific index.
    ///
    /// # Arguments
    /// * `index` - The index of the value to set
    /// * `value` - The value to set the initial guess to
    ///
    /// # Returns
    /// * `&mut Self` - The initial guesses with the updated value
    pub fn set_value_at(&mut self, index: usize, value: f64) -> &mut Self {
        self.0[index] = value;
        self
    }

    /// Get the value of the initial guess at a specific index.
    ///
    /// # Arguments
    /// * `index` - The index of the value to get
    ///
    /// # Returns
    /// * `f64` - The value of the initial guess at the specified index
    pub fn get_value_at(&self, index: usize) -> f64 {
        self.0[index]
    }

    /// Get the length of the initial guesses.
    ///
    /// # Returns
    /// * `usize` - The length of the initial guesses
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.len()
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

        // Extract all the exsitsing parameters and sort the names
        let mut param_order = self
            .parameters
            .iter()
            .map(|p| p.symbol.clone())
            .collect::<Vec<_>>();

        param_order.sort();

        let mut values = vec![0.0; param_order.len()];

        for (idx, name) in param_order.iter().enumerate() {
            let param = self.parameters.iter().find(|p| p.symbol == *name).ok_or(
                OptimizeError::ParameterNotFound {
                    param: name.clone(),
                    message: format!("Parameter {} not found", name),
                },
            )?;

            values[idx] = param.initial_value.unwrap();
        }

        // Convert to ndarray and wrap in InitialGuesses
        Ok(InitialGuesses(Array1::from_vec(values)))
    }
}

impl From<Array1<f64>> for InitialGuesses {
    fn from(value: Array1<f64>) -> Self {
        InitialGuesses(value)
    }
}
