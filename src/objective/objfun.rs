use std::fmt::Display;

use ndarray::{Array1, Array2, Array3};

use super::error::ObjectiveError;

/// Defines an objective function for optimization problems
///
/// This trait provides methods to calculate both the cost/loss value and its gradient
/// for optimization algorithms. It is typically used in parameter estimation and model fitting.
pub trait ObjectiveFunction: Copy + Display {
    /// Calculates the cost/loss value for the current state
    ///
    /// # Arguments
    /// * `residuals` - 2D array of residuals (differences between predicted and actual values)
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// * `Result<f64, Box<dyn std::error::Error>>` - The calculated cost value or an error
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError>;

    // /// Calculates the gradient of the objective function
    // ///
    // /// # Arguments
    // /// * `residuals` - 2D array of residuals (differences between predicted and actual values)
    // /// * `sensitivities` - 3D array containing sensitivity information for gradient calculation
    // /// * `n_points` - Total number of data points for normalization
    // ///
    // /// # Returns
    // /// * `Result<Array1<f64>, Box<dyn std::error::Error>>` - The calculated gradient or an error
    // fn gradient(
    //     &self,
    //     residuals: Array2<f64>,
    //     sensitivities: &Array3<f64>,
    //     n_points: usize,
    // ) -> Result<Array1<f64>, ObjectiveError>;
}
