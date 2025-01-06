use ndarray::Array2;

use super::metrics::{
    huber_loss, mean_squared_error, root_mean_squared_error, sum_of_squared_errors,
};

/// Defines different objective functions that can be used for optimization
///
/// # Variants
/// * `MSE` - Mean Squared Error, penalizes larger errors more heavily
/// * `RMSE` - Root Mean Squared Error, like MSE but in same units as original data
/// * `Huber(f64)` - Huber loss with given delta parameter, combines MSE and absolute error
#[derive(Debug, Clone, Copy)]
pub enum ObjectiveFunction {
    MSE,
    RMSE,
    Huber(f64),
    SSE,
}

impl ObjectiveFunction {
    /// Calculates the cost/loss value using the selected objective function
    ///
    /// # Arguments
    /// * `residuals` - 2D array of residuals (differences between predicted and actual values)
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// * `Result<f64, argmin::core::Error>` - The calculated cost value or an error
    pub fn cost(
        &self,
        residuals: &Array2<f64>,
        n_points: usize,
    ) -> Result<f64, argmin::core::Error> {
        match self {
            ObjectiveFunction::MSE => Ok(mean_squared_error(residuals, n_points as f64)),
            ObjectiveFunction::RMSE => Ok(root_mean_squared_error(residuals, n_points as f64)),
            ObjectiveFunction::Huber(delta) => Ok(huber_loss(residuals, n_points as f64, *delta)),
            ObjectiveFunction::SSE => Ok(sum_of_squared_errors(residuals)),
        }
    }
}
