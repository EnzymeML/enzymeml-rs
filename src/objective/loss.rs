/// # Loss Functions Module
///
/// This module provides a comprehensive implementation of various loss functions
/// used in machine learning and optimization problems. Loss functions are critical
/// for measuring the performance and guiding the optimization of predictive models.
///
/// ## Supported Loss Functions
/// - Mean Squared Error (MSE)
/// - Root Mean Squared Error (RMSE)
/// - Log-Cosh Loss
/// - Mean Absolute Error (MAE)
///
/// ## Key Features
/// - Trait-based implementation of objective functions
/// - Flexible cost and gradient calculation
/// - Support for different error measurement strategies
///
/// ## Usage
/// Loss functions can be used to:
/// - Evaluate model performance
/// - Guide optimization algorithms
/// - Compute error metrics during training
///
/// ## Design
/// The module uses an enum-based approach with trait implementations,
/// allowing easy extension and runtime selection of loss functions.
use ndarray::{Array1, Array2, Array3, Axis};

use super::{error::ObjectiveError, objfun::ObjectiveFunction};

/// Enumeration of different loss functions available for optimization
///
/// Loss functions are used to measure the discrepancy between predicted and actual values
/// in machine learning and optimization problems. Each variant represents a different
/// approach to calculating the error or cost of a model's predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LossFunction {
    /// Sum of Squared Errors (SSE): Measures sum of squared differences between predictions and actual values
    SSE,
    /// Mean Squared Error (MSE): Measures average squared difference between predictions and actual values
    #[default]
    MSE,
    /// Root Mean Squared Error (RMSE): Square root of MSE, providing error in the original scale
    RMSE,
    /// Log-Cosh Loss: Smooth approximation of absolute error with better numerical stability
    LogCosh,
    /// Mean Absolute Error (MAE): Measures average absolute difference between predictions and actual values
    MAE,
}

impl ObjectiveFunction for LossFunction {
    /// Delegates cost calculation to the specific loss function implementation
    ///
    /// # Arguments
    /// * `residuals` - 2D array of differences between predicted and actual values
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Calculated cost value for the selected loss function
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        match self {
            LossFunction::SSE => SumOfSquaredErrors.cost(residuals, n_points),
            LossFunction::MSE => MeanSquaredError.cost(residuals, n_points),
            LossFunction::RMSE => RootMeanSquaredError.cost(residuals, n_points),
            LossFunction::LogCosh => LogCosh.cost(residuals, n_points),
            LossFunction::MAE => MeanAbsoluteError.cost(residuals, n_points),
        }
    }

    /// Delegates gradient calculation to the specific loss function implementation
    ///
    /// # Arguments
    /// * `residuals` - 2D array of differences between predicted and actual values
    /// * `sensitivities` - 3D array containing sensitivity information for gradient calculation
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Calculated gradient for the selected loss function
    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        n_points: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        match self {
            LossFunction::SSE => SumOfSquaredErrors.gradient(residuals, sensitivities, n_points),
            LossFunction::MSE => MeanSquaredError.gradient(residuals, sensitivities, n_points),
            LossFunction::RMSE => RootMeanSquaredError.gradient(residuals, sensitivities, n_points),
            LossFunction::LogCosh => LogCosh.gradient(residuals, sensitivities, n_points),
            LossFunction::MAE => MeanAbsoluteError.gradient(residuals, sensitivities, n_points),
        }
    }
}

pub struct SumOfSquaredErrors;

impl ObjectiveFunction for SumOfSquaredErrors {
    fn cost(&self, residuals: &Array2<f64>, _: usize) -> Result<f64, ObjectiveError> {
        let squared_residuals = residuals.mapv(|r| r * r);
        Ok(squared_residuals.sum())
    }

    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        _: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        let (n_timepoints, n_species, n_params) = sensitivities.dim();

        // Expand residuals to match sensitivities dimensions
        let expanded = residuals.insert_axis(Axis(2));
        let residuals_expanded = expanded
            .broadcast((n_timepoints, n_species, n_params))
            .ok_or(ObjectiveError::ShapeError)?;

        // Multiply residuals with sensitivities and sum
        let product = &residuals_expanded * sensitivities;
        let sum = product.sum_axis(Axis(1)).sum_axis(Axis(0));

        Ok(2.0 * sum)
    }
}

/// Mean Squared Error (MSE) Loss Function
///
/// MSE calculates the average of the squared differences between predicted and actual values.
/// It is sensitive to outliers and penalizes large errors more heavily due to squaring.
///
/// Key characteristics:
/// - Quadratic penalty for errors
/// - Differentiable and convex
/// - Heavily penalizes large prediction errors
/// - Commonly used in regression problems
pub struct MeanSquaredError;

impl ObjectiveFunction for MeanSquaredError {
    /// Calculates the Mean Squared Error cost
    ///
    /// # Algorithm
    /// 1. Square each residual
    /// 2. Sum the squared residuals
    /// 3. Divide by total number of points to get average
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Average squared error across all data points
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        // Square each residual element-wise
        let squared_residuals = residuals.mapv(|r| r * r);

        // Sum all squared residuals and normalize by number of points
        let sum_squared_residuals = squared_residuals.sum();
        let cost = sum_squared_residuals / n_points as f64;
        Ok(cost)
    }

    /// Calculates the gradient of Mean Squared Error
    ///
    /// # Algorithm
    /// 1. Expand residuals to match sensitivities dimensions
    /// 2. Multiply expanded residuals with sensitivities
    /// 3. Sum across species and timepoints
    /// 4. Scale by 2 and normalize by number of points
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `sensitivities` - 3D array of parameter sensitivities
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Gradient of MSE with respect to parameters
    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        n_points: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        let (n_timepoints, n_species, n_params) = sensitivities.dim();

        // Expand residuals to match sensitivities dimensions
        let expanded = residuals.insert_axis(Axis(2));
        let residuals_expanded = expanded
            .broadcast((n_timepoints, n_species, n_params))
            .ok_or(ObjectiveError::ShapeError)?;

        // Multiply residuals with sensitivities and sum
        let product = &residuals_expanded * sensitivities;
        let sum = product.sum_axis(Axis(1)).sum_axis(Axis(0));

        // Scale by 2 and normalize by number of points
        Ok((2.0 * sum) / n_points as f64)
    }
}

/// Root Mean Squared Error (RMSE) Loss Function
///
/// RMSE is the square root of MSE, providing error in the original scale of the target variable.
/// It offers similar properties to MSE but with a more interpretable magnitude.
///
/// Key characteristics:
/// - Provides error in original units
/// - More interpretable than MSE
/// - Still sensitive to outliers
/// - Commonly used in regression evaluation
pub struct RootMeanSquaredError;

impl ObjectiveFunction for RootMeanSquaredError {
    /// Calculates the Root Mean Squared Error cost
    ///
    /// # Algorithm
    /// 1. Calculate MSE
    /// 2. Take square root of MSE
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Square root of mean squared error
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        let mse = MeanSquaredError;
        let cost = mse.cost(residuals, n_points)?;
        Ok(cost.sqrt())
    }

    /// Calculates the gradient of Root Mean Squared Error
    ///
    /// # Algorithm
    /// 1. Calculate MSE cost
    /// 2. Calculate MSE gradient
    /// 3. Divide gradient by MSE cost
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `sensitivities` - 3D array of parameter sensitivities
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Gradient of RMSE with respect to parameters
    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        n_points: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        let mse = MeanSquaredError;
        let cost = mse.cost(&residuals, n_points)?;
        let gradient = mse.gradient(residuals, sensitivities, n_points)?;
        Ok(gradient / cost)
    }
}

/// Log-Cosh Loss Function
///
/// Log-Cosh provides a smooth approximation of absolute error with better numerical stability.
/// It combines benefits of MSE and MAE, being less sensitive to outliers than MSE.
///
/// Key characteristics:
/// - Smooth and differentiable
/// - Less sensitive to outliers compared to MSE
/// - Computationally stable
/// - Provides a balance between quadratic and absolute error
pub struct LogCosh;

impl ObjectiveFunction for LogCosh {
    /// Calculates the Log-Cosh loss
    ///
    /// # Algorithm
    /// 1. Calculate hyperbolic cosine of each residual
    /// 2. Take natural logarithm of cosh values
    /// 3. Sum and normalize by number of points
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Average log-cosh of residuals
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        let log_cosh = residuals.mapv(|r| r.cosh().ln());
        Ok(log_cosh.sum() / n_points as f64)
    }

    /// Calculates the gradient of Log-Cosh loss
    ///
    /// # Algorithm
    /// 1. Calculate hyperbolic tangent of residuals
    /// 2. Expand tanh values to match sensitivities
    /// 3. Multiply expanded tanh with sensitivities
    /// 4. Sum and normalize by number of points
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `sensitivities` - 3D array of parameter sensitivities
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Gradient of log-cosh loss with respect to parameters
    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        n_points: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        // Extract the dimensions of the sensitivities array
        let (n_timepoints, n_species, n_params) = sensitivities.dim();

        // Get the hyperbolic tangent of the residuals
        let tanh = residuals.mapv(|r| r.tanh());

        // Expand the tanh array to match the dimensions of the sensitivities
        let expanded = tanh.insert_axis(Axis(2));
        let tanh_expanded = expanded
            .broadcast((n_timepoints, n_species, n_params))
            .ok_or(ObjectiveError::ShapeError)?;

        // Multiply the expanded tanh array with the sensitivities
        let product = &tanh_expanded * sensitivities;
        let sum = product.sum_axis(Axis(1)).sum_axis(Axis(0));
        Ok(sum / n_points as f64)
    }
}

/// Mean Absolute Error (MAE) Loss Function
///
/// MAE calculates the average of absolute differences between predicted and actual values.
/// It is less sensitive to outliers compared to MSE and provides a more robust error metric.
///
/// Key characteristics:
/// - Linear penalty for errors
/// - More robust to outliers
/// - Less aggressive penalty for large errors
/// - Suitable for scenarios with significant outliers
pub struct MeanAbsoluteError;

impl ObjectiveFunction for MeanAbsoluteError {
    /// Calculates the Mean Absolute Error cost
    ///
    /// # Algorithm
    /// 1. Take absolute value of each residual
    /// 2. Sum absolute residuals
    /// 3. Divide by total number of points
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Average absolute error across all data points
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        let abs_residuals = residuals.mapv(|r| r.abs());
        let sum = abs_residuals.sum();
        Ok(sum / n_points as f64)
    }

    /// Calculates the gradient of Mean Absolute Error
    ///
    /// # Algorithm
    /// 1. Calculate sign of each residual
    /// 2. Expand sign array to match sensitivities
    /// 3. Multiply expanded signs with sensitivities
    /// 4. Sum and normalize by number of points
    ///
    /// # Arguments
    /// * `residuals` - 2D array of prediction errors
    /// * `sensitivities` - 3D array of parameter sensitivities
    /// * `n_points` - Total number of data points for normalization
    ///
    /// # Returns
    /// Gradient of MAE with respect to parameters
    fn gradient(
        &self,
        residuals: Array2<f64>,
        sensitivities: &Array3<f64>,
        n_points: usize,
    ) -> Result<Array1<f64>, ObjectiveError> {
        let (n_timepoints, n_species, n_params) = sensitivities.dim();

        // Calculate sign of each residual
        let sign_residuals = residuals.mapv(|r| r.signum());

        // Expand sign array to match sensitivities dimensions
        let expanded = sign_residuals.insert_axis(Axis(2));
        let sign_residuals_expanded = expanded
            .broadcast((n_timepoints, n_species, n_params))
            .ok_or(ObjectiveError::ShapeError)?;

        // Multiply signs with sensitivities and sum
        let product = &sign_residuals_expanded * sensitivities;
        let sum = product.sum_axis(Axis(1)).sum_axis(Axis(0));
        Ok(sum / n_points as f64)
    }
}

// Existing test module remains unchanged
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean_squared_error_cost() {
        // Create test data
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n_points = 4;

        // Test cost calculation
        let mse = MeanSquaredError;
        let cost = mse.cost(&residuals, n_points).unwrap();

        // Calculation breakdown:
        // 1. Square each residual: [1², 2², 3², 4²] = [1, 4, 9, 16]
        // 2. Sum the squared residuals: 1 + 4 + 9 + 16 = 30
        // 3. Divide by number of points: 30 / 4 = 7.5
        assert_relative_eq!(cost, 7.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_squared_error_gradient() {
        // Create test data
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n_points = 4;
        let mse = MeanSquaredError;

        let sensitivities =
            Array3::from_shape_vec((2, 2, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap();

        let gradient = mse
            .gradient(residuals.clone(), &sensitivities, n_points)
            .unwrap();

        // Gradient calculation breakdown:
        // For param 1:
        //   2 * (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5 + 4.0 * 0.7) / 4
        // For param 2:
        //   2 * (1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6 + 4.0 * 0.8) / 4
        let expected = Array1::from_vec(vec![
            2.0 * (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5 + 4.0 * 0.7) / 4.0,
            2.0 * (1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6 + 4.0 * 0.8) / 4.0,
        ]);

        assert_relative_eq!(gradient[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(gradient[1], expected[1], epsilon = 1e-10);
    }

    #[test]
    fn test_log_cosh_cost() {
        // Create test data
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n_points = 4;

        let log_cosh = LogCosh;
        let cost = log_cosh.cost(&residuals, n_points).unwrap();

        // Calculation breakdown:
        // 1. Calculate cosh for each residual:
        //   cosh(1), cosh(2), cosh(3), cosh(4)
        // 2. Take natural log of each cosh value
        // 3. Sum the log(cosh) values
        // 4. Divide by number of points
        let expected =
            (1.0_f64.cosh().ln() + 2.0_f64.cosh().ln() + 3.0_f64.cosh().ln() + 4.0_f64.cosh().ln())
                / 4.0;
        assert_relative_eq!(cost, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_cosh_gradient() {
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n_points = 4;
        let sensitivities =
            Array3::from_shape_vec((2, 2, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap();

        let log_cosh = LogCosh;
        let gradient = log_cosh
            .gradient(residuals.clone(), &sensitivities, n_points)
            .unwrap();

        // Gradient calculation breakdown:
        // 1. Calculate tanh for each residual
        // 2. Multiply tanh values with corresponding sensitivities
        // 3. Sum the products for each parameter
        // 4. Divide by number of points
        let expected = {
            let tanh_vals = residuals.mapv(|r| r.tanh());
            let mut grad = vec![0.0; 2];
            for i in 0..2 {
                for j in 0..2 {
                    grad[0] += tanh_vals[[i, j]] * sensitivities[[i, j, 0]];
                    grad[1] += tanh_vals[[i, j]] * sensitivities[[i, j, 1]];
                }
            }
            Array1::from_vec(grad.iter().map(|&x| x / n_points as f64).collect())
        };

        assert_relative_eq!(gradient[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(gradient[1], expected[1], epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_error_cost() {
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let n_points = 4;

        let mae = MeanAbsoluteError;
        let cost = mae.cost(&residuals, n_points).unwrap();

        // Calculation breakdown:
        // 1. Take absolute value of each residual:
        //   |1| = 1, |-2| = 2, |3| = 3, |-4| = 4
        // 2. Sum the absolute values: 1 + 2 + 3 + 4 = 10
        // 3. Divide by number of points: 10 / 4 = 2.5
        assert_relative_eq!(cost, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_absolute_error_gradient() {
        let residuals = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let n_points = 4;
        let sensitivities =
            Array3::from_shape_vec((2, 2, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap();

        let mae = MeanAbsoluteError;
        let gradient = mae
            .gradient(residuals.clone(), &sensitivities, n_points)
            .unwrap();

        // Gradient calculation breakdown:
        // For param 1:
        //   (sign(1.0) * 0.1 + sign(-2.0) * 0.3 + sign(3.0) * 0.5 + sign(-4.0) * 0.7) / 4
        // For param 2:
        //   (sign(1.0) * 0.2 + sign(-2.0) * 0.4 + sign(3.0) * 0.6 + sign(-4.0) * 0.8) / 4
        let expected = Array1::from_vec(vec![
            (0.1 - 0.3 + 0.5 - 0.7) / 4.0,
            (0.2 - 0.4 + 0.6 - 0.8) / 4.0,
        ]);

        assert_relative_eq!(gradient[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(gradient[1], expected[1], epsilon = 1e-10);
    }
}
