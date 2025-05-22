use std::{
    fmt::{self, Display},
    str::FromStr,
};

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
use ndarray::Array2;

use super::{error::ObjectiveError, objfun::ObjectiveFunction};

/// Enumeration of different loss functions available for optimization
///
/// Loss functions are used to measure the discrepancy between predicted and actual values
/// in machine learning and optimization problems. Each variant represents a different
/// approach to calculating the error or cost of a model's predictions.
#[derive(Debug, Clone, Copy, Default)]
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
    /// Negative Log Likelihood (NLL): Measures the negative log likelihood of observing the data given a model with Gaussian NLL
    NLL(NegativeLogLikelihood),
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
            LossFunction::NLL(nll) => nll.cost(residuals, n_points),
        }
    }
}

impl Display for LossFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LossFunction::SSE => write!(f, "Sum of Squared Errors"),
            LossFunction::MSE => write!(f, "Mean Squared Error"),
            LossFunction::RMSE => write!(f, "Root Mean Squared Error"),
            LossFunction::LogCosh => write!(f, "Log-Cosh Loss"),
            LossFunction::MAE => write!(f, "Mean Absolute Error"),
            LossFunction::NLL(sigma) => write!(f, "Negative Log Likelihood (sigma={})", sigma),
        }
    }
}

impl FromStr for LossFunction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use regex::Regex;

        let s = s.to_lowercase();

        // Simple string matches for basic cases
        match s.as_str() {
            "sse" => return Ok(LossFunction::SSE),
            "mse" => return Ok(LossFunction::MSE),
            "rmse" => return Ok(LossFunction::RMSE),
            "logcosh" => return Ok(LossFunction::LogCosh),
            "mae" => return Ok(LossFunction::MAE),
            _ => {}
        }

        // Use regex for complex pattern matching
        let nll_pattern = Regex::new(r"^nll\((\d+\.?\d*)\)$").map_err(|e| e.to_string())?;
        if let Some(captures) = nll_pattern.captures(&s) {
            let sigma = captures
                .get(1)
                .ok_or_else(|| "Failed to capture sigma value".to_string())?
                .as_str()
                .parse::<f64>()
                .map_err(|e| format!("Invalid sigma value: {}", e))?;
            return Ok(LossFunction::NLL(NegativeLogLikelihood::new(sigma)));
        }

        Err(format!("Invalid loss function: {}", s))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SumOfSquaredErrors;

impl ObjectiveFunction for SumOfSquaredErrors {
    fn cost(&self, residuals: &Array2<f64>, _: usize) -> Result<f64, ObjectiveError> {
        let squared_residuals = residuals.mapv(|r| r * r);
        Ok(squared_residuals.sum())
    }
}

impl Display for SumOfSquaredErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sum of Squared Errors")
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

#[derive(Debug, Clone, Copy)]
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
}

impl Display for MeanSquaredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mean Squared Error")
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
#[derive(Debug, Clone, Copy)]
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
}

impl Display for RootMeanSquaredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Root Mean Squared Error")
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
#[derive(Debug, Clone, Copy)]
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
}

impl Display for LogCosh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Log-Cosh Loss")
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
#[derive(Debug, Clone, Copy)]
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
}

impl Display for MeanAbsoluteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mean Absolute Error")
    }
}

/// Negative Log Likelihood Loss Function
///
/// This loss function represents the negative log likelihood of observing the data
/// given a model with Gaussian noise. It is commonly used in statistical modeling
/// and maximum likelihood estimation.
///
/// Key characteristics:
/// - Statistically motivated cost function
/// - Accounts for measurement uncertainty via sigma parameter
/// - Equivalent to MSE with additional normalization terms
/// - Suitable for parameter estimation with known measurement error
#[derive(Debug, Clone, Copy)]
pub struct NegativeLogLikelihood {
    /// 0.5 * ln(2π σ²)   (pre‑multiplied constant part per data point)
    log_norm_half: f64,
    /// 0.5 / σ²          (multiply with SSE)
    half_inv_sigma_sq: f64,
    sigma: f64,
}

impl NegativeLogLikelihood {
    /// Construct a new loss function for a *known* noise standard deviation σ.
    #[inline]
    pub fn new(sigma: f64) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "σ must be a positive finite number"
        );
        let sigma_sq = sigma * sigma;
        Self {
            log_norm_half: 0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln(),
            half_inv_sigma_sq: 0.5 / sigma_sq,
            sigma,
        }
    }

    /// Evaluate −log L for an arbitrary Nd array of residuals.
    ///
    /// Accepts `Array1` (`y - f`), a view, or a flattened `Array2`.
    #[inline]
    fn cost(&self, residuals: &Array2<f64>, n_points: usize) -> Result<f64, ObjectiveError> {
        let sse: f64 = residuals.mapv(|r| r * r).sum();
        Ok(n_points as f64 * self.log_norm_half + self.half_inv_sigma_sq * sse)
    }
}

impl Display for NegativeLogLikelihood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Negative Log Likelihood (sigma={})", self.sigma)
    }
}

impl From<NegativeLogLikelihood> for LossFunction {
    fn from(value: NegativeLogLikelihood) -> Self {
        LossFunction::NLL(value)
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
}
