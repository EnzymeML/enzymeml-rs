use ndarray::Array2;

/// Calculates the sum of squared errors (SSE) between predicted and actual values.
///
/// SSE = Σ(y_pred - y_actual)²
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
///
/// # Returns
/// * `f64` - Sum of squared errors
pub fn sum_of_squared_errors(residuals: &Array2<f64>) -> f64 {
    residuals.mapv(|x| x * x).sum()
}

/// Calculates Mean Squared Error (MSE), a measure of prediction accuracy that penalizes larger errors more heavily.
/// Lower values indicate better fit.
///
/// MSE = (1/2n) * Σ(y_pred - y_actual)²
/// where n is the total number of data points
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
/// * `num_samples` - Total number of data points across all measurements
///
/// # Returns
/// * `f64` - Mean squared error value
pub fn mean_squared_error(residuals: &Array2<f64>, num_samples: f64) -> f64 {
    residuals.mapv(|x| x * x).sum() / (2.0 * num_samples)
}

/// Calculates Root Mean Squared Error (RMSE), the square root of MSE.
/// Like MSE, lower values indicate better fit, but RMSE is in the same units as the original data.
///
/// RMSE = √((1/n) * Σ(y_pred - y_actual)²)
/// where n is the total number of data points
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
/// * `num_samples` - Total number of data points across all measurements
///
/// # Returns
/// * `f64` - Root mean squared error value
pub fn root_mean_squared_error(residuals: &Array2<f64>, num_samples: f64) -> f64 {
    mean_squared_error(residuals, num_samples).sqrt()
}

/// Calculates Mean Absolute Error (MAE), a measure of prediction accuracy that penalizes larger errors less heavily.
/// Lower values indicate better fit.
///
/// MAE = (1/n) * Σ|y_pred - y_actual|
/// where n is the total number of data points
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
/// * `num_samples` - Total number of data points across all measurements
///
/// # Returns
/// * `f64` - Mean absolute error value
pub fn mean_absolute_error(residuals: &Array2<f64>, num_samples: f64) -> f64 {
    residuals.mapv(|x| x.abs()).sum() / num_samples
}

/// Calculates Akaike Information Criterion (AIC), a measure of model quality that balances
/// goodness of fit against model complexity. Lower values indicate better models.
///
/// AIC helps prevent overfitting by penalizing models with more parameters.
/// When comparing models, prefer the one with the lowest AIC value.
///
/// AIC = n * ln(SSE/n) + 2k
/// where:
/// - n is the number of data points
/// - SSE is the sum of squared errors
/// - k is the number of model parameters
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
/// * `num_samples` - Total number of data points across all measurements
/// * `num_parameters` - Number of parameters in the model (model complexity)
///
/// # Returns
/// * `f64` - AIC value
pub fn akaike_information_criterion(
    residuals: &Array2<f64>,
    num_samples: f64,
    num_parameters: f64,
) -> f64 {
    let sse = sum_of_squared_errors(residuals);
    num_samples * (sse / num_samples).ln() + 2.0 * num_parameters
}

/// Calculates Bayesian Information Criterion (BIC), similar to AIC but with a stronger penalty
/// for model complexity. Lower values indicate better models.
///
/// BIC tends to prefer simpler models compared to AIC, especially with large sample sizes,
/// as its penalty for complexity increases with the sample size.
/// When comparing models, prefer the one with the lowest BIC value.
///
/// BIC = n * ln(SSE/n) + k * ln(n)
/// where:
/// - n is the number of data points
/// - SSE is the sum of squared errors
/// - k is the number of model parameters
///
/// # Arguments
/// * `residuals` - 2D array of residuals (differences between predicted and actual values)
/// * `num_samples` - Total number of data points across all measurements
/// * `num_parameters` - Number of parameters in the model (model complexity)
///
/// # Returns
/// * `f64` - BIC value
pub fn bayesian_information_criterion(
    residuals: &Array2<f64>,
    num_samples: f64,
    num_parameters: f64,
) -> f64 {
    let sse = sum_of_squared_errors(residuals);
    num_samples * (sse / num_samples).ln() + num_parameters * num_samples.ln()
}
