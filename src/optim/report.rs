use std::collections::HashMap;

use nalgebra::DMatrix;
use ndarray::Array1;
use plotly::Plot;
use serde::Serialize;

use crate::prelude::{
    error::SimulationError, result::SimulationResult, runner::InitCondInput, simulate,
    EnzymeMLDocument, SimulationSetup,
};

use super::{
    error::OptimizeError,
    metrics::{
        akaike_information_criterion, bayesian_information_criterion, mean_absolute_error,
        mean_squared_error, root_mean_squared_error,
    },
    problem::Problem,
    system::get_residuals,
};

/// A report containing optimization results and evaluation metrics
///
/// This struct holds the optimization problem configuration, the EnzymeML document,
/// the best parameters found during optimization, and calculated metrics evaluating
/// the quality of the fit. It provides methods for analyzing and visualizing the
/// optimization results.
///
/// The report includes:
/// - The original optimization problem configuration
/// - The EnzymeML document with updated parameter values
/// - Best-fit parameter values found during optimization
/// - Statistical metrics evaluating the fit quality
/// - Simulated model fits to experimental data
/// - Parameter uncertainties and correlations from the inverse Hessian matrix
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationReport {
    /// The original optimization problem configuration
    #[serde(skip)]
    pub problem: Problem,
    /// The EnzymeML document containing model and data
    pub doc: EnzymeMLDocument,
    /// Map of parameter names to their optimized values
    pub best_params: HashMap<String, f64>,
    /// Collection of metrics evaluating the optimization results
    pub metrics: Metrics,
    /// Fits to experimental data, mapping measurement IDs to simulation results
    pub fits: HashMap<String, SimulationResult>,
    /// Relative uncertainties of the best parameters (drawn from the inverse Hessian)
    pub relative_uncertainties: HashMap<String, f64>,
    /// Parameter correlations (drawn from the inverse Hessian)
    pub parameter_correlations: ParameterCorrelations,
}

impl OptimizationReport {
    /// Creates a new OptimizationReport with calculated metrics and uncertainty analysis
    ///
    /// This method:
    /// 1. Transforms the parameter vector into named parameters
    /// 2. Updates the EnzymeML document with optimized values
    /// 3. Computes statistical metrics for the fit
    /// 4. Simulates model predictions using best parameters
    /// 5. Calculates parameter uncertainties and correlations
    ///
    /// # Arguments
    /// * `problem` - The original optimization Problem containing model configuration
    /// * `doc` - The EnzymeML document containing model structure and experimental data
    /// * `param_vec` - Vector of optimized parameter values in transformed space
    ///
    /// # Returns
    /// * `Result<OptimizationReport, OptimizeError>` - A report containing all optimization
    ///   results and analysis if successful, or an error if analysis fails
    pub fn new(
        problem: Problem,
        doc: EnzymeMLDocument,
        param_vec: &Array1<f64>,
    ) -> Result<Self, OptimizeError> {
        // Transform the param_vec into a HashMap
        let best_params = problem.apply_transformations(param_vec)?;

        // First we need to set the best params in the doc
        let mut doc = doc;
        for (name, value) in best_params.iter() {
            let param = doc
                .parameters
                .iter_mut()
                .find(|p| p.id == name.clone())
                .unwrap();
            param.value = Some(*value);
        }

        // Compute metrics and fits
        let metrics = Metrics::new(&doc, &best_params, problem.get_n_points());
        let mut fits = HashMap::new();

        for meas in doc.measurements.iter() {
            let mut setup: SimulationSetup = meas.try_into().unwrap();

            // Set the dt to 0.1 to get a smoother fit
            setup.dt = 0.1;

            // Simulate the model
            let initial_conditions: InitCondInput = meas.into();
            let fit = simulate(&doc, initial_conditions, setup, None, None).unwrap();
            fits.insert(meas.id.clone(), fit.first().unwrap().clone());
        }

        // Compute relative uncertainties and parameter correlations
        let mut param_names = best_params.keys().cloned().collect::<Vec<String>>();
        param_names.sort();

        let inverse_hessian = problem.inverse_hessian(param_vec)?;

        let relative_uncertainties =
            Self::compute_relative_uncertainties(param_vec, &inverse_hessian, &param_names);
        let parameter_correlations = ParameterCorrelations::new(&inverse_hessian, &param_names);

        Ok(Self {
            problem,
            doc,
            best_params,
            metrics,
            fits,
            relative_uncertainties,
            parameter_correlations,
        })
    }

    /// Creates an interactive plot comparing model predictions to experimental data
    ///
    /// Generates a Plotly visualization showing:
    /// - Experimental data points
    /// - Model predictions using best-fit parameters
    /// - Error bars if experimental uncertainties are available
    ///
    /// # Arguments
    /// * `show` - Whether to display the plot immediately in the default browser
    ///
    /// # Returns
    /// * `Result<Plot, SimulationError>` - A Plotly Plot object containing the visualization if successful,
    ///   or a SimulationError if plotting fails
    pub fn plot_fit(&self, show: bool) -> Result<Plot, SimulationError> {
        let plot = self.doc.plot(Some(2), show, None, true)?;
        Ok(plot)
    }

    /// Computes relative parameter uncertainties using the inverse Hessian matrix
    ///
    /// This method uses the inverse Hessian matrix from maximum likelihood estimation (MLE)
    /// to estimate parameter uncertainties. The approach is based on asymptotic theory
    /// which states that for large sample sizes, the MLE is approximately normally
    /// distributed around the true parameter values.
    ///
    /// The inverse Hessian matrix approximates the covariance matrix of the parameter
    /// estimates. The diagonal elements contain the variances of individual parameters,
    /// and their square roots give the standard deviations.
    ///
    /// The relative uncertainty for each parameter is computed as:
    /// ```text
    /// relative_uncertainty = (standard_deviation / parameter_value) * 100%
    /// ```
    ///
    /// This gives the uncertainty as a percentage of the parameter value, which is
    /// useful for comparing uncertainties between parameters of different scales.
    ///
    /// Note: These uncertainties are approximate and based on local curvature of the
    /// likelihood surface at the optimum. They may underestimate true uncertainty if
    /// the likelihood surface is non-Gaussian or multiple optima exist.
    ///
    /// # Arguments
    /// * `best_params` - Vector of best-fit parameter values from optimization
    /// * `inverse_hessian` - Inverse Hessian matrix at the optimal parameter values
    /// * `param_names` - Names of the parameters in the same order as best_params
    ///
    /// # Returns
    /// * `HashMap<String, f64>` mapping parameter names to their relative uncertainties
    /// expressed as percentages
    pub fn compute_relative_uncertainties(
        best_params: &Array1<f64>,
        inverse_hessian: &DMatrix<f64>,
        param_names: &[String],
    ) -> HashMap<String, f64> {
        let best_params = best_params.as_slice().unwrap().to_vec();
        let std_dev: Vec<f64> = inverse_hessian
            .diagonal()
            .map(|x| x.sqrt()) // Convert variances to standard deviations
            .as_slice()
            .to_vec();

        let relative_uncertainty: Vec<f64> = std_dev
            .iter()
            .zip(best_params.as_slice())
            .map(|(std, param)| (std / param) * 100.0) // Convert to percentage
            .collect();
        param_names
            .iter()
            .zip(relative_uncertainty)
            .map(|(name, uncertainty)| (name.clone(), uncertainty))
            .collect()
    }
}

/// Collection of metrics evaluating optimization results
///
/// Contains various statistical measures comparing the model predictions
/// using optimized parameters against experimental data:
/// - MSE: Mean Squared Error - measures average squared deviation
/// - RMSE: Root Mean Squared Error - like MSE but in original units
/// - MAE: Mean Absolute Error - average absolute deviation
/// - AIC: Akaike Information Criterion - penalizes model complexity
/// - BIC: Bayesian Information Criterion - more strongly penalizes complexity
///
/// Lower values indicate better model fit for all metrics.
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    /// Mean Squared Error - average squared difference between predictions and observations
    pub mse: f64,
    /// Root Mean Squared Error - square root of MSE, in same units as original data
    pub rmse: f64,
    /// Mean Absolute Error - average absolute difference between predictions and observations
    pub mae: f64,
    /// Akaike Information Criterion - measures model quality considering complexity
    pub aic: f64,
    /// Bayesian Information Criterion - similar to AIC but with stronger complexity penalty
    pub bic: f64,
}

impl Metrics {
    /// Creates a new Metrics struct by calculating all evaluation metrics
    ///
    /// Computes MSE, RMSE, MAE, AIC and BIC using:
    /// - Residuals between model predictions and experimental data
    /// - Number of data points for normalization
    /// - Number of parameters for complexity penalties
    ///
    /// # Arguments
    /// * `doc` - The EnzymeML document containing model and experimental data
    /// * `params` - Map of parameter names to their optimized values
    /// * `n_points` - Total number of experimental data points
    ///
    /// # Returns
    /// * `Metrics` containing all calculated evaluation metrics
    fn new(doc: &EnzymeMLDocument, params: &HashMap<String, f64>, n_points: usize) -> Self {
        let residuals = get_residuals(doc, params);
        let num_params = params.len();

        Self {
            mse: mean_squared_error(&residuals, n_points as f64),
            rmse: root_mean_squared_error(&residuals, n_points as f64),
            mae: mean_absolute_error(&residuals, n_points as f64),
            aic: akaike_information_criterion(&residuals, n_points as f64, num_params as f64),
            bic: bayesian_information_criterion(&residuals, n_points as f64, num_params as f64),
        }
    }
}

/// Stores parameter correlation information derived from the inverse Hessian
///
/// Contains parameter names and their correlation matrix. The correlation matrix
/// is symmetric, with diagonal elements equal to 1.0 and off-diagonal elements
/// between -1.0 and 1.0 indicating correlation strength and direction.
#[derive(Debug, Clone, Serialize)]
pub struct ParameterCorrelations {
    /// Names of parameters in same order as correlation matrix
    pub parameters: Vec<String>,
    /// Correlation matrix with elements between -1.0 and 1.0
    pub matrix: Vec<Vec<f64>>,
}

impl ParameterCorrelations {
    /// Creates a new ParameterCorrelations from the inverse Hessian matrix
    ///
    /// Computes correlations by normalizing covariances:
    /// correlation = covariance / (std_dev_i * std_dev_j)
    ///
    /// Only fills lower triangular part of matrix since correlations are symmetric.
    ///
    /// # Arguments
    /// * `inverse_hessian` - Inverse Hessian matrix containing parameter covariances
    /// * `param_names` - Names of parameters in same order as matrix
    ///
    /// # Returns
    /// * `ParameterCorrelations` containing correlation matrix and parameter names
    pub fn new(inverse_hessian: &DMatrix<f64>, param_names: &[String]) -> Self {
        let n = inverse_hessian.ncols();
        let std_devs = inverse_hessian.diagonal().map(|x| x.sqrt());

        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let correlation = inverse_hessian[(i, j)] / (std_devs[i] * std_devs[j]);
                matrix[i][j] = correlation;
            }
        }

        ParameterCorrelations {
            parameters: param_names.to_vec(),
            matrix,
        }
    }
}
