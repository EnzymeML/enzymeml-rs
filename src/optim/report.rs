use std::collections::HashMap;

use argmin::core::CostFunction;
use ndarray::Array1;
use peroxide::fuga::ODEIntegrator;
use plotly::Plot;
use serde::Serialize;

use crate::{
    prelude::{EnzymeMLDocument, SimulationResult},
    simulation::error::SimulationError,
};

use super::{error::OptimizeError, metrics::akaike_information_criterion, problem::Problem};

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
    /// The EnzymeML document containing model and data
    pub doc: EnzymeMLDocument,
    /// Map of parameter names to their optimized values
    pub best_params: HashMap<String, f64>,
    /// Fits to experimental data, mapping measurement IDs to simulation results
    pub fits: HashMap<String, SimulationResult>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Relative uncertainties of the best parameters
    pub uncertainties: Option<HashMap<String, f64>>,
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
    pub(crate) fn new<S: ODEIntegrator + Copy>(
        problem: &Problem<S>,
        doc: EnzymeMLDocument,
        param_vec: &Vec<f64>,
        _: Option<Vec<f64>>,
    ) -> Result<Self, OptimizeError> {
        // Transform the param_vec into a HashMap
        let param_vec = problem.apply_transformations(param_vec)?;
        let best_params = problem
            .ode_system()
            .get_sorted_params()
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), param_vec[i]))
            .collect::<HashMap<String, f64>>();

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

        let system = problem.ode_system();
        let fits = system
            .bulk_integrate::<SimulationResult>(
                problem.simulation_setup(),
                problem.initials(),
                Some(&param_vec),
                None,
                problem.solver(),
                None,
            )?
            .iter()
            .zip(doc.measurements.iter())
            .map(|(fit, measurement)| (measurement.id.clone(), fit.clone()))
            .collect::<HashMap<String, SimulationResult>>();

        let aic = akaike_information_criterion(
            problem.cost(&Array1::from_vec(param_vec)).unwrap(),
            system.get_sorted_params().len(),
        );

        Ok(Self {
            doc,
            best_params,
            fits,
            aic,
            uncertainties: None,
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
        let plot = self.doc.plot(Some(2), show, None, true).unwrap();
        Ok(plot)
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
    /// Akaike Information Criterion - measures model quality considering complexity
    pub aic: f64,
    /// Bayesian Information Criterion - similar to AIC but with stronger complexity penalty
    pub bic: f64,
}
