use std::{
    collections::{BTreeMap, HashMap},
    fmt::{self, Display},
};

use argmin::core::CostFunction;
use ndarray::Array1;
use peroxide::fuga::ODEIntegrator;
use serde::{Deserialize, Serialize};
use tabled::{builder::Builder, settings::Style};

use crate::prelude::{EnzymeMLDocument, ObjectiveFunction, SimulationResult};

use super::{
    error::OptimizeError,
    metrics::{akaike_information_criterion, bayesian_information_criterion},
    problem::Problem,
    Bound, InitialGuesses,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// The EnzymeML document containing model and data
    #[serde(skip)]
    pub doc: EnzymeMLDocument,
    /// Map of parameter names to their optimized values
    pub best_params: BTreeMap<String, f64>,
    /// Fits to experimental data, mapping measurement IDs to simulation results
    #[serde(skip)]
    pub fits: HashMap<String, SimulationResult>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Error of the fit
    pub error: f64,
    /// Loss function used
    pub loss_function: String,
    /// Relative uncertainties of the best parameters
    #[serde(skip)]
    pub uncertainties: Option<HashMap<String, f64>>,
    /// Bounds of the parameters
    #[serde(skip)]
    pub bounds: Option<Vec<Bound>>,
    /// Initial guesses
    #[serde(skip)]
    pub initial_guesses: Option<InitialGuesses>,
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
    pub(crate) fn new<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction>(
        problem: &Problem<S, L>,
        doc: EnzymeMLDocument,
        param_vec: &[f64],
        initial_guesses: Option<InitialGuesses>,
        bounds: Option<Vec<Bound>>,
    ) -> Result<Self, OptimizeError> {
        let best_params = Self::transform_parameters(problem, param_vec)?;
        let doc = Self::update_document(doc, &best_params);
        let fits = Self::simulate_fits(problem, &doc, param_vec)?;
        let (aic, bic, error) = Self::calculate_metrics(problem, param_vec, &doc)?;

        Ok(Self {
            doc,
            best_params,
            fits,
            aic,
            bic,
            error,
            loss_function: problem.objective().to_string(),
            uncertainties: None,
            bounds,
            initial_guesses,
        })
    }

    /// Transforms optimization parameters into named parameter map
    ///
    /// # Arguments
    /// * `problem` - The optimization problem containing parameter information
    /// * `param_vec` - Raw parameter vector from optimization
    ///
    /// # Returns
    /// * `Result<BTreeMap<String, f64>, OptimizeError>` - Map of parameter names to values
    fn transform_parameters<S: ODEIntegrator + Copy, L: ObjectiveFunction>(
        problem: &Problem<S, L>,
        param_vec: &[f64],
    ) -> Result<BTreeMap<String, f64>, OptimizeError> {
        let transformed_params = problem.apply_transformations(param_vec)?;
        Ok(problem
            .ode_system()
            .get_sorted_params()
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), transformed_params[i]))
            .collect())
    }

    /// Updates EnzymeML document with optimized parameter values
    ///
    /// # Arguments
    /// * `doc` - Original EnzymeML document
    /// * `best_params` - Map of parameter names to optimized values
    ///
    /// # Returns
    /// * Updated EnzymeML document
    fn update_document(
        mut doc: EnzymeMLDocument,
        best_params: &BTreeMap<String, f64>,
    ) -> EnzymeMLDocument {
        for (name, value) in best_params.iter() {
            let param = doc
                .parameters
                .iter_mut()
                .find(|p| p.id == name.clone())
                .unwrap();
            param.value = Some(*value);
        }
        doc
    }

    /// Simulates model fits using optimized parameters
    ///
    /// # Arguments
    /// * `problem` - The optimization problem
    /// * `doc` - EnzymeML document with experimental data
    /// * `param_vec` - Optimized parameter vector
    ///
    /// # Returns
    /// * `Result<HashMap<String, SimulationResult>, OptimizeError>` - Map of measurement IDs to simulation results
    fn simulate_fits<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction>(
        problem: &Problem<S, L>,
        doc: &EnzymeMLDocument,
        param_vec: &[f64],
    ) -> Result<HashMap<String, SimulationResult>, OptimizeError> {
        let system = problem.ode_system();
        Ok(system
            .bulk_integrate::<SimulationResult>(
                problem.simulation_setup(),
                problem.initials(),
                Some(param_vec),
                None,
                problem.solver(),
                None,
            )?
            .iter()
            .zip(doc.measurements.iter())
            .map(|(fit, measurement)| (measurement.id.clone(), fit.clone()))
            .collect())
    }

    /// Calculates AIC and BIC metrics for the optimization result
    ///
    /// # Arguments
    /// * `problem` - The optimization problem
    /// * `param_vec` - Optimized parameter vector
    /// * `doc` - EnzymeML document with data
    ///
    /// # Returns
    /// * `Result<(f64, f64, f64), OptimizeError>` - Tuple of (AIC, BIC, cost) values
    fn calculate_metrics<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction>(
        problem: &Problem<S, L>,
        param_vec: &[f64],
        doc: &EnzymeMLDocument,
    ) -> Result<(f64, f64, f64), OptimizeError> {
        let cost = problem
            .cost(&Array1::from_vec(param_vec.to_vec()))
            .map_err(OptimizeError::ArgMinError)?;
        let num_params = problem.ode_system().get_sorted_params().len();
        let aic = akaike_information_criterion(cost, num_params);
        let bic = bayesian_information_criterion(cost, num_params, doc.measurements.len());

        Ok((aic, bic, cost))
    }
}

impl Display for OptimizationReport {
    /// Formats the OptimizationReport as a string
    ///
    /// # Arguments
    /// * `f` - The formatter to write the report to
    ///
    /// # Returns
    /// * `fmt::Result` - The formatted report
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\nOptimization Report\n")?;

        // Create a table of all metrics
        let mut builder = Builder::default();
        builder.push_record(vec!["Metric", "Value"]);
        builder.push_record(vec![&self.loss_function, &self.error.to_string()]);
        builder.push_record(vec!["Akaike Information Criterion", &self.aic.to_string()]);
        builder.push_record(vec![
            "Bayesian Information Criterion",
            &self.bic.to_string(),
        ]);

        let mut table = builder.build();
        table.with(Style::rounded());
        write!(f, "\n{table}\n")?;

        // Create a table of the best parameters
        let mut builder = Builder::default();
        builder.push_record(vec![
            "Parameter",
            "Value",
            "Initial Guess",
            "Lower Bound",
            "Upper Bound",
        ]);

        for (i, (name, value)) in self.best_params.iter().enumerate() {
            // Get initial guess and bounds as strings, defaulting to "-" if not present
            let initial_guess = self
                .initial_guesses
                .as_ref()
                .map_or("-".to_string(), |g| g.get_value_at(i).to_string());

            let (lower_bound, upper_bound) =
                self.bounds
                    .as_ref()
                    .map_or(("-".to_string(), "-".to_string()), |bounds| {
                        let bound = bounds.iter().find(|b| b.param() == name);
                        (
                            bound.map_or("-".to_string(), |b| b.lower().to_string()),
                            bound.map_or("-".to_string(), |b| b.upper().to_string()),
                        )
                    });

            builder.push_record(vec![
                name.to_string(),
                value.to_string(),
                initial_guess,
                lower_bound,
                upper_bound,
            ]);
        }

        let mut table = builder.build();
        table.with(Style::rounded());
        write!(f, "\n{table}\n")?;

        Ok(())
    }
}
