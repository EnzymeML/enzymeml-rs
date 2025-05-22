//! Profile likelihood analysis for parameter identifiability.
//!
//! This module provides functionality for performing profile likelihood analysis,
//! which is a method for assessing parameter identifiability and confidence intervals
//! in ODE-based models.
//!
//! Profile likelihood systematically varies one parameter while re-optimizing all others,
//! allowing assessment of parameter identifiability, confidence intervals, and
//! structural/practical non-identifiability.
//!
//! Two main approaches are supported:
//! - Grid-based profile likelihood with uniform sampling across parameter ranges
//! - EGO-based profile likelihood using Bayesian optimization for adaptive sampling
//!
//! # Overview
//!
//! Profile likelihood analysis is a powerful tool for assessing parameter identifiability
//! in mathematical models. It involves systematically varying one parameter while re-optimizing
//! all others, generating a profile of the likelihood function. This profile can be used to
//! assess parameter identifiability, confidence intervals, and structural/practical non-identifiability.
//!
//! # Usage
//!
//! The module provides two main functions for performing profile likelihood analysis:
//!
//! - `profile_likelihood`: Performs profile likelihood analysis using a uniform grid sampling approach.
//! - `ego_profile_likelihood`: Performs profile likelihood analysis using Efficient Global Optimization (EGO).
//!
//! Both functions take a problem, initial guess, parameters to profile, and an optimizer as input,
//! and return a `ProfileResults` object containing the profile likelihood data.

use std::collections::HashMap;
use std::str::FromStr;

use argmin::core::{TerminationReason, TerminationStatus};
use egobox_ego::EgorBuilder;
use ndarray::{Array1, Array2, ArrayView2};
use peroxide::fuga::{CubicSpline, ODEIntegrator, Spline};
use peroxide::{linspace, seq};
use plotly::common::{Anchor, Font, Line, Mode, Title};
use plotly::layout::{Annotation, Axis, GridPattern, LayoutGrid};
use plotly::{Layout, Plot, Scatter};
use rayon::prelude::*;
use regex::Regex;

use crate::{
    optim::{InitialGuesses, OptimizeError, Optimizer, Problem},
    prelude::ObjectiveFunction,
};

const DEFAULT_HEIGHT: usize = 400;
const DEFAULT_WIDTH: usize = 500;
const UPPER_BOUND: f64 = 1.1;

/// Performs profile likelihood analysis on multiple parameters using EGO.
///
/// This function systematically varies each parameter across its range while re-optimizing
/// all other parameters, generating likelihood profiles. It uses Efficient Global Optimization (EGO)
/// to adaptively sample the parameter space, focusing on regions of interest.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem containing the model and data
/// * `initial_guess` - Initial parameter values for optimization
/// * `parameters` - List of parameters to profile with their ranges
/// * `max_iters` - Maximum number of EGO iterations per parameter
/// * `optimizer` - Optimizer to use for each sub-optimization
///
/// # Returns
///
/// A `ProfileResults` containing profile likelihood data for all parameters,
/// or an error if profiling fails
///
/// # Notes
///
/// - EGO is a Bayesian optimization method that adaptively samples the parameter space,
///   focusing on regions of interest. This can be more efficient than uniform grid sampling
///   for complex likelihood surfaces.
/// - The `max_iters` parameter controls the number of EGO iterations per parameter. A higher
///   value will result in more accurate profiles but will take longer to compute.
/// - The `optimizer` parameter is used for each sub-optimization. It should be a robust
///   optimizer that can handle the specific problem at hand.

#[bon::builder]
pub fn ego_profile_likelihood<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameters: Vec<impl Into<ProfileParameter>>,
    max_iters: usize,
    optimizer: &O,
) -> Result<ProfileResults, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone + Send + Sync,
{
    let parameters: Vec<ProfileParameter> = parameters.into_iter().map(|p| p.into()).collect();

    let results = parameters
        .par_iter()
        .map(|parameter| {
            ego_profile_likelihood_core(
                problem,
                initial_guess.clone(),
                parameter,
                optimizer,
                max_iters,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ProfileResults(results))
}

/// Core implementation for EGO-based profile likelihood for a single parameter.
///
/// This function performs profile likelihood analysis on a single parameter using
/// Efficient Global Optimization (EGO) to adaptively sample the parameter space.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `initial_guess` - Initial parameter values
/// * `parameter` - Parameter to profile with its range
/// * `optimizer` - Optimizer to use for each sub-optimization
/// * `max_iters` - Maximum number of EGO iterations
///
/// # Returns
///
/// A `ProfileResult` containing the profile likelihood data, or an error
///
/// # Notes
///
/// - This function is called by `ego_profile_likelihood` for each parameter to profile.
/// - It uses EGO to adaptively sample the parameter space, focusing on regions of interest.
/// - The `max_iters` parameter controls the number of EGO iterations. A higher value will
///   result in more accurate profiles but will take longer to compute.
pub fn ego_profile_likelihood_core<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameter: &ProfileParameter,
    optimizer: &O,
    max_iters: usize,
) -> Result<ProfileResult, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone + Send + Sync,
{
    let (err_min, initial_guess) =
        prepare_profile_problem(problem, initial_guess, parameter, optimizer)?;
    let param_index = get_param_index(problem, parameter);

    let objective = |param_value: &ArrayView2<f64>| {
        // Get the only value in the array
        let params = param_value.as_slice().expect("Failed to get slice");
        // Parallel computation of log probabilities
        let profile = params
            .par_iter()
            .map(|&param_value| {
                compute_likelihood_ratio(
                    problem,
                    &initial_guess,
                    parameter,
                    param_value.exp(),
                    optimizer,
                    err_min,
                    param_index,
                )
            })
            .collect::<Result<Vec<(f64, f64, f64)>, OptimizeError>>()
            .unwrap();

        let likelihoods: Vec<f64> = profile.iter().map(|(_, l, _)| *l).collect();
        Array2::from_shape_vec((params.len(), 1), likelihoods).unwrap()
    };

    // Set bound
    let bound =
        Array2::from_shape_vec((1, 2), vec![parameter.from.ln(), parameter.to.ln()]).unwrap();

    // Run EGO
    let result = EgorBuilder::optimize(objective)
        .configure(|config| config.max_iters(max_iters))
        .min_within(&bound)
        .run()
        .map_err(|_| OptimizeError::ConvergenceError)?;

    if let TerminationStatus::Terminated(TerminationReason::SolverExit(_)) =
        result.state.termination_status
    {
        return Err(OptimizeError::CostNaN);
    }

    let param_values: Vec<f64> = result
        .x_doe
        .as_slice()
        .unwrap()
        .to_vec()
        .iter()
        .map(|x| x.exp())
        .collect();
    let likelihoods: Vec<f64> = result.y_doe.as_slice().unwrap().to_vec();
    let ratios: Vec<f64> = likelihoods
        .iter()
        .map(|err_profile| likelihood_ratio(err_min, *err_profile))
        .collect();

    // Create indices for sorting
    let mut indices: Vec<usize> = (0..param_values.len()).collect();
    indices.sort_by(|&i, &j| param_values[i].partial_cmp(&param_values[j]).unwrap());

    // Sort all vectors using the indices
    let param_values: Vec<f64> = indices.iter().map(|&i| param_values[i]).collect();
    let likelihoods: Vec<f64> = indices.iter().map(|&i| likelihoods[i]).collect();
    let ratios: Vec<f64> = indices.iter().map(|&i| ratios[i]).collect();

    let best_value = best_parameter_val(&param_values, &ratios);

    Ok(ProfileResult {
        best_value,
        param_name: parameter.name.clone(),
        param_values,
        likelihoods,
        ratios,
    })
}

/// Performs profile likelihood analysis on multiple parameters.
///
/// This function analyzes parameter identifiability by systematically varying each parameter
/// in the provided list while re-optimizing all others, generating likelihood profiles.
/// It uses a uniform grid sampling approach with a fixed number of steps.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem containing the model and data
/// * `initial_guess` - Initial parameter values for optimization
/// * `parameters` - List of parameters to profile with their ranges
/// * `n_steps` - Number of points to evaluate within each parameter range
/// * `optimizer` - Optimizer to use for each sub-optimization
///
/// # Returns
///
/// A `ProfileResults` containing profile likelihood data for all parameters,
/// or an error if profiling fails
///
/// # Notes
///
/// - This function uses a uniform grid sampling approach, which is simpler but may be less
///   efficient than EGO for complex likelihood surfaces.
/// - The `n_steps` parameter controls the number of points to evaluate within each parameter
///   range. A higher value will result in more accurate profiles but will take longer to compute.
/// - The `optimizer` parameter is used for each sub-optimization. It should be a robust
///   optimizer that can handle the specific problem at hand.

#[bon::builder]
pub fn profile_likelihood<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameters: Vec<impl Into<ProfileParameter>>,
    n_steps: usize,
    optimizer: &O,
) -> Result<ProfileResults, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone + Send + Sync,
{
    let parameters: Vec<ProfileParameter> = parameters.into_iter().map(|p| p.into()).collect();

    let results = parameters
        .par_iter()
        .map(|parameter| {
            profile_likelihood_core(
                problem,
                initial_guess.clone(),
                parameter,
                n_steps,
                optimizer,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ProfileResults(results))
}

/// Core implementation for a single parameter profile likelihood
///
/// This function systematically varies a single parameter across a specified range
/// while re-optimizing all other parameters, to assess parameter identifiability
/// and confidence intervals. It uses a uniform grid sampling approach.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `initial_guess` - Initial parameter values
/// * `parameter` - Parameter to profile with its range
/// * `n_steps` - Number of points to evaluate within the range
/// * `optimizer` - Optimizer to use for each sub-optimization
///
/// # Returns
///
/// A `ProfileResult` containing the profile likelihood data, or an error
///
/// # Notes
///
/// - This function is called by `profile_likelihood` for each parameter to profile.
/// - It uses a uniform grid sampling approach, which is simpler but may be less efficient
///   than EGO for complex likelihood surfaces.
/// - The `n_steps` parameter controls the number of points to evaluate within the parameter
///   range. A higher value will result in more accurate profiles but will take longer to compute.
fn profile_likelihood_core<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameter: &ProfileParameter,
    n_steps: usize,
    optimizer: &O,
) -> Result<ProfileResult, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Prepare the problem
    let (err_min, initial_guess) =
        prepare_profile_problem(problem, initial_guess, parameter, optimizer)?;

    // Then, profile the likelihood
    let param_range = linspace!(parameter.from, parameter.to, n_steps);
    let param_index = get_param_index(problem, parameter);

    // Parallel computation of log probabilities
    let profile = param_range
        .par_iter()
        .map(|&param_value| {
            compute_likelihood_ratio(
                problem,
                &initial_guess,
                parameter,
                param_value,
                optimizer,
                err_min,
                param_index,
            )
        })
        .collect::<Result<Vec<(f64, f64, f64)>, OptimizeError>>()?;

    let param_values: Vec<f64> = profile.iter().map(|(p, _, _)| *p).collect();
    let likelihoods: Vec<f64> = profile.iter().map(|(_, l, _)| *l).collect();
    let ratios: Vec<f64> = profile.iter().map(|(_, _, r)| *r).collect();

    // Extract the best parameter
    let best_value = best_parameter_val(&param_values, &ratios);

    Ok(ProfileResult {
        best_value,
        param_name: parameter.name.clone(),
        param_values,
        likelihoods,
        ratios,
    })
}

/// Prepares a problem for profile likelihood analysis.
///
/// This function performs an initial optimization to find the best parameter values,
/// which will be used as the starting point for profile likelihood analysis.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `initial_guess` - Initial parameter values
/// * `parameter` - Parameter to profile
/// * `optimizer` - Optimizer to use
///
/// # Returns
///
/// A tuple containing the minimum error value and the optimized initial guess,
/// or an error if preparation fails
///
/// # Notes
///
/// - This function is called by both `profile_likelihood` and `ego_profile_likelihood`
///   to prepare the problem for profiling.
/// - It performs an initial optimization to find the best parameter values, which will
///   be used as the starting point for profile likelihood analysis.
/// - The `optimizer` parameter is used for the initial optimization. It should be a robust
///   optimizer that can handle the specific problem at hand.
fn prepare_profile_problem<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameter: &ProfileParameter,
    optimizer: &O,
) -> Result<(f64, InitialGuesses), OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Check if the fixed parameter is in the problem
    if !problem
        .ode_system()
        .get_sorted_params()
        .contains(&parameter.name)
    {
        return Err(OptimizeError::UnknownParameter(parameter.name.clone()));
    }

    // First, fit the problem to get the initial guess
    let initial_guess = initial_guess.into();
    let report = optimizer.optimize(problem, Some(initial_guess))?;
    let err_min = report.error;

    // Use the optimized parameters as the initial guess
    let initial_guess: InitialGuesses = Array1::from_vec(
        problem
            .ode_system()
            .get_sorted_params()
            .iter()
            .map(|param| {
                report.best_params.get(param).copied().unwrap_or_else(|| {
                    panic!("Parameter {} not found in optimized parameters", param)
                })
            })
            .collect(),
    )
    .into();

    Ok((err_min, initial_guess))
}

/// Gets the index of a parameter in the problem's sorted parameter list.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `parameter` - Parameter to find
///
/// # Returns
///
/// The index of the parameter in the problem's sorted parameter list
///
/// # Notes
///
/// - This function is used to find the index of a parameter in the problem's sorted
///   parameter list. This index is used to set the parameter value in the initial guess.
/// - The function assumes that the parameter exists in the problem's sorted parameter list.
///   If the parameter does not exist, the function will panic.
fn get_param_index<S, L>(problem: &Problem<S, L>, parameter: &ProfileParameter) -> usize
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
{
    problem
        .ode_system()
        .get_sorted_params()
        .iter()
        .position(|p| p == &parameter.name)
        .unwrap()
}

/// Computes the likelihood ratio for a specific parameter value.
///
/// This function fixes a parameter at a specific value, optimizes all other parameters,
/// and computes the likelihood ratio relative to the best fit.
///
/// # Type Parameters
///
/// * `S` - ODE integrator type
/// * `L` - Objective function type
/// * `O` - Optimizer type
/// * `T` - Initial guess type
///
/// # Arguments
///
/// * `problem` - The optimization problem
/// * `initial_guess` - Initial parameter values
/// * `parameter` - Parameter being profiled
/// * `param_value` - Value to fix the parameter at
/// * `optimizer` - Optimizer to use
/// * `err_min` - Minimum error value from the best fit
/// * `param_index` - Index of the parameter in the problem's sorted parameter list
///
/// # Returns
///
/// A tuple containing the parameter value, the error, and the likelihood ratio,
/// or an error if optimization fails
///
/// # Notes
///
/// - This function is called by both `profile_likelihood` and `ego_profile_likelihood`
///   to compute the likelihood ratio for a specific parameter value.
/// - It fixes the parameter at the specified value, optimizes all other parameters,
///   and computes the likelihood ratio relative to the best fit.
/// - The `optimizer` parameter is used for the optimization. It should be a robust
///   optimizer that can handle the specific problem at hand.
fn compute_likelihood_ratio<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: &T,
    parameter: &ProfileParameter,
    param_value: f64,
    optimizer: &O,
    err_min: f64,
    param_index: usize,
) -> Result<(f64, f64, f64), OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Fix the parameter in problem
    let mut problem = problem.clone();
    problem.fix_param(&parameter.name)?;

    // Set the parameter in the initial guess
    let mut initial_guess = initial_guess.clone().into();
    initial_guess.set_value_at(param_index, param_value);

    // Optimize the problem
    let report = optimizer.optimize(&problem, Some(initial_guess))?;
    let ratio = likelihood_ratio(err_min, report.error);

    Ok((param_value, report.error, ratio))
}

/// Configuration for a parameter to be profiled.
///
/// Specifies the parameter name and the range of values to test during profiling.
/// Can be created using the builder pattern or parsed from a string.
///
/// # Notes
///
/// - The `name` field specifies the name of the parameter to profile.
/// - The `from` field specifies the lower bound of the parameter range.
/// - The `to` field specifies the upper bound of the parameter range.
/// - The `from` value must be less than the `to` value.
#[derive(Debug, Clone, bon::Builder)]
#[allow(clippy::duplicated_attributes)]
#[builder(on(String, into), on(f64, into))]
pub struct ProfileParameter {
    /// Name of the profiled parameter
    name: String,
    /// From value (lower bound of the range)
    from: f64,
    /// To value (upper bound of the range)
    to: f64,
}

impl ProfileParameter {
    /// Creates a new ProfileParameter instance.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter
    /// * `from` - The starting value of the parameter
    /// * `to` - The ending value of the parameter
    ///
    /// # Returns
    ///
    /// A new ProfileParameter instance, or an error if the range is invalid
    pub fn new(name: String, from: f64, to: f64) -> Result<Self, OptimizeError> {
        if from >= to {
            return Err(OptimizeError::ProfileParameterParseError(format!(
                "From value {} must be less than to value {}",
                from, to
            )));
        }

        Ok(Self { name, from, to })
    }
}

impl FromStr for ProfileParameter {
    type Err = OptimizeError;

    /// Parses a ProfileParameter from a string.
    ///
    /// The string should be in the format "name=from:to", where:
    /// - name is the parameter name
    /// - from is the lower bound of the range
    /// - to is the upper bound of the range
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Example: "k_cat=1.0:2.0"
        let pattern = Regex::new(r"^(\w+)=([0-9.]+):([0-9.]+)$").unwrap();
        let caps = pattern
            .captures(s)
            .ok_or(OptimizeError::ProfileParameterParseError(s.to_string()))?;

        if caps.len() != 4 {
            return Err(OptimizeError::ProfileParameterParseError(s.to_string()));
        }

        let from = caps[2]
            .parse::<f64>()
            .map_err(|e| OptimizeError::ProfileParameterParseError(e.to_string()))?;
        let to = caps[3]
            .parse::<f64>()
            .map_err(|e| OptimizeError::ProfileParameterParseError(e.to_string()))?;

        Self::new(caps[1].to_string(), from, to)
    }
}

/// Results from a profile likelihood analysis for a single parameter.
///
/// Contains the profile likelihood data for a single parameter, including
/// the parameter values tested, corresponding likelihood values, and likelihood ratios.
/// Also stores the parameter value with the best likelihood.
///
/// # Notes
///
/// - The `best_value` field stores the parameter value with the best (lowest) likelihood.
/// - The `param_name` field stores the name of the profiled parameter.
/// - The `param_values` field stores the vector of parameter values tested during profiling.
/// - The `likelihoods` field stores the vector of likelihood values corresponding to each parameter value.
/// - The `ratios` field stores the vector of likelihood ratios (normalized to the best likelihood).
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// The parameter value with the best (lowest) likelihood
    best_value: f64,
    /// Name of the profiled parameter
    param_name: String,
    /// Vector of parameter values tested during profiling
    param_values: Vec<f64>,
    /// Vector of likelihood values corresponding to each parameter value
    likelihoods: Vec<f64>,
    /// Vector of likelihood ratios (normalized to the best likelihood)
    ratios: Vec<f64>,
}

impl ProfileResult {
    /// Creates a new ProfileResult instance.
    ///
    /// # Arguments
    ///
    /// * `best_value` - The parameter value with the best likelihood
    /// * `param_name` - Name of the profiled parameter
    /// * `param_values` - Vector of parameter values tested
    /// * `likelihoods` - Vector of likelihood values for each parameter value
    /// * `ratios` - Vector of likelihood ratios
    pub fn new(
        best_value: f64,
        param_name: String,
        param_values: Vec<f64>,
        likelihoods: Vec<f64>,
        ratios: Vec<f64>,
    ) -> Self {
        Self {
            best_value,
            param_name,
            param_values,
            likelihoods,
            ratios,
        }
    }

    /// Returns the parameter value with the best likelihood.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Returns the name of the profiled parameter.
    pub fn param_name(&self) -> &str {
        &self.param_name
    }

    /// Returns the vector of parameter values tested during profiling.
    pub fn param_values(&self) -> &[f64] {
        &self.param_values
    }

    /// Returns the vector of likelihood values for each parameter value.
    pub fn likelihoods(&self) -> &[f64] {
        &self.likelihoods
    }

    /// Returns the vector of likelihood ratios (normalized to the best likelihood).
    pub fn ratios(&self) -> &[f64] {
        &self.ratios
    }
}

/// Calculates the likelihood ratio from error values.
///
/// Converts the difference between the minimum error and a profile error
/// into a likelihood ratio using the chi-squared distribution relationship.
///
/// # Arguments
///
/// * `err_min` - The minimum error value (from the best fit)
/// * `err_profile` - The error value at a specific profile point
///
/// # Returns
///
/// The likelihood ratio value, which is exp(-0.5 * (err_profile - err_min))
///
/// # Notes
///
/// - The likelihood ratio is calculated using the chi-squared distribution relationship:
///   exp(-0.5 * (err_profile - err_min)).
/// - A likelihood ratio of 1.0 indicates that the profile point is as likely as the best fit.
/// - A likelihood ratio less than 1.0 indicates that the profile point is less likely than the best fit.
fn likelihood_ratio(err_min: f64, err_profile: f64) -> f64 {
    (-0.5 * (err_profile - err_min)).exp()
}

/// Finds the parameter value with the best (lowest) likelihood.
///
/// # Arguments
///
/// * `param_values` - Vector of parameter values
/// * `ratios` - Vector of corresponding likelihood ratios
///
/// # Returns
///
/// The parameter value with the highest likelihood ratio
///
/// # Notes
///
/// - This function finds the parameter value with the highest likelihood ratio.
/// - The likelihood ratio is calculated using the chi-squared distribution relationship:
///   exp(-0.5 * (err_profile - err_min)).
/// - A likelihood ratio of 1.0 indicates that the profile point is as likely as the best fit.
/// - A likelihood ratio less than 1.0 indicates that the profile point is less likely than the best fit.
fn best_parameter_val(param_values: &[f64], ratios: &[f64]) -> f64 {
    let position_max = ratios
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();
    param_values[position_max]
}

/// Converts a ProfileResult into a plotly Plot for visualization.
///
/// Creates a line plot showing the likelihood ratio vs parameter value.
/// This implementation allows easy visualization of a single parameter's profile.
///
/// # Notes
///
/// - This implementation creates a line plot showing the likelihood ratio vs parameter value.
/// - The plot includes a title, x-axis label, and y-axis label.
/// - The x-axis label is the parameter name.
/// - The y-axis label is "Likelihood Ratio".
impl From<ProfileResult> for Plot {
    fn from(result: ProfileResult) -> Self {
        let mut plot = Plot::new();

        let trace = Scatter::new(result.param_values, result.ratios)
            .name(format!("Likelihood Ratio of {}", result.param_name))
            .mode(Mode::LinesMarkers);

        plot.add_trace(trace);

        let layout = Layout::default()
            .title(format!("Profile Likelihood of {}", result.param_name))
            .x_axis(
                Axis::new()
                    .title(result.param_name.clone())
                    .auto_range(true),
            )
            .y_axis(Axis::new().title("Likelihood Ratio").auto_range(true));

        plot.set_layout(layout);

        plot
    }
}

/// Collection of profile likelihood results for multiple parameters.
///
/// Provides convenient access to individual profile results and methods
/// for visualization and conversion to other formats.
///
/// # Notes
///
/// - This struct is a collection of `ProfileResult` objects, one for each parameter profiled.
/// - It provides methods for accessing individual profile results and converting the results
///   to other formats, such as a `HashMap` or a `Plot`.
#[derive(Debug, Clone)]
pub struct ProfileResults(pub Vec<ProfileResult>);

impl ProfileResults {
    /// Returns the number of parameter profiles in the collection.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a reference to the first profile result, if any.
    pub fn first(&self) -> Option<&ProfileResult> {
        self.0.first()
    }

    /// Returns a reference to the last profile result, if any.
    pub fn last(&self) -> Option<&ProfileResult> {
        self.0.last()
    }
}

impl From<ProfileResults> for HashMap<String, ProfileResult> {
    /// Converts the profile results into a HashMap keyed by parameter name.
    ///
    /// This allows easy lookup of profile results by parameter name.
    fn from(results: ProfileResults) -> HashMap<String, ProfileResult> {
        results
            .0
            .into_iter()
            .map(|r| (r.param_name.clone(), r))
            .collect()
    }
}

impl IntoIterator for ProfileResults {
    type Item = ProfileResult;
    type IntoIter = std::vec::IntoIter<ProfileResult>;

    /// Allows iterating over the profile results.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Converts ProfileResults into a multi-panel plotly Plot for visualization.
///
/// Creates a grid of line plots showing likelihood ratios for all profiled parameters.
/// This implementation allows easy visualization of multiple parameter profiles at once.
///
/// # Notes
///
/// - This implementation creates a grid of line plots, one for each parameter profiled.
/// - Each plot shows the likelihood ratio vs parameter value.
/// - The plots are arranged in a grid, with two columns and a number of rows equal to the
///   number of parameters profiled.
/// - Each plot includes a title, x-axis label, and y-axis label.
/// - The x-axis label is the parameter name.
/// - The y-axis label is "Likelihood Ratio".
impl From<ProfileResults> for Plot {
    fn from(results: ProfileResults) -> Self {
        let mut plot = Plot::new();
        let mut layout = Layout::new();

        for (i, result) in results.0.iter().enumerate() {
            let sup_name = to_html_sub(&result.param_name);
            let markers_trace = Scatter::new(result.param_values.clone(), result.ratios.clone())
                .name(format!("Likelihood Ratio of {}", &sup_name))
                .mode(Mode::Markers)
                .x_axis(format!("x{}", i + 1))
                .y_axis(format!("y{}", i + 1));

            let ratios = result
                .ratios
                .clone()
                .into_iter()
                .map(|r| if r < 0.1 { 0.0 } else { r })
                .collect::<Vec<f64>>();
            let (query_values, interpolated_ratios) =
                interpolate_ratios(&ratios, &result.param_values);
            let line_trace = Scatter::new(query_values, interpolated_ratios)
                .name(format!("Interpolated Likelihood Ratio of {}", &sup_name))
                .mode(Mode::Lines)
                .x_axis(format!("x{}", i + 1))
                .y_axis(format!("y{}", i + 1))
                .line(
                    Line::new()
                        .width(1.3)
                        .dash(plotly::common::DashType::DashDot)
                        .color("gray"),
                );

            plot.add_trace(line_trace);
            plot.add_trace(markers_trace);

            layout.add_annotation(
                Annotation::new()
                    .y_ref(format!("y{} domain", i + 1))
                    .y_anchor(Anchor::Bottom)
                    .y(1)
                    .text(&sup_name)
                    .x_ref(format!("x{} domain", i + 1))
                    .x_anchor(Anchor::Center)
                    .x(0.5)
                    .font(Font::new().size(18))
                    .show_arrow(false),
            );

            let x_title = format!("Parameter Value of {}", &sup_name);
            let y_title = format!("Likelihood Ratio of {}", &sup_name);

            layout = match i + 1 {
                1 => layout
                    .x_axis(Axis::new().title(Title::from(x_title)))
                    .y_axis(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                2 => layout
                    .x_axis2(Axis::new().title(Title::from(x_title)))
                    .y_axis2(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                3 => layout
                    .x_axis3(Axis::new().title(Title::from(x_title)))
                    .y_axis3(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                4 => layout
                    .x_axis4(Axis::new().title(Title::from(x_title)))
                    .y_axis4(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                5 => layout
                    .x_axis5(Axis::new().title(Title::from(x_title)))
                    .y_axis5(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                6 => layout
                    .x_axis6(Axis::new().title(Title::from(x_title)))
                    .y_axis6(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                7 => layout
                    .x_axis7(Axis::new().title(Title::from(x_title)))
                    .y_axis7(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                8 => layout
                    .x_axis8(Axis::new().title(Title::from(x_title)))
                    .y_axis8(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                _ => layout,
            };
        }

        // Determine the number of columns based on the number of parameters
        // There should be two columns, and the number of rows should be the number of parameters
        let n_params = results.0.len();
        let n_cols = 2;
        let n_rows = n_params.div_ceil(n_cols);

        layout = layout
            .grid(
                LayoutGrid::new()
                    .columns(n_cols)
                    .rows(n_rows)
                    .pattern(GridPattern::Independent),
            )
            .height(DEFAULT_HEIGHT * n_rows)
            .width(DEFAULT_WIDTH * n_cols);

        plot.set_layout(layout);

        plot
    }
}

fn interpolate_ratios(ratios: &[f64], param_values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let cs = CubicSpline::from_nodes(param_values, ratios).unwrap();
    let query_values = linspace!(param_values[0], param_values[param_values.len() - 1], 50);
    let interpolated_ratios = cs
        .eval_vec(&query_values)
        .into_iter()
        .map(|x| x.max(0.01))
        .collect::<Vec<f64>>();
    (query_values, interpolated_ratios)
}

/// Converts parameter names with underscores to HTML with subscripts
///
/// For example, "k_cat" becomes "k<sub>cat</sub>"
///
/// # Arguments
///
/// * `s` - The parameter name to convert
///
/// # Returns
///
/// The converted parameter name
///
/// # Notes
///
/// - This function converts parameter names with underscores to HTML with subscripts.
/// - For example, "k_cat" becomes "k<sub>cat</sub>".
/// - If the parameter name does not contain an underscore, it is returned unchanged.
fn to_html_sub(s: &str) -> String {
    if !s.contains('_') {
        return s.to_string();
    }

    let mut parts = s.splitn(2, '_');
    let base = parts.next().unwrap_or("");
    let superscript = parts.next().unwrap_or("");

    if superscript.is_empty() {
        base.to_string()
    } else {
        format!("{}<sub>{}</sub>", base, superscript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        io::load_enzmldoc,
        optim::{ProblemBuilder, SR1TrustRegionBuilder, SubProblem, Transformation},
        prelude::{EnzymeMLDocument, LossFunction, NegativeLogLikelihood},
    };
    use approx::assert_relative_eq;
    use peroxide::fuga::RK4;

    #[test]
    fn test_profile_likelihood() {
        // ARRANGE
        let mut doc = get_doc();
        doc.derive_system().expect("Failed to derive system");

        let problem = ProblemBuilder::new(
            &doc,
            RK4,
            LossFunction::NLL(NegativeLogLikelihood::new(1.0)),
        )
        .dt(10.0)
        .transform(Transformation::Log("k_cat".into()))
        .transform(Transformation::Log("k_ie".into()))
        .transform(Transformation::Log("K_M".into()))
        .build()
        .expect("Failed to build problem");

        // ACT
        let sr1trustregion = SR1TrustRegionBuilder::default()
            .max_iters(50)
            .subproblem(SubProblem::Steihaug)
            .build();

        let inits = Array1::from_vec(vec![80.0, 0.83, 0.0009]);

        let res = profile_likelihood()
            .problem(&problem)
            .initial_guess(inits)
            .parameters(vec![ProfileParameter::builder()
                .name("K_M")
                .from(50.0)
                .to(100.0)
                .build()])
            .n_steps(100)
            .optimizer(&sr1trustregion)
            .call()
            .expect("Failed to profile likelihood");

        // ASSERT
        assert_eq!(res.len(), 1);

        let profile = res.first().unwrap();
        assert_relative_eq!(profile.best_value, 82.0, epsilon = 1.2);
        assert_eq!(profile.likelihoods().len(), 100);
        assert_eq!(profile.param_values().len(), 100);
        assert_eq!(profile.ratios().len(), 100);
    }

    #[test]
    fn test_ego_profile_likelihood() {
        // ARRANGE
        let mut doc = get_doc();
        doc.derive_system().expect("Failed to derive system");

        let problem = ProblemBuilder::new(
            &doc,
            RK4,
            LossFunction::NLL(NegativeLogLikelihood::new(1.0)),
        )
        .dt(10.0)
        .transform(Transformation::Log("k_cat".into()))
        .transform(Transformation::Log("k_ie".into()))
        .transform(Transformation::Log("K_M".into()))
        .build()
        .expect("Failed to build problem");

        // ACT
        let sr1trustregion = SR1TrustRegionBuilder::default()
            .max_iters(50)
            .subproblem(SubProblem::Steihaug)
            .build();

        let inits = Array1::from_vec(vec![80.0, 0.83, 0.0009]);

        let result = ego_profile_likelihood()
            .problem(&problem)
            .initial_guess(inits)
            .max_iters(10)
            .parameters(vec![ProfileParameter::builder()
                .name("K_M")
                .from(10.0)
                .to(300.0)
                .build()])
            .optimizer(&sr1trustregion)
            .call()
            .expect("Failed to profile likelihood");

        let profile = result.first().unwrap();
        assert_relative_eq!(profile.best_value, 82.0, epsilon = 1.2);
    }

    #[test]
    fn test_profile_parameter_parse() {
        let input = "K_M=50.0:100.0";
        let param = ProfileParameter::from_str(input).expect("Should not fail");
        assert_eq!(param.name, "K_M");
        assert_eq!(param.from, 50.0);
        assert_eq!(param.to, 100.0);

        let input = "K_1M=50.0:100.0";
        let param = ProfileParameter::from_str(input).expect("Should not fail");
        assert_eq!(param.name, "K_1M");
        assert_eq!(param.from, 50.0);
        assert_eq!(param.to, 100.0);
    }

    #[test]
    fn test_profile_parameter_parse_error() {
        let input = "K_M=50.0:100.0:200.0";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {} should fail", input);

        let input = "K_M=50,0:100,0";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {} should fail", input);

        let input = "K_M=50.0,100.0:";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {} should fail", input);
    }

    fn get_doc() -> EnzymeMLDocument {
        load_enzmldoc("tests/data/enzmldoc_reaction.json").expect("Failed to load enzmldoc")
    }
}
