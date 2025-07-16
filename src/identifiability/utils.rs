use ndarray::Array1;
use peroxide::fuga::ODEIntegrator;

use crate::{
    optim::{InitialGuesses, OptimizeError, Optimizer, Problem},
    prelude::ObjectiveFunction,
};

use super::ProfileParameter;

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
pub(crate) fn compute_likelihood_ratio<'a, S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: &T,
    parameter: &[ProfileParameter],
    param_values: &'a [f64],
    optimizer: &O,
    err_min: f64,
    param_indices: &[usize],
) -> Result<(&'a [f64], f64, f64), OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Fix the parameter in problem
    let mut problem = problem.clone();
    let mut parameters = parameter.to_vec();
    parameters.sort_by(|a, b| a.name.cmp(&b.name));

    // Fix all parameters at once
    for param in parameter {
        problem.fix_param(&param.name)?;
    }

    // Set the parameter values in the initial guess
    let mut initial_guess = initial_guess.clone().into();
    for (i, &value) in param_indices.iter().zip(param_values.iter()) {
        initial_guess.set_value_at(*i, value);
    }

    // Optimize the problem
    let report = optimizer.optimize(&problem, Some(initial_guess), None)?;
    let ratio = likelihood_ratio(err_min, report.error);

    Ok((param_values, report.error, ratio))
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
pub(crate) fn best_parameter_val(param_values: &[f64], ratios: &[f64]) -> f64 {
    let position_max = ratios
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();
    param_values[position_max]
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
pub(crate) fn likelihood_ratio(err_min: f64, err_profile: f64) -> f64 {
    (-0.5 * (err_profile - err_min)).exp()
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
pub(crate) fn get_param_indices<S, L>(
    problem: &Problem<S, L>,
    parameters: &[ProfileParameter],
) -> Result<Vec<usize>, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
{
    let sorted_params = problem.ode_system().get_sorted_params();
    let mut param_indices = Vec::new();
    for param in parameters {
        if !sorted_params.contains(&param.name) {
            return Err(OptimizeError::UnknownParameter(param.name.clone()));
        }
        param_indices.push(sorted_params.iter().position(|p| p == &param.name).unwrap());
    }

    Ok(param_indices)
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
pub(crate) fn prepare_profile_problem<S, L, O, T>(
    problem: &Problem<S, L>,
    initial_guess: T,
    parameters: &[ProfileParameter],
    optimizer: &O,
) -> Result<(f64, InitialGuesses), OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Check if the fixed parameter is in the problem
    let mut missing_params = Vec::new();
    for parameter in parameters {
        if !problem
            .ode_system()
            .get_sorted_params()
            .contains(&parameter.name)
        {
            missing_params.push(parameter.name.clone());
        }
    }
    if !missing_params.is_empty() {
        return Err(OptimizeError::UnknownParameter(missing_params.join(", ")));
    }

    // First, fit the problem to get the initial guess
    let initial_guess = initial_guess.into();
    let report = optimizer.optimize(problem, Some(initial_guess), None)?;
    let err_min = report.error;

    // Use the optimized parameters as the initial guess
    let initial_guess: InitialGuesses = Array1::from_vec(
        problem
            .ode_system()
            .get_sorted_params()
            .iter()
            .map(|param| {
                report.best_params.get(param).copied().unwrap_or_else(|| {
                    panic!("Parameter {param} not found in optimized parameters")
                })
            })
            .collect(),
    )
    .into();

    Ok((err_min, initial_guess))
}
