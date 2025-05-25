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

use peroxide::fuga::ODEIntegrator;
use rayon::prelude::*;

use crate::{
    optim::{InitialGuesses, OptimizeError, Optimizer, Problem},
    prelude::ObjectiveFunction,
};

use super::grid::ProfileGrid;
use super::parameter::ProfileParameter;
use super::results::{ProfileResult, ProfileResults};
use super::utils::{
    best_parameter_val, compute_likelihood_ratio, get_param_indices, prepare_profile_problem,
};

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

    let results = profile_likelihood_core(problem, initial_guess, &parameters, n_steps, optimizer)?;

    Ok(results)
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
    parameters: &[ProfileParameter],
    n_steps: usize,
    optimizer: &O,
) -> Result<ProfileResults, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone,
{
    // Prepare the problem
    let (err_min, initial_guess) =
        prepare_profile_problem(problem, initial_guess, parameters, optimizer)?;

    // Then, profile the likelihood
    let grid = ProfileGrid::from_profile_parameters(parameters, n_steps);
    let param_indices = get_param_indices(problem, parameters)?;

    // Parallel computation of log probabilities
    let profile = grid
        .par_iter()
        .map(|(_, param_values)| {
            compute_likelihood_ratio(
                problem,
                &initial_guess,
                parameters,
                param_values,
                optimizer,
                err_min,
                &param_indices,
            )
        })
        .collect::<Result<Vec<_>, OptimizeError>>()?;

    // Pre-allocate all vectors with the exact size needed
    let grid_len = grid.len();
    let mut param_values = vec![Vec::with_capacity(grid_len); parameters.len()];
    let mut likelihoods = Vec::with_capacity(grid_len);
    let mut ratios = Vec::with_capacity(grid_len);

    // Extract and organize the data in a single pass
    for (values, likelihood, ratio) in profile {
        for (i, &value) in values.iter().enumerate() {
            param_values[i].push(value);
        }
        likelihoods.push(likelihood);
        ratios.push(ratio);
    }

    // Create profile results for each parameter
    let mut results = Vec::with_capacity(parameters.len());
    for i in 0..parameters.len() {
        let best_value = best_parameter_val(&param_values[i], &ratios);
        results.push(ProfileResult {
            best_value,
            param_name: parameters[i].name.clone(),
            param_values: param_values[i].clone(),
            likelihoods: likelihoods.clone(),
            ratios: ratios.clone(),
        });
    }

    Ok(ProfileResults(results))
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
    use ndarray::Array1;
    use peroxide::fuga::RK4;

    #[test]
    fn test_profile_likelihood() {
        // ARRANGE
        let doc = get_doc();

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

    fn get_doc() -> EnzymeMLDocument {
        load_enzmldoc("tests/data/enzmldoc_reaction.json").expect("Failed to load enzmldoc")
    }
}
