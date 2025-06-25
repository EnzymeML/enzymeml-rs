use argmin::core::{TerminationReason, TerminationStatus};
use egobox_ego::EgorBuilder;
use ndarray::{Array2, ArrayView2};
use peroxide::fuga::ODEIntegrator;

use crate::{
    optim::{InitialGuesses, OptimizeError, Optimizer, Problem},
    prelude::ObjectiveFunction,
};

use super::{
    results::{ProfileResult, ProfileResults},
    utils::{
        best_parameter_val, compute_likelihood_ratio, get_param_indices, likelihood_ratio,
        prepare_profile_problem,
    },
    ProfileParameter,
};

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
    let mut parameters: Vec<ProfileParameter> = parameters.into_iter().map(|p| p.into()).collect();

    // Sort parameters by name
    parameters.sort_by(|a, b| a.name.cmp(&b.name));

    let results = ego_profile_likelihood_core(
        problem,
        initial_guess.clone(),
        &parameters,
        optimizer,
        max_iters,
    )?;

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
    parameters: &[ProfileParameter],
    optimizer: &O,
    max_iters: usize,
) -> Result<Vec<ProfileResult>, OptimizeError>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Send + Sync,
    O: Optimizer<S, L> + Sync,
    T: Into<InitialGuesses> + Clone + Send + Sync,
{
    // Prepare the problem and get initial optimization result
    let (err_min, initial_guess) =
        prepare_profile_problem(problem, initial_guess, parameters, optimizer)?;

    let param_indices = get_param_indices(problem, parameters)?;

    // Define objective function for EGO optimization
    let objective = |param_value: &ArrayView2<f64>| {
        // Pre-allocate vector with exact capacity
        let n_rows = param_value.nrows();
        let mut likelihoods = Vec::with_capacity(n_rows);

        for row_idx in 0..n_rows {
            // Convert log-parameters back to original scale
            let params: Vec<f64> = param_value.row(row_idx).iter().copied().collect();

            // Compute likelihood ratio for this parameter set
            match compute_likelihood_ratio(
                problem,
                &initial_guess,
                parameters,
                &params,
                optimizer,
                err_min,
                &param_indices,
            ) {
                Ok((_, likelihood, _)) => likelihoods.push(likelihood),
                Err(e) => {
                    // Handle error gracefully with a warning instead of panic
                    eprintln!("Warning: Failed to compute likelihood ratio: {}", e);
                    likelihoods.push(f64::MAX); // Use worst possible value
                }
            }
        }

        // Return as a column vector (required format for EGO)
        Array2::from_shape_vec((n_rows, 1), likelihoods).unwrap()
    };

    // Set up parameter bounds (log-transformed)
    let bounds = Array2::from_shape_vec(
        (parameters.len(), 2),
        parameters.iter().flat_map(|p| [p.from, p.to]).collect(),
    )
    .unwrap();

    // Run EGO optimization
    let result = EgorBuilder::optimize(objective)
        .configure(|config| {
            config
                .max_iters(max_iters)
                .infill_strategy(egobox_ego::InfillStrategy::EI)
        })
        .min_within(&bounds)
        .run()
        .map_err(|_| OptimizeError::ConvergenceError)?;

    // Check for invalid termination
    if let TerminationStatus::Terminated(TerminationReason::SolverExit(_)) =
        result.state.termination_status
    {
        return Err(OptimizeError::CostNaN);
    }

    // Extract results
    let likelihoods: Vec<f64> = result.y_doe.as_slice().unwrap().to_vec();
    let ratios: Vec<f64> = likelihoods
        .iter()
        .map(|&err_profile| likelihood_ratio(err_min, err_profile))
        .collect();

    // Create profile results for each parameter
    let mut profile_results = Vec::with_capacity(parameters.len());
    for (col, parameter) in parameters.iter().enumerate().take(result.x_doe.ncols()) {
        let param_values: Vec<f64> = result.x_doe.column(col).iter().copied().collect();
        let best_value = best_parameter_val(&param_values, &ratios);
        profile_results.push(ProfileResult {
            best_value,
            param_name: parameter.name.clone(),
            param_values,
            likelihoods: likelihoods.clone(),
            ratios: ratios.clone(),
        });
    }

    Ok(profile_results)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array1;
    use peroxide::fuga::RK4;

    use crate::{
        io::load_enzmldoc,
        optim::{ProblemBuilder, SR1TrustRegionBuilder, SubProblem, Transformation},
        prelude::{EnzymeMLDocument, LossFunction, NegativeLogLikelihood},
    };

    use super::*;

    #[test]
    fn test_ego_profile_likelihood() {
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
        assert_relative_eq!(profile.best_value, 82.0, epsilon = 5.0);
    }

    fn get_doc() -> EnzymeMLDocument {
        load_enzmldoc("tests/data/enzmldoc_reaction.json").expect("Failed to load enzmldoc")
    }
}
