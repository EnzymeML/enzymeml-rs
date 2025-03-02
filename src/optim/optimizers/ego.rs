//! Efficient Global Optimization (EGO) algorithm implementation.
//!
//! This module provides an implementation of the Efficient Global Optimization algorithm,
//! which is a surrogate-based optimization method particularly effective for expensive black-box
//! optimization problems. The implementation includes:
//!
//! - The main `EfficientGlobalOptimization` optimizer struct and implementation
//! - Integration with the egobox-ego crate for core EGO functionality
//! - Support for bound constraints on parameters
//!
//! The EGO algorithm works by building a surrogate model (typically a Gaussian Process)
//! of the objective function and using it to intelligently select new points for evaluation
//! based on an acquisition function that balances exploration and exploitation.

use argmin::core::CostFunction;
use egobox_ego::EgorBuilder;
use ndarray::{Array1, Array2, ArrayView2};
use peroxide::fuga::ODEIntegrator;

use crate::optim::{bounds_to_array2, Bound, InitialGuesses, OptimizeError, Optimizer, Problem};

/// Implementation of the Efficient Global Optimization algorithm.
///
/// EGO is a surrogate-based optimization algorithm that is particularly well-suited
/// for expensive black-box optimization problems. It builds a surrogate model of the
/// objective function and uses it to make intelligent decisions about where to sample next.
pub struct EfficientGlobalOptimization {
    /// Bounds for the optimization parameters, stored as an Nx2 array where N is the number
    /// of parameters and each row contains (lower_bound, upper_bound) for that parameter
    bounds: Vec<Bound>,
    /// Maximum number of iterations before stopping
    max_iters: usize,
}

impl EfficientGlobalOptimization {
    /// Creates a new EGO optimizer instance with the specified parameter bounds.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A vector of `Bound` structs specifying the bounds for each parameter
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn new(bounds: Vec<Bound>, max_iters: usize) -> Self {
        Self { bounds, max_iters }
    }
}

impl<S: ODEIntegrator + Copy> Optimizer<S> for EfficientGlobalOptimization {
    /// Optimizes the given problem using the EGO algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    /// * `_` - Initial parameter values (not used by EGO)
    ///
    /// # Returns
    ///
    /// * `Ok(Array1<f64>)` - The optimal parameters if optimization succeeds
    /// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
    fn optimize<T>(&self, problem: &Problem<S>, _: Option<T>) -> Result<Array1<f64>, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        // Closure to evaluate the objective function
        let objective = |params: &ArrayView2<f64>| {
            let mut results = Vec::with_capacity(params.nrows());
            for row in params.axis_iter(ndarray::Axis(0)) {
                results.push(problem.cost(&row.to_owned()).unwrap());
            }

            Array2::from_shape_vec((results.len(), 1), results).unwrap()
        };

        // Convert bounds to Array2
        let bounds = bounds_to_array2(&problem, &self.bounds)?;

        // Build and run EGO optimizer
        let result = EgorBuilder::optimize(objective)
            .configure(|config| config.max_iters(self.max_iters))
            .min_within(&bounds)
            .run()
            .map_err(|_| OptimizeError::ConvergenceError)?;

        // Return best parameters
        if let Some(params) = result.state.best_param {
            Ok(params)
        } else {
            Err(OptimizeError::ConvergenceError)
        }
    }
}

/// Builder for configuring and constructing EfficientGlobalOptimization instances.
///
/// This builder provides a fluent interface for setting up EGO optimizer instances
/// with custom parameters and configuration options.
pub struct EGOBuilder {
    /// Bounds for each parameter as an Nx2 array where N is the number of parameters
    /// and each row contains (lower_bound, upper_bound)
    bounds: Vec<Bound>,
    /// Maximum number of iterations before stopping
    max_iters: usize,
}

impl EGOBuilder {
    /// Sets the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the bounds for the optimization parameters.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A vector of `Bound` structs specifying the bounds for each parameter
    pub fn bounds(mut self, bounds: Vec<Bound>) -> Self {
        self.bounds = bounds;
        self
    }

    /// Sets the lower and upper bounds for a single parameter.
    ///
    /// # Arguments
    ///
    /// * `param` - The name of the parameter
    /// * `lower` - The lower bound for the parameter
    /// * `upper` - The upper bound for the parameter
    pub fn bound(mut self, param: &str, lower: f64, upper: f64) -> Self {
        self.bounds
            .push(Bound::new(param.to_string(), lower, upper));
        self
    }

    /// Builds and returns an EfficientGlobalOptimization instance with the configured settings.
    ///
    /// # Returns
    ///
    /// A new EfficientGlobalOptimization optimizer instance with all configured parameters
    pub fn build(self) -> EfficientGlobalOptimization {
        EfficientGlobalOptimization::new(self.bounds, self.max_iters)
    }
}

impl Default for EGOBuilder {
    fn default() -> Self {
        Self {
            bounds: vec![],
            max_iters: 100,
        }
    }
}
