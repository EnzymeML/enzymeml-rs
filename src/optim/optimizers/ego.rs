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

use crate::optim::{InitialGuesses, OptimizeError, Optimizer, Problem};

/// Implementation of the Efficient Global Optimization algorithm.
///
/// EGO is a surrogate-based optimization algorithm that is particularly well-suited
/// for expensive black-box optimization problems. It builds a surrogate model of the
/// objective function and uses it to make intelligent decisions about where to sample next.
pub struct EfficientGlobalOptimization {
    /// Bounds for the optimization parameters, stored as an Nx2 array where N is the number
    /// of parameters and each row contains (lower_bound, upper_bound) for that parameter
    bounds: Array2<f64>,
    /// Maximum number of iterations before stopping
    max_iters: usize,
}

impl EfficientGlobalOptimization {
    /// Creates a new EGO optimizer instance with the specified parameter bounds.
    ///
    /// # Arguments
    ///
    /// * `bounds` - An Nx2 array specifying the bounds for each parameter, where N is the
    ///             number of parameters and each row contains (lower_bound, upper_bound)
    pub fn new(bounds: Array2<f64>, max_iters: usize) -> Self {
        Self { bounds, max_iters }
    }
}

impl Optimizer for EfficientGlobalOptimization {
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
    fn optimize<T>(&self, problem: &Problem, _: Option<T>) -> Result<Array1<f64>, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        let objective = |params: &ArrayView2<f64>| {
            // Preallocate results vector with capacity
            let mut results = Vec::with_capacity(params.nrows());

            // Iterate over rows and evaluate cost function
            for row in params.axis_iter(ndarray::Axis(0)) {
                results.push(problem.cost(&row.to_owned()).unwrap());
            }

            // Convert to Array2 in one allocation
            Array2::from_shape_vec((results.len(), 1), results).unwrap()
        };

        let result = EgorBuilder::optimize(objective)
            .configure(|config| config.max_iters(self.max_iters))
            .min_within(&self.bounds)
            .run()
            .map_err(|_| OptimizeError::ConvergenceError)?;
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
    bounds: Array2<f64>,
    /// Maximum number of iterations before stopping
    max_iters: usize,
}

impl EGOBuilder {
    /// Creates a new EGOBuilder with the specified parameter bounds.
    ///
    /// # Arguments
    ///
    /// * `bounds` - An Nx2 array specifying bounds for each parameter, where N is the
    ///             number of parameters and each row contains (lower_bound, upper_bound)
    pub fn new(bounds: Array2<f64>) -> Self {
        Self {
            bounds,
            max_iters: 100,
        }
    }

    /// Sets the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
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
