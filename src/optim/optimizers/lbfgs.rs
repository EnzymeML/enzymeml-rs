//! L-BFGS optimization algorithm implementation.
//!
//! This module provides an implementation of the Limited-memory BFGS (L-BFGS) optimization algorithm,
//! which is a quasi-Newton method for solving unconstrained optimization problems. The implementation
//! includes:
//!
//! - The main `LBFGS` optimizer struct and implementation
//! - A builder pattern via `LBFGSBuilder` for convenient configuration
//! - Support for line search parameters and convergence criteria
//! - Observer pattern for monitoring optimization progress
//!
//! The L-BFGS algorithm approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using
//! a limited amount of memory, making it suitable for large-scale optimization problems.

use argmin::core::observers::ObserverMode;
use argmin::core::Executor;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS as ArgminLBFGS;
use argmin_observer_slog::SlogLogger;
use ndarray::Array1;
use peroxide::fuga::ODEIntegrator;

use crate::optim::{InitialGuesses, OptimizeError, Optimizer, Problem};

/// Implementation of the L-BFGS optimization algorithm.
///
/// L-BFGS (Limited-memory BFGS) is a quasi-Newton method for solving unconstrained
/// optimization problems that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS)
/// algorithm using a limited amount of memory.
///
/// # Type Parameters
///
/// * `I` - The state type that implements the `State` trait
/// * `O` - The observer type that implements the `Observe` trait
pub struct LBFGS {
    /// The history size for storing previous iterations
    pub m: usize,
    /// Maximum number of iterations before stopping
    pub max_iters: u64,
    /// Target cost function value for convergence criteria
    pub target_cost: f64,
    /// Line search parameter c1 (sufficient decrease condition)
    pub c1: f64,
    /// Line search parameter c2 (curvature condition)
    pub c2: f64,
}

impl LBFGS {
    /// Creates a new LBFGS optimizer instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `c1` - Line search parameter for sufficient decrease condition
    /// * `c2` - Line search parameter for curvature condition
    /// * `m` - History size for storing previous iterations
    /// * `max_iters` - Maximum number of iterations
    /// * `target_cost` - Target cost function value for convergence
    /// * `observers` - List of observers for monitoring progress
    pub fn new(c1: f64, c2: f64, m: usize, max_iters: u64, target_cost: f64) -> Self {
        Self {
            c1,
            c2,
            m,
            max_iters,
            target_cost,
        }
    }
}

impl<S: ODEIntegrator + Copy> Optimizer<S> for LBFGS {
    /// Optimizes the given problem using the L-BFGS algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    /// * `initial_guess` - Initial parameter values to start optimization from
    ///
    /// # Returns
    ///
    /// * `Ok(Array1<f64>)` - The optimal parameters if optimization succeeds
    /// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
    fn optimize<T>(
        &self,
        problem: &Problem<S>,
        initial_guess: Option<T>,
    ) -> Result<Array1<f64>, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        let initial_guess = initial_guess.ok_or(OptimizeError::MissingInitialGuesses {
            missing: vec!["all".to_string()],
        })?;

        let initial_guess = initial_guess.into().get_values();
        let linesearch = MoreThuenteLineSearch::new()
            .with_c(self.c1, self.c2)
            .unwrap();
        let solver = ArgminLBFGS::new(linesearch, self.m);
        let res = Executor::new(problem.clone(), solver)
            .configure(|state| {
                state
                    .param(initial_guess)
                    .max_iters(self.max_iters)
                    .target_cost(self.target_cost)
            })
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .unwrap();

        res.state
            .get_best_param()
            .cloned()
            .ok_or(OptimizeError::ConvergenceError)
    }
}

/// Builder for configuring and constructing LBFGS instances.
///
/// This builder provides a fluent interface for setting up LBFGS optimizer instances
/// with custom parameters and configuration options.
///
/// # Type Parameters
///
/// * `I` - The state type that implements the `State` trait
/// * `O` - The observer type that implements the `Observe` trait
pub struct LBFGSBuilder {
    /// Line search parameter c1 for sufficient decrease condition
    c1: f64,
    /// Line search parameter c2 for curvature condition
    c2: f64,
    /// Maximum number of iterations before stopping
    max_iters: u64,
    /// Target cost function value for convergence criteria
    target_cost: f64,
    /// The history size for storing previous iterations
    m: usize,
}

impl LBFGSBuilder {
    /// Sets the line search parameters.
    ///
    /// # Arguments
    ///
    /// * `c1` - Sufficient decrease parameter (0 < c1 < c2 < 1)
    /// * `c2` - Curvature condition parameter (c1 < c2 < 1)
    pub fn linesearch(mut self, c1: f64, c2: f64) -> Self {
        self.c1 = c1;
        self.c2 = c2;
        self
    }

    /// Sets the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the target cost function value for convergence.
    ///
    /// # Arguments
    ///
    /// * `target_cost` - Target cost value below which optimization stops
    pub fn target_cost(mut self, target_cost: f64) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Builds and returns an LBFGS instance with the configured settings.
    ///
    /// # Returns
    ///
    /// A new LBFGS optimizer instance with all configured parameters
    pub fn build(self) -> LBFGS {
        LBFGS {
            c1: self.c1,
            c2: self.c2,
            m: self.m,
            max_iters: self.max_iters,
            target_cost: self.target_cost,
        }
    }
}

impl Default for LBFGSBuilder {
    /// Creates a new LBFGSBuilder with default settings.
    ///
    /// Default values:
    /// - c1: 1e-4 (sufficient decrease parameter)
    /// - c2: 0.9 (curvature condition parameter)
    /// - m: 5 (history size)
    /// - max_iters: 500
    /// - target_cost: 1e-6
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            m: 5,
            max_iters: 500,
            target_cost: 1e-6,
        }
    }
}
