//! Particle Swarm Optimization (PSO) algorithm implementation.
//!
//! This module provides an implementation of the Particle Swarm Optimization (PSO) algorithm,
//! which is a population-based optimization algorithm for solving unconstrained optimization problems. The implementation
//! includes:
//!
//! - The main `ParticleSwarmOpt` optimizer struct and implementation
//! - A builder pattern via `PSOBuilder` for convenient configuration
//! - Observer pattern for monitoring optimization progress
//!
//! The PSO algorithm is a population-based optimization algorithm that uses a swarm of particles to search for the optimal solution.
//! Each particle has a position and a velocity, and the particles move in the search space according to the following equations:

use argmin::core::observers::ObserverMode;
use argmin::core::Executor;
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_observer_slog::SlogLogger;
use ndarray::s;
use peroxide::fuga::ODEIntegrator;

use crate::optim::{
    bounds_to_array2, report::OptimizationReport, Bound, InitialGuesses, OptimizeError, Optimizer,
    Problem,
};

use super::utils::transform_bounds;

/// Implementation of the BFGS optimization algorithm.
///
/// BFGS is a quasi-Newton method for solving unconstrained
/// optimization problems that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS)
/// algorithm using a limited amount of memory.
///
/// # Type Parameters
///
/// * `I` - The state type that implements the `State` trait
/// * `O` - The observer type that implements the `Observe` trait
pub struct ParticleSwarmOpt {
    /// Maximum number of iterations before stopping
    pub pop_size: usize,
    /// Target cost function value for convergence criteria
    pub max_iters: u64,
    /// Parameter bounds
    pub bounds: Vec<Bound>,
}

impl ParticleSwarmOpt {
    /// Creates a new BFGS optimizer instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `pop_size` - Size of the population
    /// * `max_iters` - Maximum number of iterations
    /// * `bounds` - Parameter bounds
    pub fn new(pop_size: usize, max_iters: u64, bounds: Vec<Bound>) -> Self {
        Self {
            pop_size,
            max_iters,
            bounds,
        }
    }
}

impl<S: ODEIntegrator + Copy> Optimizer<S> for ParticleSwarmOpt {
    /// Optimizes the given problem using the PSO algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    /// * `initial_guess` - Initial parameter values to start optimization from. Not used for PSO.
    ///
    /// # Returns
    ///
    /// * `Ok(Array1<f64>)` - The optimal parameters if optimization succeeds
    /// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
    fn optimize<T>(
        &self,
        problem: &Problem<S>,
        _: Option<T>,
    ) -> Result<OptimizationReport, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        // Get bounds and transform them based on the transformations
        let mut bounds = self.bounds.clone();
        transform_bounds(&mut bounds, problem.transformations());

        // Split bounds into (Array1<f64>, Array1<f64>)
        let bounds = bounds_to_array2(problem, &bounds)?;
        let lower_bound = bounds.slice(s![.., 0]);
        let upper_bound = bounds.slice(s![.., 1]);
        let bounds = (lower_bound.to_owned(), upper_bound.to_owned());

        // Create PSO solver
        let solver = ParticleSwarm::new(bounds, self.pop_size);

        // Run optimization
        let mut res: argmin::core::OptimizationResult<Problem<S>, _, _> =
            Executor::new(problem.clone(), solver)
                .configure(|state| state.max_iters(self.max_iters))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();

        let best_individual = res
            .state
            .take_best_individual()
            .ok_or(OptimizeError::ConvergenceError)?;

        OptimizationReport::new(
            problem,
            problem.enzmldoc().clone(),
            &best_individual.position.to_vec(),
            None,
            Some(self.bounds.clone()),
        )
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
pub struct PSOBuilder {
    /// Maximum number of iterations before stopping
    max_iters: u64,
    /// Population size
    pop_size: usize,
    /// Parameter bounds
    bounds: Vec<Bound>,
}

impl PSOBuilder {
    /// Sets the parameter bounds.
    ///
    /// # Arguments
    ///
    /// * `bounds` - Parameter bounds
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

    /// Sets the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the population size.
    ///
    /// # Arguments
    ///
    /// * `pop_size` - Population size
    pub fn pop_size(mut self, pop_size: usize) -> Self {
        self.pop_size = pop_size;
        self
    }

    /// Builds and returns an LBFGS instance with the configured settings.
    ///
    /// # Returns
    ///
    /// A new LBFGS optimizer instance with all configured parameters
    pub fn build(self) -> ParticleSwarmOpt {
        ParticleSwarmOpt {
            max_iters: self.max_iters,
            pop_size: self.pop_size,
            bounds: self.bounds,
        }
    }
}

impl Default for PSOBuilder {
    fn default() -> Self {
        Self {
            max_iters: 500,
            pop_size: 100,
            bounds: vec![],
        }
    }
}
