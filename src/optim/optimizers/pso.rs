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
//!
//! v_i(t+1) = w * v_i(t) + c1 * r1 * (p_i - x_i(t)) + c2 * r2 * (g_i - x_i(t))
//! x_i(t+1) = x_i(t) + v_i(t+1)
//!
//! where:
//! - v_i(t) is the velocity of particle i at time t
//! - x_i(t) is the position of particle i at time t
//! - p_i is the best position of particle i
//! - g_i is the best position of the entire swarm
//! - r1 and r2 are random numbers between 0 and 1
//! - c1 and c2 are the cognitive and social parameters, respectively
//! - w is the inertia weight
//! - r1 and r2 are random numbers between 0 and 1
//! - c1 and c2 are the cognitive and social parameters, respectively
//! - w is the inertia weight

use argmin::core::observers::ObserverMode;
use argmin::core::Executor;
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_observer_slog::SlogLogger;
use ndarray::Array1;

use crate::optim::{InitialGuesses, OptimizeError, Optimizer, Problem};

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
    /// Lower bound for the parameters
    pub lower_bound: Array1<f64>,
    /// Upper bound for the parameters
    pub upper_bound: Array1<f64>,
}

impl ParticleSwarmOpt {
    /// Creates a new BFGS optimizer instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `pop_size` - Size of the population
    /// * `max_iters` - Maximum number of iterations
    /// * `lower_bound` - Lower bound for the parameters
    /// * `upper_bound` - Upper bound for the parameters
    pub fn new(
        pop_size: usize,
        max_iters: u64,
        lower_bound: Array1<f64>,
        upper_bound: Array1<f64>,
    ) -> Self {
        Self {
            pop_size,
            max_iters,
            lower_bound,
            upper_bound,
        }
    }
}

impl Optimizer for ParticleSwarmOpt {
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
    fn optimize<T>(&self, problem: &Problem, _: Option<T>) -> Result<Array1<f64>, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        let bounds = (self.lower_bound.clone(), self.upper_bound.clone());

        let solver = ParticleSwarm::new(bounds, self.pop_size);
        let mut res: argmin::core::OptimizationResult<Problem, _, _> =
            Executor::new(problem.clone(), solver)
                .configure(|state| state.max_iters(self.max_iters))
                .add_observer(SlogLogger::term(), ObserverMode::Always)
                .run()
                .unwrap();

        let best_individual = res.state.take_best_individual();
        let best_individual = best_individual.ok_or(OptimizeError::ConvergenceError)?;
        Ok(best_individual.position)
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
    /// Lower bound for the parameters
    lower_bound: Array1<f64>,
    /// Upper bound for the parameters
    upper_bound: Array1<f64>,
}

impl PSOBuilder {
    /// Creates a new PSOBuilder with default settings.
    ///
    /// Default values:
    /// - max_iters: 500
    /// - pop_size: 100
    pub fn default(lower_bound: Array1<f64>, upper_bound: Array1<f64>) -> Self {
        Self {
            max_iters: 500,
            pop_size: 100,
            lower_bound,
            upper_bound,
        }
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
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
        }
    }
}
