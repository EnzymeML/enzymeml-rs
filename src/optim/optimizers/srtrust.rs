//! SR1 Trust Region optimization algorithm implementation.
//!
//! This module provides an implementation of the Symmetric Rank-One (SR1) Trust Region optimization algorithm,
//! which is a quasi-Newton method for solving unconstrained optimization problems. The implementation
//! includes:
//!
//! - The main `SR1TrustRegion` optimizer struct and implementation
//! - A builder pattern via `SR1TrustRegionBuilder` for convenient configuration
//! - Support for different subproblem solvers (Steihaug and Cauchy point)
//! - Integration with the argmin optimization framework
//!
//! The SR1 Trust Region algorithm uses a symmetric rank-one update to approximate the Hessian matrix
//! and employs a trust region strategy to determine step sizes. This approach is particularly effective
//! for nonlinear optimization problems where line search methods might struggle with convergence.

use crate::optim::report::OptimizationReport;
use crate::optim::{InitialGuesses, OptimizeError, Optimizer, Problem};
use crate::prelude::ObjectiveFunction;
use argmin::core::observers::ObserverMode;
use argmin::core::Executor;
use argmin::core::State;
use argmin::core::TerminationStatus;
use argmin::solver::quasinewton::SR1TrustRegion as ArgminSR1TrustRegion;
use argmin::solver::trustregion::CauchyPoint;
use argmin::solver::trustregion::Steihaug;
use argmin_observer_slog::SlogLogger;
use clap::ValueEnum;
use ndarray::Array1;
use ndarray::Array2;
use peroxide::fuga::ODEIntegrator;

use super::utils::transform_initial_guesses;

/// Implementation of the SR1 Trust Region optimization algorithm.
///
/// The SR1 Trust Region method is a quasi-Newton optimization technique that uses
/// symmetric rank-one updates to approximate the Hessian matrix while constraining
/// steps to remain within a trust region. This approach provides robust convergence
/// properties for nonlinear optimization problems.
pub struct SR1TrustRegion {
    /// Maximum number of iterations before stopping
    pub max_iters: u64,

    /// Subproblem solver
    pub subproblem: SubProblem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SubProblem {
    /// Steihaug subproblem solver
    ///
    /// This solver implements the Steihaug-Toint conjugate gradient method for solving
    /// the trust-region subproblem. It is generally more accurate than the Cauchy point
    /// method and works well for large-scale problems.
    Steihaug,

    /// Cauchy point subproblem solver
    ///
    /// This solver computes the Cauchy point, which is the minimizer of the quadratic model
    /// along the steepest descent direction within the trust region. It is computationally
    /// less expensive than Steihaug but may produce less optimal steps.
    Cauchy,
}

impl SR1TrustRegion {
    /// Creates a new SR1TrustRegion optimizer instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations
    /// * `subproblem` - The subproblem solver to use (Steihaug or Cauchy point)
    pub fn new(max_iters: u64, subproblem: SubProblem) -> Self {
        Self {
            max_iters,
            subproblem,
        }
    }
}

impl<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction> Optimizer<S, L>
    for SR1TrustRegion
{
    /// Optimizes the given problem using the SR1 Trust Region algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    /// * `initial_guess` - Initial parameter values to start optimization from
    ///
    /// # Returns
    ///
    /// * `Ok(OptimizationReport)` - Report containing the optimal parameters and other information
    /// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
    fn optimize<T>(
        &self,
        problem: &Problem<S, L>,
        initial_guess: Option<T>,
    ) -> Result<OptimizationReport, OptimizeError>
    where
        T: Into<InitialGuesses>,
    {
        // Get the initial guesses
        let initial_guess = initial_guess.ok_or(OptimizeError::MissingInitialGuesses {
            missing: vec!["all".to_string()],
        })?;
        let mut initial_guess: InitialGuesses = initial_guess.into();

        // We need this for the report before transforming the initial guesses
        let init_guess_copy = initial_guess.clone();

        // Extract the initial guesses
        let init_hessian = Array2::eye(initial_guess.len());

        // Transform the initial guesses
        transform_initial_guesses(
            &problem.ode_system().get_sorted_params(),
            &mut initial_guess,
            problem.transformations(),
        );

        let best_params = match self.subproblem {
            SubProblem::Steihaug => {
                solve_steihaug(problem, initial_guess.clone(), self.max_iters, init_hessian)
            }
            SubProblem::Cauchy => {
                solve_cauchy_point(problem, initial_guess.clone(), self.max_iters, init_hessian)
            }
        }?;

        OptimizationReport::new(
            problem,
            problem.enzmldoc().clone(),
            &best_params.to_vec(),
            Some(init_guess_copy),
            None,
        )
    }
}

/// Solves the subproblem using the Steihaug method.
///
/// This function configures and runs the Steihaug-Toint conjugate gradient method
/// to solve the trust-region subproblem within the SR1 Trust Region algorithm.
///
/// # Arguments
///
/// * `problem` - The optimization problem to solve
/// * `initial_guess` - Initial parameter values to start optimization from
/// * `max_iters` - Maximum number of iterations
/// * `init_hessian` - Initial Hessian matrix
///
/// # Returns
///
/// * `Ok(Array1<f64>)` - The optimal parameters if optimization succeeds
/// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
fn solve_steihaug<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction>(
    problem: &Problem<S, L>,
    initial_guess: InitialGuesses,
    max_iters: u64,
    init_hessian: Array2<f64>,
) -> Result<Array1<f64>, OptimizeError> {
    let subproblem = Steihaug::new();
    let solver = ArgminSR1TrustRegion::new(subproblem);
    let res = Executor::new(problem.clone(), solver)
        .configure(|state| {
            state
                .param(initial_guess.get_values())
                .hessian(init_hessian)
                .max_iters(max_iters)
        })
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .map_err(OptimizeError::ArgMinError)?;

    if let TerminationStatus::Terminated(argmin::core::TerminationReason::SolverExit(_)) =
        res.state.termination_status
    {
        return Err(OptimizeError::CostNaN);
    }

    res.state
        .get_best_param()
        .cloned()
        .ok_or(OptimizeError::ConvergenceError)
}

/// Solves the subproblem using the Cauchy Point method.
///
/// This function configures and runs the Cauchy Point method to solve the
/// trust-region subproblem within the SR1 Trust Region algorithm. The Cauchy Point
/// is the minimizer of the quadratic model along the steepest descent direction
/// within the trust region.
///
/// # Arguments
///
/// * `problem` - The optimization problem to solve
/// * `initial_guess` - Initial parameter values to start optimization from
/// * `max_iters` - Maximum number of iterations
/// * `init_hessian` - Initial Hessian matrix
///
/// # Returns
///
/// * `Ok(Array1<f64>)` - The optimal parameters if optimization succeeds
/// * `Err(OptimizeError)` - Error if optimization fails or doesn't converge
fn solve_cauchy_point<S: ODEIntegrator + Copy + Send + Sync, L: ObjectiveFunction>(
    problem: &Problem<S, L>,
    initial_guess: InitialGuesses,
    max_iters: u64,
    init_hessian: Array2<f64>,
) -> Result<Array1<f64>, OptimizeError> {
    let subproblem = CauchyPoint::new();
    let solver = ArgminSR1TrustRegion::new(subproblem);
    let res = Executor::new(problem.clone(), solver)
        .configure(|state| {
            state
                .param(initial_guess.get_values())
                .hessian(init_hessian)
                .max_iters(max_iters)
        })
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .map_err(OptimizeError::ArgMinError)?;

    if let TerminationStatus::Terminated(argmin::core::TerminationReason::SolverExit(_)) =
        res.state.termination_status
    {
        return Err(OptimizeError::CostNaN);
    }

    res.state
        .get_best_param()
        .cloned()
        .ok_or(OptimizeError::ConvergenceError)
}

/// Builder for configuring and constructing SR1TrustRegion instances.
///
/// This builder provides a fluent interface for setting up SR1TrustRegion optimizer instances
/// with custom parameters and configuration options.
pub struct SR1TrustRegionBuilder {
    /// Maximum number of iterations before stopping
    max_iters: u64,
    /// Subproblem solver
    subproblem: SubProblem,
}

impl SR1TrustRegionBuilder {
    /// Sets the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of iterations before stopping
    pub fn max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the subproblem solver.
    ///
    /// # Arguments
    ///
    /// * `subproblem` - The subproblem solver to use (Steihaug or Cauchy point)
    pub fn subproblem(mut self, subproblem: SubProblem) -> Self {
        self.subproblem = subproblem;
        self
    }

    /// Builds and returns an SR1TrustRegion instance with the configured settings.
    ///
    /// # Returns
    ///
    /// A new SR1TrustRegion optimizer instance with all configured parameters
    pub fn build(self) -> SR1TrustRegion {
        SR1TrustRegion {
            max_iters: self.max_iters,
            subproblem: self.subproblem,
        }
    }
}

impl Default for SR1TrustRegionBuilder {
    /// Creates a new SR1TrustRegionBuilder with default settings.
    ///
    /// Default values:
    /// - max_iters: 50
    /// - subproblem: Steihaug
    fn default() -> Self {
        Self {
            max_iters: 50,
            subproblem: SubProblem::Steihaug,
        }
    }
}
