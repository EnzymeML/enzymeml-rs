use std::collections::HashMap;

use super::{
    initials::prepare_initials, observer::CallbackObserver, problem::Problem,
    report::OptimizationReport,
};
use crate::{optim::error::OptimizeError, prelude::Initials};
use argmin::core::{observers::ObserverMode, Executor, State};
use argmin_observer_slog::SlogLogger;

/// Input types for providing initial conditions to the optimization
///
/// # Variants
/// * `Initials` - Direct input of an Initials struct
/// * `Mapping` - HashMap mapping parameter names to their initial values
pub enum InitialsInput {
    Initials(Initials),
    Mapping(HashMap<String, f64>),
}

impl From<InitialsInput> for Initials {
    /// Converts InitialsInput into an Initials struct
    ///
    /// # Arguments
    /// * `input` - The InitialsInput to convert
    ///
    /// # Returns
    /// An Initials struct containing the initial conditions
    fn from(input: InitialsInput) -> Self {
        match input {
            InitialsInput::Initials(initials) => initials,
            InitialsInput::Mapping(mapping) => prepare_initials(mapping),
        }
    }
}

/// Runs the optimization problem to find optimal parameter values
///
/// This function:
/// 1. Sets up any parameter transformations
/// 2. Prepares initial conditions
/// 3. Configures and runs the optimization solver
/// 4. Retrieves and transforms the best parameters found
///
/// # Arguments
/// * `problem` - The optimization Problem containing model and solver configuration
///
/// # Returns
/// * `Result<(), OptimizeError>` - Ok(()) on success, or an error if optimization fails
///
/// # Errors
/// Returns OptimizeError if:
/// * Parameter transformations fail to set up
/// * The solver fails to execute
/// * Parameter transformations cannot be applied to results
pub fn optimize(
    mut problem: Problem,
    callback: Option<CallbackObserver>,
    show_progress: bool,
) -> Result<OptimizationReport, OptimizeError> {
    // Keep a copy of the original document for the report
    let doc = problem.doc.clone();

    // Add temporary transformations to the document
    problem.setup_transformations()?;

    let initials: Initials = problem.initials();
    let solver = problem.solver().setup()?;

    // Run solver with panic handling
    // TODO: Get rid of the panic handling and check for errors instead in optim/system.rs
    let res = std::panic::catch_unwind(|| {
        let mut executor = Executor::new(problem.clone(), solver)
            .configure(|state| state.param(initials).max_iters(100));

        // Add observer if provided
        if let Some(observer) = callback {
            executor = executor.add_observer(observer, ObserverMode::Always);
        }

        // Add progress observer if requested
        if show_progress {
            executor = executor.add_observer(SlogLogger::term(), ObserverMode::Always);
        }

        executor.run()
    })
    .map_err(|_| OptimizeError::SolverPanic)?
    .map_err(OptimizeError::ArgMinError)?;

    match res.state.get_best_param() {
        Some(best_params) => Ok(OptimizationReport::new(problem, doc, best_params)?),
        None => Err(OptimizeError::NoSolution),
    }
}
