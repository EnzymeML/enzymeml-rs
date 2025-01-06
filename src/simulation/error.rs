use evalexpr::EvalexprError;
use ode_solvers::dop_shared::IntegrationError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Error evaluating expression")]
    EvalExpressionError(EvalexprError),
    #[error("Error integrating ODEs: {0}")]
    IntegrationError(IntegrationError),
    #[error("Error creating ODE system")]
    ODESystemError(String),
    #[error("Error validating parameters")]
    ValidateParametersError(String),
    #[error("Error collecting results")]
    CollectResultsError(String),
    #[error("Error calculating initial assignments")]
    CalculateInitialAssignmentsError(String),
    #[error("Other error: {0}")]
    Other(String),
}
