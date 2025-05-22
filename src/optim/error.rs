use thiserror::Error;

use crate::{
    conversion::ConversionError, objective::error::ObjectiveError,
    simulation::error::SimulationError,
};

#[derive(Error, Debug)]
pub enum OptimizeError {
    #[error("Solver panic")]
    SolverPanic,
    #[error("Missing initial values for parameters: {missing:?}")]
    MissingInitialValues { missing: Vec<String> },
    #[error("Error optimizing")]
    ArgMinError(argmin::core::Error),
    #[error("No time data found in measurement")]
    MissingTimeData,
    #[error("Failed to simulate with given parameters")]
    SimulationError(#[from] SimulationError),
    #[error(
        "Failed to build equation for transformation {variable} of {transformation}: {message}"
    )]
    TransformationError {
        variable: String,
        transformation: String,
        message: String,
    },
    #[error("Parameter {param} not found: {message}")]
    ParameterNotFound { param: String, message: String },
    #[error("Species data not found for {0}")]
    SpeciesDataNotFound(String),
    #[error("No solution found")]
    NoSolution,
    #[error("Non-homogenous times")]
    NonHomogenousTimes,
    #[error("No time points found")]
    NoTimePoints,
    #[error("No measurement found for species {0}")]
    NoMeasurement(String),
    #[error("No measurement data found for measurement {0}")]
    NoMeasurementData(String),
    #[error("Measurement data has wrong shape")]
    MeasurementShapeError(#[from] ndarray::ShapeError),
    #[error("Sensitivities not found")]
    SensitivitiesNotFound,
    #[error("Non-homogenous observations")]
    NonHomogenousObservations,
    #[error("Initial guess array has wrong length")]
    InitialGuessLengthError { expected: usize, found: usize },
    #[error("Missing initial guesses for parameters: {missing:?}")]
    MissingInitialGuesses { missing: Vec<String> },
    #[error("Problem did not converge")]
    ConvergenceError,
    #[error("Invalid bounds")]
    InvalidBounds {
        expected: Vec<String>,
        found: Vec<String>,
    },
    #[error("Failed to convert measurement data")]
    ConversionError(#[from] ConversionError),
    #[error("Missing lower bound for parameter {param}")]
    MissingLowerBound { param: String },
    #[error("Missing upper bound for parameter {param}")]
    MissingUpperBound { param: String },
    #[error("Cost is NaN")]
    CostNaN,
    #[error("Hessian is not invertible")]
    HessianNotInvertible,
    #[error("Hessian can not be converted to ndarray")]
    HessianNotConvertibleToNdarray,
    #[error("Unknown parameter {0}")]
    UnknownParameter(String),
    #[error("Failed to calculate log probability")]
    ObjectiveError(#[from] ObjectiveError),
    #[error("Failed to parse profile parameter {0}. Expected format: 'param_name=from,to'.")]
    ProfileParameterParseError(String),
}
