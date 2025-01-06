use thiserror::Error;

use crate::prelude::error::SimulationError;

#[derive(Error, Debug)]
pub enum OptimizeError {
    #[error("Solver panic")]
    SolverPanic,
    #[error("Missing initial values for parameters: {missing:?}")]
    MissingInitialValues { missing: Vec<String> },
    #[error("Error optimizing")]
    ArgMinError(argmin::core::Error),
    #[error("Failed to convert measurement to array format")]
    MeasurementConversionError(#[from] Box<dyn std::error::Error>),
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
    #[error("Species data not found for {0}")]
    SpeciesDataNotFound(String),
    #[error("No solution found")]
    NoSolution,
}
