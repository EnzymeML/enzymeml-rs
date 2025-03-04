//! Simulation Error Module
//!
//! This module provides a comprehensive error handling mechanism for ODE simulations.
//!
//! # Key Error Types
//!
//! The [`SimulationError`] enum covers various potential failure points in the simulation process:
//! - Equation evaluation errors
//! - ODE integration errors
//! - ODE system creation errors
//! - Parameter validation errors
//! - Result collection errors
//! - Initial assignment calculation errors
//! - Output type mismatches
//!
//! # Usage
//!
//! This error type can be used to provide detailed error information
//! when simulation processes encounter issues, allowing for precise
//! error diagnosis and handling.

use evalexpr_jit::errors::EquationError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Error evaluating expression: {0}")]
    EquationError(EquationError),
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
    #[error("Invalid output type: Expected '{0}'")]
    InvalidOutputType(String),
    #[error("Equation error: {0}")]
    AssigmentRecalculationError(#[from] EquationError),
    #[error("Failed to convert EnzymeMLDocument to {0}")]
    ConversionError(String),
    #[error("ArgMinMath error: {0}")]
    ArgMinMathError(#[from] argmin_math::Error),
    #[error("No data provided for interpolation")]
    NoDataForInterpolation,
}
