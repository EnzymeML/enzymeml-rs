//! Output Module for ODE Simulation Results
//!
//! This module provides functionality for transforming and formatting ODE simulation results
//! into different output structures. It defines the [`OutputFormat`] trait and implements
//! specific output formats like [`MatrixResult`].
//!
//! # Key Components
//!
//! - [`OutputFormat`] trait: Defines a standard interface for creating simulation outputs
//! - [`MatrixResult`] struct: Provides a matrix-based representation of simulation results
//!
//! # Features
//!
//! - Converts simulation results into structured formats
//! - Supports optional assignment and parameter sensitivity outputs
//! - Flexible output generation based on simulation system configuration
//!
//! # Usage
//!
//! The module allows converting raw simulation data into more convenient formats
//! for analysis, visualization, and further processing.

use ndarray::{Array1, Array2, Array3};

use super::{
    result::{SimulationResult, TimeSeriesMapping},
    system::{ODESystem, StepperOutput},
};

/// Trait to specify the output format and structure
pub trait OutputFormat {
    /// The type of output this format produces
    type Output;

    /// Creates the output in the specified format
    fn create_output(
        y_out: StepperOutput,
        assignments_out: Option<StepperOutput>,
        times: Array1<f64>,
        system: &ODESystem,
        has_sensitivities: bool,
    ) -> Self::Output;
}

/// Type alias for matrix output results
#[derive(Debug, Clone)]
pub struct MatrixResult {
    pub times: Array1<f64>,
    pub species: Array2<f64>,
    pub parameter_sensitivities: Option<Array3<f64>>,
    pub assignments: Option<Array2<f64>>,
}

impl OutputFormat for MatrixResult {
    type Output = MatrixResult;

    fn create_output(
        y_out: StepperOutput,
        assignments_out: Option<StepperOutput>,
        times: Array1<f64>,
        system: &ODESystem,
        has_sensitivities: bool,
    ) -> Self::Output {
        let n_rows = y_out.len();
        let species_len = system.num_equations();
        let parameters_len = system.num_parameters();

        // Calculate parameter sensitivities if needed
        let parameter_sensitivities = if has_sensitivities {
            let mut sensitivities = Array3::zeros((n_rows, species_len, parameters_len));
            for (t, row) in y_out.iter().enumerate() {
                for i in 0..species_len {
                    for j in 0..parameters_len {
                        let idx = species_len + i * parameters_len + j;
                        sensitivities[[t, i, j]] = row[idx];
                    }
                }
            }
            Some(sensitivities)
        } else {
            None
        };

        if let Some(_assignments_out) = assignments_out {
            todo!("Implement assignments");
        }

        // Convert species data to matrix
        let mut species_matrix = Array2::zeros((n_rows, species_len));
        for (i, row) in y_out.iter().enumerate() {
            for j in 0..species_len {
                species_matrix[(i, j)] = row[j];
            }
        }

        MatrixResult {
            times,
            species: species_matrix,
            parameter_sensitivities,
            assignments: None,
        }
    }
}

impl OutputFormat for SimulationResult {
    type Output = SimulationResult;

    fn create_output(
        y_out: StepperOutput,
        assignments_out: Option<StepperOutput>,
        times: Array1<f64>,
        system: &ODESystem,
        has_sensitivities: bool,
    ) -> Self::Output {
        // Add species to the result
        let mut species = TimeSeriesMapping::new();
        for (i, name) in system.get_sorted_species().iter().enumerate() {
            let values = y_out.iter().map(|row| row[i]).collect();
            species.insert(name.clone(), values);
        }

        if has_sensitivities {
            todo!("Implement parameter sensitivities");
        }

        let assignments = if let Some(assignments_out) = assignments_out {
            let mut assignments = TimeSeriesMapping::new();
            for (i, name) in system.get_sorted_assignments().iter().enumerate() {
                let values = assignments_out.iter().map(|row| row[i]).collect();
                assignments.insert(name.clone(), values);
            }
            Some(assignments)
        } else {
            None
        };

        SimulationResult {
            time: times.to_vec(),
            species,
            assignments,
            parameter_sensitivities: None,
        }
    }
}
