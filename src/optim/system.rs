//! This module provides optimization functionality for EnzymeML documents, including cost function
//! calculation and gradient computation for parameter optimization.

use std::collections::{HashMap, HashSet};

use argmin::core::{CostFunction, Gradient, Hessian};
use finitediff::FiniteDiff;
use ndarray::{Array1, Array2, Axis};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::prelude::{
    error::SimulationError, result::SimulationResult, simulate, EnzymeMLDocument, Measurement,
};

use super::problem::Problem;

/// Implementation of the CostFunction trait for Problem to enable parameter optimization.
/// This allows using the Problem with optimization algorithms that minimize a cost function.
impl CostFunction for Problem {
    type Param = Array1<f64>;
    type Output = f64;

    /// Calculates the cost between simulated and measured data.
    /// The cost is computed by running simulations with the provided parameters and comparing
    /// the results to experimental measurements.
    ///
    /// # Arguments
    /// * `params` - Vector of parameter values to test in the simulation
    ///
    /// # Returns
    /// * `Result<f64, argmin::core::Error>` - The calculated error according to the objective function
    ///
    /// # Details
    /// The function performs the following steps:
    /// 1. Maps the parameter vector to named parameters
    /// 2. Simulates the system with the given parameters in parallel chunks
    /// 3. Computes the error according to the objective function
    fn cost(&self, params: &Self::Param) -> Result<f64, argmin::core::Error> {
        // Assemble model parameters
        let mut model_params: Vec<String> =
            self.doc.parameters.iter().map(|p| p.id.clone()).collect();
        model_params.sort_unstable();

        let mut parameters: HashMap<String, f64> = HashMap::with_capacity(params.len());
        parameters.extend(
            params
                .iter()
                .zip(&model_params)
                .map(|(p, m)| (m.clone(), *p)),
        );

        // Parallelize simulation and residual calculation in a single pass
        let residuals = get_residuals(&self.doc, &parameters);

        let objective = self
            .objective
            .cost(&residuals, self.get_n_points())
            .unwrap();

        Ok(objective)
    }
}

/// Implementation of the Gradient trait for Problem to enable gradient-based optimization.
/// The gradient is computed using central finite differences for better accuracy.
impl Gradient for Problem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    /// Computes the gradient of the cost function with respect to the parameters using central differences.
    ///
    /// # Arguments
    /// * `params` - Vector of parameter values at which to evaluate the gradient
    ///
    /// # Returns
    /// * `Result<Array1<f64>, argmin::core::Error>` - The computed gradient vector or an error
    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // Create a closure with bounded parameters
        let cost_fn = |x: &Array1<f64>| self.cost(x).unwrap();

        // Use central differences instead of forward differences for better accuracy
        let gradient = params.central_diff(&cost_fn);
        // let gradient = gradient.mapv(|g| g.clamp(-1e1, 1e1));

        Ok(gradient)
    }
}

/// Implementation of the Hessian trait for Problem to enable second-order optimization methods.
/// The Hessian is computed using central finite differences for better accuracy.
///
/// The Hessian matrix contains second-order partial derivatives of the cost function
/// with respect to pairs of parameters. It provides curvature information that can
/// help optimization algorithms like Newton's method converge more quickly.
///
/// # Implementation Details
/// - Uses central finite differences to approximate second derivatives
/// - Computes the full n x n Hessian matrix where n is the number of parameters
/// - Leverages the gradient method to compute directional derivatives
///
/// # Arguments
/// * `params` - Vector of parameter values at which to evaluate the Hessian
///
/// # Returns
/// * `Result<Array2<f64>, argmin::core::Error>` - The computed Hessian matrix or an error
impl Hessian for Problem {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, params: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        // Create a closure with bounded parameters
        let cost_fn = |x: &Array1<f64>| self.gradient(x).unwrap();

        // Use central differences instead of forward differences for better accuracy
        let hessian = params.central_hessian(&cost_fn);

        Ok(hessian)
    }
}

/// Calculates residuals between measured and simulated data for all measurements in parallel
///
/// This function:
/// 1. Splits measurements into chunks of 4 for parallel processing
/// 2. Simulates each chunk using the provided parameters
/// 3. Calculates residuals between measured and simulated data
/// 4. Flattens and combines all residuals into a single vector
///
/// # Arguments
/// * `doc` - The EnzymeML document containing model definition and measurements
/// * `parameters` - Map of parameter names to values for simulation
///
/// # Returns
/// * `Vec<Array2<f64>>` - Vector of 2D arrays containing residuals for each measurement
///
/// # Panics
/// * If measurement chunk processing fails
/// * If residual calculation fails for any measurement
pub fn get_residuals(doc: &EnzymeMLDocument, parameters: &HashMap<String, f64>) -> Array2<f64> {
    let residuals: Vec<Array2<f64>> = doc
        .measurements
        .par_chunks(4)
        .map(|chunk| {
            // Process chunk and calculate residuals immediately
            let chunk_results = process_measurement_chunk(doc, chunk, parameters)
                .expect("Error processing measurement chunk");
            let chunk_residuals = chunk_results
                .into_iter()
                .zip(chunk.iter())
                .map(|(r, m)| calculate_residuals(m, &r))
                .collect::<Vec<_>>();
            Ok::<_, SimulationError>(chunk_residuals)
        })
        .collect::<Result<Vec<_>, SimulationError>>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();

    let view = residuals.iter().map(|r| r.view()).collect::<Vec<_>>();
    ndarray::concatenate(Axis(0), &view).unwrap()
}

/// Processes a chunk of measurements in parallel to produce simulation results
///
/// # Arguments
/// * `doc` - The EnzymeML document containing the model definition
/// * `chunk` - Vector of measurement references to simulate
/// * `parameters` - Map of parameter names to values for the simulation
///
/// # Returns
/// * `Result<Vec<SimulationResult>, Box<dyn std::error::Error>>` - Vector of simulation results for each measurement or an error
fn process_measurement_chunk(
    doc: &EnzymeMLDocument,
    chunk: &[Measurement],
    parameters: &HashMap<String, f64>,
) -> Result<Vec<SimulationResult>, SimulationError> {
    let results = chunk
        .par_iter()
        .map(|m| {
            simulate(
                doc,
                m.into(),
                m.try_into().unwrap(),
                Some(parameters.clone()),
                get_time_points(m),
            )
        })
        .collect::<Result<Vec<_>, SimulationError>>();

    match results {
        Ok(results) => Ok(results.into_iter().flatten().collect()),
        Err(e) => Err(e),
    }
}

/// Calculates the residuals between measured and simulated data
///
/// # Arguments
/// * `measurement` - The measurement containing experimental data
/// * `result` - The simulation result to compare against
///
/// # Returns
/// * `Array2<f64>` - The residuals between measured and simulated data points
fn calculate_residuals(measurement: &Measurement, result: &SimulationResult) -> Array2<f64> {
    let measured = Array2::try_from(measurement).unwrap();
    let simulated = convert_result_to_array(result, measurement);
    measured - simulated
}

/// Converts a SimulationResult into a 2D array based on species data from a Measurement.
///
/// This function takes simulation results and a measurement specification, extracts the relevant
/// species data, and organizes it into a 2D array where each row represents a timepoint and
/// each column represents a species.
///
/// # Arguments
/// * `result` - The SimulationResult containing the simulated data
/// * `measurement` - The Measurement containing species specifications
///
/// # Returns
/// An Array2<f64> where:
/// - Rows correspond to timepoints
/// - Columns correspond to species (sorted by species ID)
/// - Values are the species concentrations at each timepoint
fn convert_result_to_array(result: &SimulationResult, measurement: &Measurement) -> Array2<f64> {
    // First find all species that contain data and then sort them by species id
    let mut species_names = measurement
        .species_data
        .iter()
        .filter_map(|s| {
            if s.data.is_some() && s.time.is_some() {
                Some(s.species_id.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    species_names.sort_unstable();

    // Determine the number of timepoints
    let time_len = result.time.len();

    let mut array = Array2::zeros((time_len, species_names.len()));

    for (j, species_id) in species_names.iter().enumerate() {
        let data = result.get_species_data(species_id);
        for i in 0..time_len {
            array[[i, j]] = data[i];
        }
    }

    array
}

/// Gets the time points from a measurement by finding the first species with time data
///
/// # Arguments
/// * `measurement` - The measurement to extract time points from
///
/// # Returns
/// * `Option<&Vec<f64>>` - Reference to the vector of time points if found
fn get_time_points(measurement: &Measurement) -> Option<&Vec<f64>> {
    measurement
        .species_data
        .iter()
        .find(|s| s.time.is_some())
        .unwrap()
        .time
        .as_ref()
}

/// Implementation of TryFrom trait to convert Measurement into a 2D array.
/// Converts measurement data into a 2D array where rows represent timepoints
/// and columns represent different species.
impl TryFrom<&Measurement> for ndarray::Array2<f64> {
    type Error = Box<dyn std::error::Error>;

    /// Attempts to convert a Measurement into an Array2<f64>.
    /// The resulting array has dimensions [timepoints × species], where:
    /// - Each row corresponds to a timepoint
    /// - Each column corresponds to a species (sorted by ID)
    /// - Values represent measured concentrations
    ///
    /// # Arguments
    /// * `m` - Reference to a Measurement to convert
    ///
    /// # Returns
    /// * `Result<Array2<f64>, Box<dyn std::error::Error>>` - The converted 2D array or an error
    ///
    /// # Errors
    /// Returns an error if species have different numbers of timepoints or if required data is missing
    fn try_from(m: &Measurement) -> Result<Self, Self::Error> {
        let mut species_names = Vec::with_capacity(m.species_data.len());
        let mut time_lens = HashSet::with_capacity(1);

        for species in &m.species_data {
            if species.data.is_some() {
                species_names.push(&species.species_id);
                if let Some(time) = &species.time {
                    time_lens.insert(time.len());
                }
            }
        }

        species_names.sort_unstable();

        if time_lens.len() > 1 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "All species must have the same length of time points",
            )));
        }

        let time_len = time_lens.iter().next().unwrap();
        let mut array = Array2::zeros((*time_len, species_names.len()));

        for (j, species_id) in species_names.iter().enumerate() {
            let data = m
                .species_data
                .par_iter()
                .find_any(|s| &s.species_id == *species_id)
                .unwrap()
                .data
                .as_ref()
                .unwrap();
            let data_ref = data;
            for i in 0..*time_len {
                array[[i, j]] = data_ref[i];
            }
        }

        Ok(array)
    }
}
