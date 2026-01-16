//! Interpolation Module for Simulation Results
//!
//! This module provides functionality for interpolating simulation data across different time points.
//! It supports both cubic (Catmull-Rom) and linear interpolation strategies to ensure robust
//! value estimation between simulation time points.
//!
//! # Key Features
//!
//! - Cubic spline interpolation with fallback to linear interpolation
//! - Handles interpolation of multi-dimensional simulation data
//! - Supports flexible time point querying
//!
//! # Main Functions
//!
//! - [`interpolate`]: Primary function for interpolating simulation results
//! - Supports interpolation of vector-based simulation outputs
//!
//! # Usage
//!
//! The module is typically used to resample simulation results at specific time points,
//! enabling more flexible analysis and visualization of simulation data.

use splines::{Interpolation, Key, Spline};

use super::{error::SimulationError, system::StepperOutput};

/// Interpolates values at specified query times using both cubic and linear splines.
///
/// If cubic interpolation fails, linear interpolation is used. This is to ensure that the
/// interpolation is always valid. The reason cubic interpolation fails is that the cubic
/// spline is not defined at the end points of the simulation data, but the linear spline is.
///
/// # Arguments
///
/// * `data` - Vector of matrices containing the simulation data
/// * `times` - Vector of time points corresponding to the simulation data
/// * `query_times` - Vector of time points at which to interpolate values
///
/// # Returns
///
/// A vector of matrices where each matrix corresponds to a query time and contains the interpolated
/// values for all variables at that time point. The interpolation uses cubic splines where possible,
/// falling back to linear interpolation when cubic interpolation fails.
pub fn interpolate(
    data: &[Vec<f64>],
    times: &[f64],
    query_times: &[f64],
) -> Result<StepperOutput, SimulationError> {
    // Setup the splines
    let cubic_splines = setup_splines(data, times, Interpolation::CatmullRom)?;
    let linear_splines = setup_splines(data, times, Interpolation::Linear)?;

    // Interpolate the values
    let mut interpolated_values: StepperOutput = Vec::with_capacity(query_times.len());
    for t in query_times {
        let mut interpolated_values_row = Vec::with_capacity(linear_splines.len());
        for (linear_spline, cubic_spline) in linear_splines.iter().zip(cubic_splines.iter()) {
            match cubic_spline.sample(*t) {
                Some(value) => interpolated_values_row.push(value),
                None => interpolated_values_row.push(linear_spline.clamped_sample(*t).unwrap()),
            }
        }
        interpolated_values.push(interpolated_values_row);
    }

    Ok(interpolated_values)
}

/// Creates a vector of splines, one for each column of simulation data.
///
/// # Arguments
///
/// * `sim_data` - Matrix containing simulation data, where each column represents a different variable
/// * `sim_times` - Vector of simulation time points corresponding to each row in sim_data
/// * `interpol` - The interpolation method to use for the splines (e.g. Linear, CatmullRom)
///
/// # Returns
///
/// A vector of splines, where each spline interpolates one column of the simulation data
#[allow(clippy::needless_range_loop)]
fn setup_splines(
    sim_data: &[Vec<f64>],
    sim_times: &[f64],
    interpol: Interpolation<f64, f64>,
) -> Result<Vec<Spline<f64, f64>>, SimulationError> {
    // Return a spline for each column
    let n_species = sim_data
        .first()
        .ok_or(SimulationError::NoDataForInterpolation)?
        .len();
    let mut splines = Vec::with_capacity(n_species);

    for species in 0..n_species {
        let keys: Vec<_> = sim_times
            .iter()
            .enumerate()
            .map(|(i, &time)| Key::new(time, sim_data[i][species], interpol))
            .collect();
        splines.push(Spline::from_vec(keys));
    }

    Ok(splines)
}
