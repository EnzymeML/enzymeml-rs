//! Conversion traits and functions for measurement data.
//!
//! This module provides conversion functionality between EnzymeML measurement data
//! and ndarray types, including:
//!
//! - Converting EnzymeML documents to 2D arrays of measurement data
//! - Converting individual measurements to 2D arrays
//! - Helper functions for extracting and formatting measurement data

use ndarray::{Array2, Axis};

use crate::{
    optim::error::OptimizeError,
    prelude::{EnzymeMLDocument, Measurement},
};

/// Converts an EnzymeML document into a 2D array of measurement data.
///
/// Concatenates all measurements in the document into a single array along axis 0.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document containing measurements
///
/// # Returns
///
/// * `Result<Array2<f64>, OptimizeError>` - The concatenated measurement data or an error
///
/// # Errors
///
/// Returns an error if:
/// - No measurement data is found
/// - Measurements cannot be concatenated due to shape mismatch
impl TryFrom<&EnzymeMLDocument> for Array2<f64> {
    type Error = OptimizeError;

    fn try_from(enzmldoc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        let arrays: Result<Vec<Array2<f64>>, _> =
            enzmldoc.measurements.iter().map(|m| m.try_into()).collect();
        let arrays = arrays?;

        if arrays.is_empty() {
            return Err(OptimizeError::NoMeasurementData("all".to_string()));
        }

        let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();

        ndarray::concatenate(Axis(0), &views).map_err(OptimizeError::MeasurementShapeError)
    }
}

/// Converts a single measurement into a 2D array of measurement data.
///
/// Extracts data for all species that have measurements and formats it into a 2D array.
///
/// # Arguments
///
/// * `measurement` - The measurement containing species data
///
/// # Returns
///
/// * `Result<Array2<f64>, OptimizeError>` - The measurement data array or an error
///
/// # Errors
///
/// Returns an error if no measurement data is found for any species
impl TryFrom<&Measurement> for Array2<f64> {
    type Error = OptimizeError;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        let species_w_data = measurement
            .species_data
            .iter()
            .filter(|s| s.data.is_some())
            .map(|s| s.species_id.clone())
            .collect::<Vec<_>>();

        get_measurement_data(measurement, &species_w_data)
    }
}

/// Helper function to extract and format measurement data for specified species.
///
/// # Arguments
///
/// * `measurements` - The measurement containing the data
/// * `species` - List of species IDs to extract data for
///
/// # Returns
///
/// * `Result<Array2<f64>, OptimizeError>` - 2D array with species data or an error
///
/// # Errors
///
/// Returns an error if:
/// - No data is found for a specified species
/// - The measurement contains no data
fn get_measurement_data(
    measurements: &Measurement,
    species: &[String],
) -> Result<Array2<f64>, OptimizeError> {
    let mut data = Vec::with_capacity(species.len());

    for species_id in species {
        let meas_data = measurements
            .species_data
            .iter()
            .find(|s| s.species_id == *species_id)
            .ok_or_else(|| OptimizeError::NoMeasurement(species_id.clone()))?;

        let data_vec = meas_data
            .data
            .as_ref()
            .ok_or_else(|| OptimizeError::NoMeasurement(species_id.clone()))?;

        data.push(data_vec);
    }

    let ncols = data.len();
    if ncols == 0 {
        return Err(OptimizeError::NoMeasurementData(measurements.id.clone()));
    }

    let nrows = data[0].len();

    // Create array directly from collected data
    Ok(Array2::from_shape_fn((nrows, ncols), |(i, j)| data[j][i]))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::io::load_enzmldoc;

    use super::*;

    #[test]
    fn test_try_from_enzmldoc() {
        // Arrange
        let path = PathBuf::from("tests/data/enzmldoc.json");
        let enzmldoc = load_enzmldoc(&path).unwrap();

        // Act
        let result = Array2::try_from(&enzmldoc);

        // Assert
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.shape(), &[264, 1]);
    }

    #[test]
    fn test_try_from_measurement() {
        // Arrange
        let path = PathBuf::from("tests/data/enzmldoc.json");
        let enzmldoc = load_enzmldoc(&path).unwrap();

        // Act
        let result = Array2::try_from(&enzmldoc.measurements[0]);

        // Assert
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.shape(), &[11, 1]);
    }
}
