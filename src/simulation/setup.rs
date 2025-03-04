//! Setup module for configuring ODE simulations.
//!
//! This module provides the [`SimulationSetup`] struct and its builder for configuring
//! numerical integration parameters used in ODE simulations. It handles:
//!
//! - Time range specification (start and end times)
//! - Integration step size
//! - Error tolerance settings (relative and absolute)
//!
//! The configuration can be created programmatically or derived from measurement data
//! in EnzymeML documents.

use std::{collections::HashMap, error::Error};

use derive_builder::Builder;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::prelude::{EnzymeMLDocument, Measurement};

use super::error::SimulationError;

/// Configuration for numerical integration of ODE systems
///
/// This struct contains all the parameters needed to control the numerical integration
/// of an ODE system, including time range, step size, and error tolerances.
///
/// # Fields
///
/// * `t0` - Start time of the simulation (default: 0.0)
/// * `t1` - End time of the simulation (default: 10.0)
/// * `dt` - Time step size for output points (default: 1.0)
/// * `rtol` - Relative tolerance for error control (default: 1e-4)
/// * `atol` - Absolute tolerance for error control (default: 1e-8)
///
/// # Examples
///
/// ```
/// use enzymeml::prelude::SimulationSetupBuilder;
///
/// let setup = SimulationSetupBuilder::default()
///     .t0(0.0)
///     .t1(100.0)
///     .dt(0.1)
///     .rtol(1e-6)
///     .atol(1e-8)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder, Serialize, Deserialize)]
pub struct SimulationSetup {
    #[builder(default = "0.0")]
    pub t0: f64,
    #[builder(default = "10.0")]
    pub t1: f64,
    #[builder(default = "1.0")]
    pub dt: f64,
    #[builder(default = "1e-4")]
    pub rtol: f64,
    #[builder(default = "1e-8")]
    pub atol: f64,
}

impl SimulationSetup {
    /// Merges simulation settings from another setup into this one
    ///
    /// Updates the time step and tolerance settings of this setup with values from
    /// the provided setup, while preserving the time range (t0, t1).
    ///
    /// # Arguments
    /// * `other` - The SimulationSetup to merge settings from
    ///
    /// # Examples
    ///
    /// ```
    /// use enzymeml::prelude::SimulationSetupBuilder;
    ///
    /// let mut setup1 = SimulationSetupBuilder::default()
    ///     .t0(0.0)
    ///     .t1(10.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let setup2 = SimulationSetupBuilder::default()
    ///     .dt(0.1)
    ///     .rtol(1e-6)
    ///     .build()
    ///     .unwrap();
    ///
    /// setup1.merge(&setup2);
    /// ```
    pub fn merge(&mut self, other: &SimulationSetup) {
        self.dt = other.dt;
        self.rtol = other.rtol;
        self.atol = other.atol;
    }
}

/// Converts an EnzymeML document into a map of simulation setups
///
/// This implementation creates simulation setups for each measurement in the model,
/// using the earliest and latest time points from the measurement data to set the
/// simulation time range.
///
/// # Arguments
///
/// * `model` - The EnzymeML document containing measurements to create setups for
///
/// # Returns
///
/// Returns a Result containing either:
/// * A HashMap mapping measurement IDs to their corresponding SimulationSetup
/// * An error if time data could not be extracted from any measurement
impl TryFrom<EnzymeMLDocument> for HashMap<String, SimulationSetup> {
    type Error = Box<dyn Error>;

    fn try_from(model: EnzymeMLDocument) -> Result<Self, Self::Error> {
        let mut setups = HashMap::new();

        for measurement in model.measurements.iter() {
            let t0 = get_t0(measurement)?;
            let t1 = get_t1(measurement)?;

            setups.insert(
                measurement.id.clone(),
                SimulationSetupBuilder::default()
                    .t0(t0)
                    .t1(t1)
                    .build()
                    .unwrap(),
            );
        }

        Ok(setups)
    }
}

/// Converts an EnzymeML document into a vector of simulation setups
///
/// This implementation creates simulation setups for each measurement in the model,
/// using the earliest and latest time points from the measurement data to set the
/// simulation time range.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document containing measurements to create setups for
///
/// # Returns
///
/// Returns a Result containing either:
/// * A Vec of SimulationSetups, one for each measurement in the document
/// * An error if time data could not be extracted from any measurement
impl TryFrom<&EnzymeMLDocument> for Vec<SimulationSetup> {
    type Error = SimulationError;

    fn try_from(enzmldoc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        enzmldoc
            .get_measurements()
            .iter()
            .map(|m| {
                m.try_into()
                    .map_err(|_| SimulationError::ConversionError(m.id.clone()))
            })
            .collect()
    }
}

/// Converts a measurement into a simulation setup
///
/// This implementation creates a simulation setup using the earliest and latest time points
/// from the measurement data to set the simulation time range.
///
/// # Arguments
///
/// * `measurement` - The measurement containing time series data to create a setup for
///
/// # Returns
///
/// Returns a Result containing either:
/// * A SimulationSetup configured with the measurement's time range
/// * An error if time data could not be extracted from the measurement
impl TryFrom<&Measurement> for SimulationSetup {
    type Error = Box<dyn Error>;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        let t0 = get_t0(measurement)?;
        let t1 = get_t1(measurement)?;

        Ok(SimulationSetupBuilder::default()
            .t0(t0)
            .t1(t1)
            .build()
            .unwrap())
    }
}

/// Gets the earliest time point across all species in a measurement
///
/// Examines all time series data in the measurement and finds the globally earliest
/// time point across all species.
///
/// # Arguments
///
/// * `measurement` - The measurement containing time series data for multiple species
///
/// # Returns
///
/// Returns a Result containing either:
/// * The earliest time point as an f64
/// * An error if no time data is found in the measurement
///
/// # Errors
///
/// Returns an error if:
/// * No species in the measurement have time data
/// * All time series in the measurement are empty
fn get_t0(measurement: &Measurement) -> Result<f64, Box<dyn Error>> {
    let mut min_times: BTreeSet<OrderedFloat<f64>> = BTreeSet::new();
    for species in measurement.species_data.iter() {
        if let Some(time) = &species.time {
            let mut min_time = BTreeSet::new();

            for t in time.iter() {
                min_time.insert(OrderedFloat(*t));
            }

            if min_time.is_empty() {
                continue;
            }

            min_times.insert(*min_time.iter().min().unwrap());
        }
    }

    if min_times.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No time data found in measurement",
        )));
    }

    Ok(min_times.iter().min().unwrap().into_inner())
}

/// Gets the latest time point across all species in a measurement
///
/// Examines all time series data in the measurement and finds the globally latest
/// time point across all species.
///
/// # Arguments
///
/// * `measurement` - The measurement containing time series data for multiple species
///
/// # Returns
///
/// Returns a Result containing either:
/// * The latest time point as an f64
/// * An error if no time data is found in the measurement
///
/// # Errors
///
/// Returns an error if:
/// * No species in the measurement have time data
/// * All time series in the measurement are empty
fn get_t1(measurement: &Measurement) -> Result<f64, Box<dyn Error>> {
    let mut max_times: BTreeSet<OrderedFloat<f64>> = BTreeSet::new();

    for species in measurement.species_data.iter() {
        if let Some(time) = &species.time {
            let mut max_time = BTreeSet::new();

            for t in time.iter() {
                max_time.insert(OrderedFloat(*t));
            }

            if max_time.is_empty() {
                continue;
            }

            max_times.insert(*max_time.iter().max().unwrap());
        }
    }

    if max_times.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No time data found in measurement",
        )));
    }

    Ok(max_times.iter().max().unwrap().into_inner())
}

#[cfg(test)]
mod tests {
    use crate::prelude::MeasurementData;

    use super::*;

    #[test]
    fn test_get_t0() {
        let mut measurement = Measurement::default();
        let mut species = MeasurementData::default();

        species.time = Some(vec![0.0, 1.0, 2.0]);
        measurement.species_data.push(species);

        let t0 = get_t0(&measurement).unwrap();
        assert_eq!(t0, 0.0);
    }

    #[test]
    #[should_panic]
    fn test_get_t0_none() {
        let mut measurement = Measurement::default();
        let mut species = MeasurementData::default();

        species.time = None;
        measurement.species_data.push(species);

        get_t0(&measurement).expect("No time data found in measurement");
    }

    #[test]
    fn test_get_t1() {
        let mut measurement = Measurement::default();
        let mut species = MeasurementData::default();

        species.time = Some(vec![0.0, 1.0, 2.0]);
        measurement.species_data.push(species);

        let t1 = get_t1(&measurement).unwrap();
        assert_eq!(t1, 2.0);
    }

    #[test]
    #[should_panic]
    fn test_get_t1_none() {
        let mut measurement = Measurement::default();
        let mut species = MeasurementData::default();

        species.time = None;
        measurement.species_data.push(species);

        get_t1(&measurement).expect("No time data found in measurement");
    }
}
