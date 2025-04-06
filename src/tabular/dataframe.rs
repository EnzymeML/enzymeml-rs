//! Tabular Data Conversion Module
//!
//! This module provides functionality for converting between `Measurement` and `DataFrame`
//! representations, enabling seamless data manipulation and analysis.
//!
//! # Key Features
//!
//! - Convert `Measurement` objects to Polars DataFrames
//! - Convert DataFrames back to `MeasurementBuilder`
//! - Handle time series and species data transformations
//!
//! # Conversion Methods
//!
//! - `From<Measurement>` and `From<&Measurement>` trait implementations for DataFrame conversion
//! - `to_dataframe()` method on `Measurement`
//! - `from_dataframe()` static method on `Measurement` and `MeasurementBuilder`
//!
//! # Data Handling
//!
//! The module supports:
//! - Extracting time series data
//! - Managing initial conditions
//! - Handling missing or placeholder data

use std::error::Error;

use polars::prelude::*;

use crate::prelude::{Measurement, MeasurementBuilder, MeasurementData, MeasurementDataBuilder};

/// Implements the conversion from a Measurement to a DataFrame.
impl From<Measurement> for DataFrame {
    /// Converts a Measurement into a DataFrame.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The Measurement to be converted.
    ///
    /// # Returns
    ///
    /// Returns a DataFrame representing the Measurement.
    fn from(measurement: Measurement) -> DataFrame {
        measurement_to_dataframe(&measurement)
    }
}

impl From<&Measurement> for DataFrame {
    /// Converts a Measurement into a DataFrame.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The Measurement to be converted.
    ///
    /// # Returns
    ///
    /// Returns a DataFrame representing the Measurement.
    fn from(measurement: &Measurement) -> DataFrame {
        measurement_to_dataframe(measurement)
    }
}

impl Measurement {
    /// Converts the Measurement into a DataFrame.
    ///
    /// # Returns
    ///
    /// Returns a DataFrame representing the Measurement.
    pub fn to_dataframe(&self) -> DataFrame {
        measurement_to_dataframe(self)
    }

    /// Converts a DataFrame into a MeasurementBuilder.
    ///
    /// # Arguments
    ///
    /// * `df` - The DataFrame to be converted.
    ///
    /// # Returns
    ///
    /// Returns a MeasurementBuilder representing the DataFrame.
    pub fn from_dataframe(df: &DataFrame) -> Result<MeasurementBuilder, Box<dyn Error>> {
        MeasurementBuilder::from_dataframe(df)
    }
}

impl MeasurementBuilder {
    /// Creates a `MeasurementBuilder` from a `DataFrame`.
    ///
    /// # Arguments
    ///
    /// * `df` - The `DataFrame` to be converted.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `MeasurementBuilder` or an error if the conversion fails.
    pub fn from_dataframe(df: &DataFrame) -> Result<MeasurementBuilder, Box<dyn Error>> {
        let time_series: Vec<f64> = df
            .column("time")
            .expect("Time column not found")
            .f64()
            .unwrap()
            .into_iter()
            .map(|opt| opt.unwrap_or_default())
            .collect();

        // Check if there is t=0
        if !time_series.is_empty() && time_series[0] != 0.0 {
            return Err("Time series must start with t=0".into());
        }

        // Iterate through the columns and create MeasurementData for each species
        let mut species_data = Vec::new();

        for column in df.get_columns() {
            if column.name() != "time" {
                let species_id = column.name().to_string();
                let mut time = time_series.clone();
                let mut data: Vec<f64> = column
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|opt| opt.unwrap_or_default())
                    .collect();

                let initial = data.first().copied().unwrap_or_default();

                if data[0] != -1.0 && data.iter().skip(1).all(|&x| x == -1.0) {
                    // If the first value is not -1 and all the rest are -1,
                    // set data to an empty vector
                    data.clear();
                    time.clear();
                } else if data.iter().all(|&x| x == -1.0) {
                    return Err(format!(
                        "Missing initial value at t=0 for species '{}'",
                        species_id
                    )
                    .into());
                }

                species_data.push(
                    MeasurementDataBuilder::default()
                        .species_id(species_id)
                        .time(time)
                        .initial(initial)
                        .data(data)
                        .build()
                        .expect("Failed to create measurement data"),
                );
            }
        }

        Ok(MeasurementBuilder::default()
            .species_data(species_data)
            .clone())
    }
}

/// Converts a Measurement into a DataFrame.
///
/// # Arguments
///
/// * `measurement` - The Measurement to be converted.
///
/// # Returns
///
/// Returns a DataFrame representing the Measurement.
pub fn measurement_to_dataframe(measurement: &Measurement) -> DataFrame {
    let mut series = vec![];
    let mut non_measured = vec![];
    let mut times = vec![];

    for data in measurement.species_data.iter() {
        collect_data(&mut series, &mut non_measured, &mut times, data);
    }

    if series.is_empty() {
        series.push(Series::new("time", vec![0.0; 1])); // Use floating-point values (f64)
        for (species, initial) in non_measured {
            series.push(Series::new(species, vec![*initial; 1])); // Dereference `initial`
        }
        return DataFrame::new(series).unwrap();
    }

    if !times.is_empty() {
        let first_time = &times[0];
        for time in &times {
            assert_eq!(first_time, time, "Time series do not match");
        }

        series.push(Series::new("time", first_time));
    } else {
        // Seems to only contain initials
        series.push(Series::new("time", vec![0; 1]));
    }

    let length = if let Some(s) = series.first() {
        s.len()
    } else {
        1
    };

    for s in &series {
        assert_eq!(length, s.len(), "Time series do not have the same length");
    }

    for (species, _) in non_measured {
        series.push(Series::new_null(species, length));
    }

    // Sort series with time first and other alphabetical
    series.sort_by(|a, b| {
        if a.name() == "time" {
            std::cmp::Ordering::Less
        } else if b.name() == "time" {
            std::cmp::Ordering::Greater
        } else {
            a.name().cmp(b.name())
        }
    });

    DataFrame::new(series).unwrap()
}

/// Collects data from a `MeasurementData` instance and updates the provided vectors.
///
/// # Arguments
///
/// * `series` - A mutable reference to a vector of `Series` that will be updated with the data.
/// * `non_measured` - A mutable reference to a vector of tuples containing species IDs and initial values for non-measured data.
/// * `times` - A mutable reference to a vector of vectors containing time series data.
/// * `data` - A reference to the `MeasurementData` instance from which data will be collected.
fn collect_data<'a>(
    series: &mut Vec<Series>,
    non_measured: &mut Vec<(&'a String, &'a f64)>, // Lifetimes tied to 'a
    times: &mut Vec<Vec<f64>>,
    data: &'a MeasurementData, // Lifetimes tied to 'a
) {
    if !data.data.is_empty() && !data.time.is_empty() {
        times.push(data.time.clone());
        series.push(Series::new(&data.species_id, data.data.clone()))
    } else {
        non_measured.push((&data.species_id, &data.initial))
    }
}
