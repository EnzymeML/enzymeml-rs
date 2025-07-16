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
        measurement_to_dataframe(&measurement, None)
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
        measurement_to_dataframe(measurement, None)
    }
}

impl Measurement {
    /// Converts the Measurement into a DataFrame.
    ///
    /// # Returns
    ///
    /// Returns a DataFrame representing the Measurement.
    pub fn to_dataframe(&self, include_non_measured: impl Into<Option<bool>>) -> DataFrame {
        measurement_to_dataframe(self, include_non_measured)
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
                    return Err(
                        format!("Missing initial value at t=0 for species '{species_id}'").into(),
                    );
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
pub fn measurement_to_dataframe(
    measurement: &Measurement,
    include_non_measured: impl Into<Option<bool>>,
) -> DataFrame {
    let include_non_measured = include_non_measured.into().unwrap_or(true);
    let mut series = vec![];
    let mut non_measured = vec![];
    let mut times = vec![];

    // Collect all data in one pass
    measurement.species_data.iter().for_each(|data| {
        collect_data(&mut series, &mut non_measured, &mut times, data);
    });

    // Handle case with no time series data
    if series.is_empty() {
        let time_series = Series::new("time", vec![0.0; 1]);
        let mut all_series = vec![time_series];

        // Add initial values for non-measured species
        all_series.extend(
            non_measured
                .into_iter()
                .map(|(species, initial)| Series::new(species, vec![*initial; 1])),
        );

        return DataFrame::new(all_series).unwrap();
    }

    // Add time series
    let time_series = if !times.is_empty() {
        // Verify all time series match
        let first_time = &times[0];
        debug_assert!(
            times.iter().all(|time| time == first_time),
            "Time series do not match"
        );

        Series::new("time", first_time)
    } else {
        // Only contains initials
        Series::new("time", vec![0.0; 1])
    };

    // Get length of data for consistency checks
    let length = series.first().map_or(1, |s| s.len());

    // Verify all series have same length
    debug_assert!(
        series.iter().all(|s| s.len() == length),
        "Time series do not have the same length"
    );

    // Add null series for non-measured species
    let null_series: Vec<Series> = non_measured
        .into_iter()
        .map(|(species, _)| Series::new_null(species, length))
        .collect();

    // Combine all series
    let mut all_series = vec![time_series];
    all_series.extend(series);

    if include_non_measured {
        all_series.extend(null_series);
    }

    DataFrame::new(all_series).unwrap()
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
    } else if let Some(initial) = data.initial.as_ref() {
        non_measured.push((&data.species_id, initial));
    }
}
