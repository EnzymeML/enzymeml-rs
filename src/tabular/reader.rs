use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use calamine::{open_workbook_auto, RangeDeserializerBuilder, Reader, Sheets};
use polars::frame::DataFrame;
use polars::prelude::NamedFrom;
use polars::series::Series;

use crate::enzyme_ml::{Measurement, MeasurementBuilder};
use crate::prelude::{EnzymeMLDocument, EnzymeMLDocumentBuilder};

impl EnzymeMLDocument {
    /// Adds measurements from an Excel file to the `EnzymeMLDocument`.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the Excel file.
    /// * `overwrite` - A boolean indicating whether to overwrite existing measurements.
    ///
    /// # Returns
    ///
    /// Returns a `Result` indicating success or failure.
    pub fn add_from_excel(&mut self, path: PathBuf, overwrite: bool) -> Result<(), Box<dyn Error>> {
        let dfs = read_excel(path)?;

        for (name, df) in dfs {
            let meas = self.create_new_measurement(&name, &df)?;

            if !overwrite {
                self.measurements.push(meas);
                continue;
            }

            // Check if either name or id already exists
            if let Some(same_meas) = self.find_measurement(name) {
                same_meas.species_data = meas.species_data;
                continue;
            } else {
                self.measurements.push(meas);
            }

            let mut meas = MeasurementBuilder::from_dataframe(&df)?;
            meas.id(self.generate_id());
            meas.name(self.generate_name());
            self.measurements.push(meas.build()?);
        }

        Ok(())
    }

    /// Finds a measurement by name or id.
    ///
    /// # Arguments
    ///
    /// * `name` - The name or id of the measurement to find.
    ///
    /// # Returns
    ///
    /// Returns an `Option` containing a mutable reference to the `Measurement` if found.
    fn find_measurement(&mut self, name: String) -> Option<&mut Measurement> {
        self.measurements
            .iter_mut()
            .find(|m| m.id == name || m.name == name)
    }

    /// Creates a new measurement from a `DataFrame`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the measurement.
    /// * `df` - The `DataFrame` containing the measurement data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Measurement` or an error if creation fails.
    fn create_new_measurement(
        &mut self,
        name: &String,
        df: &DataFrame,
    ) -> Result<Measurement, Box<dyn Error>> {
        let mut meas = MeasurementBuilder::from_dataframe(df)?;
        meas.id(name.clone());
        meas.name(name);

        Ok(meas.build()?)
    }

    /// Generates a unique id for a new measurement.
    ///
    /// # Returns
    ///
    /// Returns a `String` containing the unique id.
    fn generate_id(&self) -> String {
        let mut index = 1;
        let prefix = "m";
        while self
            .measurements
            .iter()
            .any(|m| m.id == format!("{}{}", prefix, index))
        {
            index += 1;
        }
        format!("{}{}", prefix, index)
    }

    /// Generates a unique name for a new measurement.
    ///
    /// # Returns
    ///
    /// Returns a `String` containing the unique name.
    fn generate_name(&self) -> String {
        let mut index = 1;
        let prefix = "Measurement";
        while self
            .measurements
            .iter()
            .any(|m| m.name == format!("{} {}", prefix, index))
        {
            index += 1;
        }
        format!("{} {}", prefix, index)
    }
}

impl EnzymeMLDocumentBuilder {
    /// Creates an `EnzymeMLDocumentBuilder` from an Excel file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the Excel file.
    ///
    /// # Returns
    ///
    /// Returns a `Result` indicating success or failure.
    pub fn from_excel(&mut self, path: PathBuf) -> Result<(), Box<dyn Error>> {
        let dfs = read_excel(path)?;

        for (i, (name, df)) in dfs.iter().enumerate() {
            let mut meas = MeasurementBuilder::from_dataframe(df)?;
            meas.id(format!("m{}", i));
            meas.name(name);
            self.to_measurements(meas.build()?);
        }

        Ok(())
    }
}

/// Reads an Excel file and converts it to a `HashMap` of `DataFrame`s.
///
/// # Arguments
///
/// * `path` - The path to the Excel file.
///
/// # Returns
///
/// Returns a `Result` containing a `HashMap` of `DataFrame`s or an error if reading fails.
pub fn read_excel(path: PathBuf) -> Result<HashMap<String, DataFrame>, Box<dyn Error>> {
    let mut workbook = open_workbook_auto(path).expect("Failed to open workbook");
    let mut dfs: HashMap<String, DataFrame> = HashMap::new();

    // Iterate through the sheets in the workbook
    for sheet in workbook.sheet_names() {
        let species_data = process_sheet(&mut workbook, &sheet)?;

        // Convert the HashMap to a DataFrame
        dfs.insert(
            sheet,
            DataFrame::new(species_data.into_values().collect())?,
        );
    }

    Ok(dfs)
}

/// Processes a sheet in the Excel workbook and converts it to a `HashMap` of `Series`.
///
/// # Arguments
///
/// * `workbook` - A mutable reference to the Excel workbook.
/// * `sheet` - The name of the sheet to process.
///
/// # Returns
///
/// Returns a `Result` containing a `HashMap` of `Series` or an error if processing fails.
fn process_sheet(
    workbook: &mut Sheets<BufReader<File>>,
    sheet: &String,
) -> Result<HashMap<String, Series>, Box<dyn Error>> {
    let mut species_data: HashMap<String, Series> = HashMap::new();
    let range = workbook
        .worksheet_range(sheet)
        .expect("Failed to read sheet");

    let header_mapping: HashMap<usize, String> = range
        .rows()
        .next()
        .unwrap_or_else(|| panic!("No headers found in sheet: {}", sheet))
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let name = if *cell == "Time" {
                "time".to_string()
            } else {
                cell.to_string()
            };

            (i, name)
        })
        .collect();

    let iter = RangeDeserializerBuilder::new()
        .has_headers(true)
        .from_range(&range)?;

    for result in iter {
        let row: Vec<String> = result?;

        for (i, value) in row.iter().enumerate() {
            let species_name = &header_mapping[&i];
            let measurement_value: f32 = value.parse().unwrap_or_else(|_| -1.0);

            species_data
                .entry(species_name.clone())
                .or_insert_with(|| Series::new(species_name, Vec::<f32>::new()))
                .append(&Series::new(species_name, vec![measurement_value]))?;
        }
    }
    Ok(species_data)
}
