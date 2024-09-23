use std::error::Error;
use std::path::{Path, PathBuf};

use polars::prelude::{AnyValue, DataFrame};
use xlsxwriter::prelude::*;
use xlsxwriter::worksheet::validation::{
    DataValidation, DataValidationErrorType, DataValidationNumberOptions, DataValidationType,
    ErrorAlertOptions,
};

use crate::prelude::{EnzymeMLDocument, Measurement, MeasurementDataBuilder};
use crate::validation::validator::get_species_ids;

static ERROR_MESSAGE: &str = "Only positive numbers are allowed in this cell.";

impl EnzymeMLDocument {
    /// Converts the EnzymeMLDocument to a spreadsheet and saves it to the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the spreadsheet will be saved.
    pub fn to_excel(
        &self,
        path: PathBuf,
        create_template: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        to_excel(
            ConversionOptions::EnzymeMLDocument(self.clone()),
            path,
            create_template,
        )
    }
}

impl Measurement {
    /// Converts the Measurement to a spreadsheet and saves it to the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where the spreadsheet will be saved.
    pub fn to_excel(&self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        to_excel(ConversionOptions::Measurement(self.clone()), path, false)
    }
}

/// Represents the conversion options for the spreadsheet.
///
/// This enum allows for the conversion of either an EnzymeMLDocument or a Measurement.
/// Hence, it provides a flexible way to handle different types of conversions based
/// on the type provided.
pub enum ConversionOptions {
    EnzymeMLDocument(EnzymeMLDocument),
    Measurement(Measurement),
}

impl From<EnzymeMLDocument> for ConversionOptions {
    fn from(enzmldoc: EnzymeMLDocument) -> Self {
        ConversionOptions::EnzymeMLDocument(enzmldoc)
    }
}

impl From<Measurement> for ConversionOptions {
    fn from(measurement: Measurement) -> Self {
        ConversionOptions::Measurement(measurement)
    }
}

/// Converts the given EnzymeMLDocument or Measurement to a spreadsheet and saves it to the specified path.
///
/// # Arguments
///
/// * `options` - The conversion options, which can be either an EnzymeMLDocument or a Measurement.
/// * `path` - The path where the spreadsheet will be saved.
pub fn to_excel(
    options: ConversionOptions,
    path: PathBuf,
    create_template: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match options {
        ConversionOptions::EnzymeMLDocument(enzmldoc) => {
            doc_to_spreadsheet(&enzmldoc, path, create_template)
        }
        ConversionOptions::Measurement(measurement) => meas_to_spreadsheet(&measurement, path),
    }
}

/// Converts the given EnzymeMLDocument to a spreadsheet and saves it to the specified path.
///
/// # Arguments
///
/// * `enzmldoc` - A reference to the EnzymeMLDocument to be converted.
/// * `path` - The path where the spreadsheet will be saved.
fn doc_to_spreadsheet(
    enzmldoc: &EnzymeMLDocument,
    path: PathBuf,
    create_template: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    directory_exists(&path)?;

    let path = path.to_str().ok_or("Invalid path")?;
    let workbook = Workbook::new(path)?;

    if enzmldoc.measurements.is_empty() || create_template {
        // Create a template if there are no measurements
        create_meas_template(enzmldoc, &workbook)?;
    } else {
        // Add a worksheet for each measurement in the document
        for measurement in &enzmldoc.measurements {
            add_meas_sheet(measurement, &workbook)?;
        }
    }

    // Close the workbook
    workbook.close()?;

    Ok(())
}

/// Creates a measurement template and adds it to the workbook.
///
/// # Arguments
///
/// * `enzmldoc` - A reference to the EnzymeMLDocument.
/// * `workbook` - A reference to the Workbook where the template will be added.
fn create_meas_template(
    enzmldoc: &EnzymeMLDocument,
    workbook: &Workbook,
) -> Result<(), Box<dyn Error>> {
    // Take all the species that exists in the document and create a template
    let all_species = get_species_ids(enzmldoc);

    // Create a measurement that has all species with empty data
    let mut measurement_template = Measurement::default();
    measurement_template.name = String::from("EnzymeML Measurement Template");

    for species in all_species {
        measurement_template.species_data.push(
            MeasurementDataBuilder::default()
                .species_id(species)
                .initial(0.0)
                .time(vec![])
                .data(vec![])
                .build()
                .expect("Failed to create measurement data"),
        );
    }

    // Now add this template to the workbook
    add_meas_sheet(&measurement_template, workbook)
}

/// Converts the given Measurement to a spreadsheet and saves it to the specified path.
///
/// # Arguments
///
/// * `measurement` - A reference to the Measurement to be converted.
/// * `path` - The path where the spreadsheet will be saved.
fn meas_to_spreadsheet(
    measurement: &Measurement,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    directory_exists(&path)?;

    let path = path.to_str().ok_or("Invalid path")?;
    let workbook = Workbook::new(path)?;

    // Add a worksheet for the measurement
    add_meas_sheet(measurement, &workbook)?;

    // Close the workbook
    workbook.close()?;

    Ok(())
}

/// Adds a measurement sheet to the workbook.
///
/// # Arguments
///
/// * `measurement` - A reference to the Measurement to be added.
/// * `workbook` - A reference to the Workbook where the sheet will be added.
fn add_meas_sheet(measurement: &Measurement, workbook: &Workbook) -> Result<(), Box<dyn Error>> {
    // If the length of the name is 0, return an error
    if measurement.name.is_empty() {
        return Err("Measurement name cannot be empty".into());
    }

    let mut sheet = workbook.add_worksheet(Some(&measurement.name))?;

    // First, convert the measurement to a dataframe
    let df: DataFrame = measurement.clone().into();

    // Use the dataframe to write the data
    for (i, column) in df.get_columns().iter().enumerate() {
        let header_name = if column.name() == "time" {
            "Time"
        } else {
            column.name()
        };

        sheet.write_string(0, i as u16, header_name, Some(&get_header_format()))?;
    }

    // Write the data
    for (col_index, row) in df.iter().enumerate() {
        for (row_index, value) in row.iter().enumerate() {
            let value = match value {
                AnyValue::Null => String::from(""),
                _ => value.to_string(),
            };

            sheet.write_string(row_index as u32 + 1, col_index as u16, &value, None)?;
        }
    }

    // Set the column widths
    for i in 0..df.get_columns().len() {
        sheet.set_column(i as u16, i as u16, 20.0, Some(&get_non_header_format()))?;
    }

    // Set all rows to 15.0 height
    for i in 0..99 {
        sheet.set_row(i as u32, 40.0, None)?;
    }

    // Add validation to all rows in all columns to only accept numbers
    sheet.data_validation_range(
        1,
        0,
        99,
        df.width() as u16 - 1,
        &DataValidation::new(
            DataValidationType::Decimal {
                ignore_blank: true,
                number_options: DataValidationNumberOptions::GreaterThanOrEqualTo(0.0),
            },
            None,
            Some(ErrorAlertOptions {
                style: DataValidationErrorType::Stop,
                title: "Invalid input".to_string(),
                message: ERROR_MESSAGE.to_string(),
            }),
        ),
    )?;

    Ok(())
}

/// Checks if the directory of the given path exists.
///
/// # Arguments
///
/// * `path` - A reference to the PathBuf to be checked.
///
/// # Returns
///
/// Returns true if the directory exists, false otherwise.
fn directory_exists(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(dir_path) = path.parent() {
        // Check if the directory exists
        if !dir_path.exists() {
            return Err(format!("Directory does not exist: {:?}", dir_path).into());
        }
    }

    Ok(())
}

fn get_non_header_format() -> Format {
    Format::new()
        .set_font_size(14f64)
        .set_vertical_align(FormatVerticalAlignment::VerticalCenter)
        .set_align(FormatAlignment::Center)
        .set_border_left(FormatBorder::Thin)
        .set_border_left_color(FormatColor::Custom(0xB0B0B0))
        .set_border_right(FormatBorder::Thin)
        .set_border_right_color(FormatColor::Custom(0xB0B0B0))
        .set_border_top(FormatBorder::Thin)
        .set_border_top_color(FormatColor::Custom(0xB0B0B0))
        .set_border_bottom(FormatBorder::Thin)
        .set_border_bottom_color(FormatColor::Custom(0xB0B0B0))
        .clone()
}

fn get_header_format() -> Format {
    Format::new()
        .set_bg_color(FormatColor::Custom(0xD9EAD3))
        .set_bold()
        .set_font_size(18f64)
        .set_border_left(FormatBorder::Thin)
        .set_border_left_color(FormatColor::Gray)
        .set_border_right(FormatBorder::Thin)
        .set_border_right_color(FormatColor::Gray)
        .set_border_top(FormatBorder::Thin)
        .set_border_top_color(FormatColor::Gray)
        .set_border_bottom(FormatBorder::Double)
        .set_border_bottom_color(FormatColor::Gray)
        .set_vertical_align(FormatVerticalAlignment::VerticalCenter)
        .set_align(FormatAlignment::Center)
        .clone()
}
