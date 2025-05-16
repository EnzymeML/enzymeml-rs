//! Tabular Data Writing Module
//!
//! This module provides functionality for writing tabular data to spreadsheet files,
//! supporting conversion of `EnzymeMLDocument` and `Measurement` objects to spreadsheets.
//!
//! # Key Features
//!
//! - Convert `EnzymeMLDocument` to Excel spreadsheets
//! - Convert individual `Measurement` objects to Excel files
//! - Support for creating templates and full data export
//!
//! # Usage
//!
//! The module enables:
//! - Exporting complete enzyme kinetics documents
//! - Generating measurement templates
//! - Flexible spreadsheet generation with validation options
//!
//! # Methods
//!
//! - `to_excel()` on `EnzymeMLDocument` and `Measurement`
//! - Conversion of data with optional template creation

use std::error::Error;

use polars::prelude::{AnyValue, DataFrame};
use rust_xlsxwriter::workbook::Workbook;
use rust_xlsxwriter::{
    DataValidation, DataValidationErrorStyle, DataValidationRule, Format, FormatAlign, FormatBorder,
};

use crate::prelude::{EnzymeMLDocument, Measurement, MeasurementBuilder, MeasurementDataBuilder};
use crate::validation::consistency::get_species_ids;

/// Error message for data validation
const ERROR_MESSAGE: &str = "Only positive numbers are allowed in this cell.";

/// Default number of rows to set up in worksheet
const DEFAULT_ROW_COUNT: u32 = 99;

/// Default column width
const DEFAULT_COLUMN_WIDTH: f64 = 20.0;

/// Default row height
const DEFAULT_ROW_HEIGHT: f64 = 15.0;

/// Border color for cells
const BORDER_COLOR: u32 = 0xB0B0B0;

/// Header background color
const HEADER_BG_COLOR: u32 = 0xD9EAD3;

impl TryFrom<EnzymeMLDocument> for Workbook {
    type Error = Box<dyn std::error::Error>;

    /// Converts an EnzymeMLDocument into a Workbook
    ///
    /// Creates a template if the document has no measurements,
    /// otherwise adds each measurement as a separate worksheet.
    fn try_from(enzmldoc: EnzymeMLDocument) -> Result<Self, Self::Error> {
        let mut workbook = Workbook::new();
        if enzmldoc.measurements.is_empty() {
            create_meas_template(&enzmldoc, &mut workbook)?;
        } else {
            for measurement in &enzmldoc.measurements {
                add_meas_sheet(measurement, &mut workbook)?;
            }
        }

        Ok(workbook)
    }
}

impl TryFrom<Measurement> for Workbook {
    type Error = Box<dyn std::error::Error>;

    /// Converts a single Measurement into a Workbook
    fn try_from(measurement: Measurement) -> Result<Self, Self::Error> {
        let mut workbook = Workbook::new();
        add_meas_sheet(&measurement, &mut workbook)?;
        Ok(workbook)
    }
}

/// Creates a measurement template and adds it to the workbook
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeMLDocument to use as template basis
/// * `workbook` - The Workbook to add the template sheet to
fn create_meas_template(
    enzmldoc: &EnzymeMLDocument,
    workbook: &mut Workbook,
) -> Result<(), Box<dyn Error>> {
    let all_species = get_species_ids(enzmldoc);

    let mut measurement_template = MeasurementBuilder::default()
        .name(String::from("EnzymeML Measurement Template"))
        .build()?;

    for species in all_species {
        let data = MeasurementDataBuilder::default()
            .species_id(species)
            .initial(0.0)
            .time(Vec::new())
            .data(Vec::new())
            .build()
            .map_err(|e| format!("Failed to create measurement data: {}", e))?;

        measurement_template.species_data.push(data);
    }

    add_meas_sheet(&measurement_template, workbook)
}

/// Adds a measurement as a worksheet to the workbook
///
/// # Arguments
///
/// * `measurement` - The Measurement to add as a worksheet
/// * `workbook` - The Workbook to add the worksheet to
///
/// # Returns
///
/// * `Ok(())` if the worksheet was successfully added
/// * `Err(...)` if an error occurred during the process
fn add_meas_sheet(
    measurement: &Measurement,
    workbook: &mut Workbook,
) -> Result<(), Box<dyn Error>> {
    if measurement.name.is_empty() {
        return Err("Measurement name cannot be empty".into());
    }

    let sheet = workbook.add_worksheet();
    sheet.set_name(&measurement.name)?;

    // Convert the measurement to a dataframe
    let df: DataFrame = measurement.clone().into();
    let column_count = df.width();

    // Write headers
    let header_format = get_header_format();
    for (i, column) in df.get_columns().iter().enumerate() {
        let col_idx = i as u16;
        let header_name = if column.name() == "time" {
            "Time"
        } else {
            column.name()
        };

        sheet.write_string(0, col_idx, header_name)?;
        sheet.set_cell_format(0, col_idx, &header_format)?;
    }

    // Write data rows
    let data_format = get_non_header_format();
    for (col_idx, row) in df.iter().enumerate() {
        for (row_idx, value) in row.iter().enumerate() {
            let value_str = match value {
                AnyValue::Null => String::new(),
                _ => value.to_string(),
            };

            let row_pos = row_idx as u32 + 1;
            let col_pos = col_idx as u16;

            sheet.write_string(row_pos, col_pos, &value_str)?;
            sheet.set_cell_format(row_pos, col_pos, &data_format)?;
        }
    }

    // Set column widths
    for i in 0..column_count {
        sheet.set_column_width(i as u16, DEFAULT_COLUMN_WIDTH)?;
    }

    // Set row heights
    for i in 0..DEFAULT_ROW_COUNT {
        sheet.set_row_height(i, DEFAULT_ROW_HEIGHT)?;
    }

    // Add data validation
    add_data_validation(sheet, column_count as u16)?;

    Ok(())
}

/// Adds positive number validation to the data cells in the worksheet
///
/// # Arguments
///
/// * `sheet` - The worksheet to add validation to
/// * `column_count` - The number of columns in the worksheet
fn add_data_validation(
    sheet: &mut rust_xlsxwriter::Worksheet,
    column_count: u16,
) -> Result<(), Box<dyn Error>> {
    let validation = DataValidation::new()
        .allow_decimal_number(DataValidationRule::GreaterThanOrEqualTo(0.0))
        .ignore_blank(true)
        .set_error_style(DataValidationErrorStyle::Stop)
        .set_error_title("Invalid input")?
        .set_error_message(ERROR_MESSAGE)?;

    sheet.add_data_validation(1, 0, DEFAULT_ROW_COUNT, column_count - 1, &validation)?;
    Ok(())
}

/// Returns a format for non-header cells
fn get_non_header_format() -> Format {
    Format::new()
        .set_font_size(14f64)
        .set_align(FormatAlign::VerticalCenter)
        .set_align(FormatAlign::Center)
        .set_border_left(FormatBorder::Thin)
        .set_border_left_color(BORDER_COLOR)
        .set_border_right(FormatBorder::Thin)
        .set_border_right_color(BORDER_COLOR)
        .set_border_top(FormatBorder::Thin)
        .set_border_top_color(BORDER_COLOR)
        .set_border_bottom(FormatBorder::Thin)
        .set_border_bottom_color(BORDER_COLOR)
}

/// Returns a format for header cells
fn get_header_format() -> Format {
    Format::new()
        .set_background_color(HEADER_BG_COLOR)
        .set_bold()
        .set_font_size(18f64)
        .set_border_left(FormatBorder::Thin)
        .set_border_left_color(BORDER_COLOR)
        .set_border_right(FormatBorder::Thin)
        .set_border_right_color(BORDER_COLOR)
        .set_border_top(FormatBorder::Thin)
        .set_border_top_color(BORDER_COLOR)
        .set_border_bottom(FormatBorder::Double)
        .set_border_bottom_color(BORDER_COLOR)
        .set_align(FormatAlign::VerticalCenter)
        .set_align(FormatAlign::Center)
}
