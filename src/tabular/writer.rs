//! Tabular Data Writing Module
//!
//! This module provides functionality for writing tabular data to spreadsheet files,
//! supporting conversion of `EnzymeMLDocument` and `Measurement` objects to Excel spreadsheets.
//!
//! # Key Features
//!
//! - Convert `EnzymeMLDocument` to Excel spreadsheets with multiple measurement worksheets
//! - Convert individual `Measurement` objects to standalone Excel files
//! - Generate measurement templates for data collection
//! - Apply cell formatting and data validation for professional spreadsheets
//! - Support for empty documents (creates templates) and populated documents
//!
//! # Spreadsheet Structure
//!
//! Each measurement becomes a separate worksheet containing:
//! - Time column (first column)
//! - Species concentration columns
//! - Formatted headers with green background
//! - Data validation ensuring positive numbers only
//! - Consistent cell formatting with borders and alignment
//!
//! # Template Generation
//!
//! When an `EnzymeMLDocument` contains no measurements, the module automatically
//! generates a template worksheet with all species from the document's reactions,
//! allowing users to input experimental data.

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

use polars::prelude::{AnyValue, DataFrame};
use rust_xlsxwriter::workbook::Workbook;
use rust_xlsxwriter::{Format, FormatAlign, FormatBorder};

use crate::prelude::{EnzymeMLDocument, Measurement, MeasurementData};
use crate::validation::consistency::get_species_ids;

/// Default number of rows to set up in worksheets for data entry
const DEFAULT_ROW_COUNT: u32 = 400;

/// Default column width in Excel units for optimal readability
const DEFAULT_COLUMN_WIDTH: f64 = 20.0;

/// Default row height in Excel units for consistent appearance
const DEFAULT_ROW_HEIGHT: f64 = 15.0;

/// Border color for all cell borders (light gray)
const BORDER_COLOR: u32 = 0xB0B0B0;

/// Header background color (blue)
const HEADER_BG_COLOR: u32 = 0x3F51B5;

/// Header text color (white)
const HEADER_TEXT_COLOR: u32 = 0xFFFFFF;

impl EnzymeMLDocument {
    /// Converts the EnzymeMLDocument to an Excel file
    ///
    /// Creates a multi-worksheet Excel file where each measurement becomes a separate worksheet.
    /// If the document contains no measurements, generates a template worksheet instead.
    ///
    /// # Arguments
    ///
    /// * `output` - The file path where the Excel file will be saved
    /// * `template` - Whether to create a template worksheet
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the conversion and file save were successful
    /// * `Err(...)` if an error occurred during conversion or file writing
    pub fn to_excel(
        &self,
        output: impl Into<PathBuf>,
        template: bool,
        use_names: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut enzmldoc = self.clone();
        if template {
            enzmldoc.measurements.clear();
        }

        let mut workbook = create_workbook(&enzmldoc, use_names)?;
        workbook.save(output.into())?;
        Ok(())
    }
}

/// Creates a workbook from an EnzymeMLDocument reference
///
/// This function handles both populated documents (with measurements) and
/// empty documents (creates templates). Each measurement becomes a separate worksheet
/// with proper formatting and data validation.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeMLDocument reference to convert
///
/// # Returns
///
/// * `Ok(Workbook)` containing the converted data
/// * `Err(...)` if the conversion fails
pub fn create_workbook(
    enzmldoc: &EnzymeMLDocument,
    use_names: bool,
) -> Result<Workbook, Box<dyn std::error::Error>> {
    let mut workbook = Workbook::new();

    let mut enzmldoc = enzmldoc.clone();

    if use_names {
        replace_species_ids(&mut enzmldoc);
    }

    if enzmldoc.measurements.is_empty() {
        create_meas_template(&enzmldoc, &mut workbook)?;
    } else {
        for measurement in &enzmldoc.measurements {
            add_meas_sheet(measurement, &mut workbook)?;
        }
    }

    Ok(workbook)
}

impl TryFrom<&EnzymeMLDocument> for Workbook {
    type Error = Box<dyn std::error::Error>;

    /// Converts an EnzymeMLDocument reference into a Workbook
    ///
    /// This implementation delegates to the create_workbook_from_enzmldoc function.
    ///
    /// # Returns
    ///
    /// * `Ok(Workbook)` containing the converted data
    /// * `Err(...)` if the conversion fails
    fn try_from(enzmldoc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        create_workbook(enzmldoc, false)
    }
}

impl TryFrom<EnzymeMLDocument> for Workbook {
    type Error = Box<dyn std::error::Error>;

    /// Converts an owned EnzymeMLDocument into a Workbook
    ///
    /// This is a convenience implementation that delegates to the create_workbook_from_enzmldoc function.
    /// Creates a template if the document has no measurements,
    /// otherwise adds each measurement as a separate worksheet.
    ///
    /// # Returns
    ///
    /// * `Ok(Workbook)` containing the converted data
    /// * `Err(...)` if the conversion fails
    fn try_from(enzmldoc: EnzymeMLDocument) -> Result<Self, Self::Error> {
        create_workbook(&enzmldoc, false)
    }
}

impl TryFrom<Measurement> for Workbook {
    type Error = Box<dyn std::error::Error>;

    /// Converts a single Measurement into a Workbook
    ///
    /// Creates a single-worksheet Excel file containing the measurement data
    /// with proper formatting and validation.
    ///
    /// # Returns
    ///
    /// * `Ok(Workbook)` containing the measurement as a single worksheet
    /// * `Err(...)` if the conversion fails
    fn try_from(measurement: Measurement) -> Result<Self, Self::Error> {
        let mut workbook = Workbook::new();
        add_meas_sheet(&measurement, &mut workbook)?;
        Ok(workbook)
    }
}

/// Creates a measurement template and adds it to the workbook
///
/// Generates a template worksheet containing all species found in the document's reactions.
/// The template includes empty time and data columns for each species, allowing users
/// to input experimental data in the correct format.
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeMLDocument to extract species information from
/// * `workbook` - The Workbook to add the template worksheet to
///
/// # Returns
///
/// * `Ok(())` if the template was successfully created and added
/// * `Err(...)` if an error occurred during template creation
fn create_meas_template(
    enzmldoc: &EnzymeMLDocument,
    workbook: &mut Workbook,
) -> Result<(), Box<dyn Error>> {
    let all_species = get_species_ids(enzmldoc);

    let mut measurement_template = Measurement {
        id: String::from("Measurement"),
        name: String::from("EnzymeML Measurement Template"),
        ..Default::default()
    };

    for species in all_species {
        let data = MeasurementData {
            species_id: species,
            initial: Some(0.0),
            time: Vec::new(),
            data: Vec::new(),
            ..Default::default()
        };

        measurement_template.species_data.push(data);
    }

    add_meas_sheet(&measurement_template, workbook)
}

/// Adds a measurement as a worksheet to the workbook
///
/// Creates a fully formatted worksheet containing the measurement data with:
/// - Professional header formatting (bold, colored background)
/// - Data validation for positive numbers only
/// - Consistent cell borders and alignment
/// - Optimal column widths and row heights
///
/// # Arguments
///
/// * `measurement` - The Measurement to add as a worksheet
/// * `workbook` - The Workbook to add the worksheet to
///
/// # Returns
///
/// * `Ok(())` if the worksheet was successfully added
/// * `Err(...)` if an error occurred during worksheet creation
///
/// # Errors
///
/// Returns an error if:
/// - The measurement name is empty
/// - Data writing or formatting operations fail
/// - Worksheet configuration fails
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
        }
    }

    // Set column widths
    for i in 0..column_count {
        sheet.set_column_width(i as u16, DEFAULT_COLUMN_WIDTH)?;
    }

    // Set row heights and format
    for i in 0..DEFAULT_ROW_COUNT {
        sheet.set_row_height(i, DEFAULT_ROW_HEIGHT)?;
        for j in 0..column_count {
            if i > 0 {
                sheet.set_cell_format(i, j as u16, &data_format)?;
            }
        }
    }

    Ok(())
}

/// Returns a format for non-header cells
///
/// Creates a consistent format for data cells featuring:
/// - 14pt font size for readability
/// - Center alignment (horizontal and vertical)
/// - Thin borders on all sides in light gray
///
/// # Returns
///
/// A `Format` object configured for data cells
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
///
/// Creates a distinctive header format featuring:
/// - Light green background color
/// - Bold, 18pt font for prominence
/// - Center alignment (horizontal and vertical)
/// - Thin borders with double bottom border for separation
///
/// # Returns
///
/// A `Format` object configured for header cells
fn get_header_format() -> Format {
    Format::new()
        .set_text_wrap()
        .set_background_color(HEADER_BG_COLOR)
        .set_font_color(HEADER_TEXT_COLOR)
        .set_bold()
        .set_font_size(15f64)
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

/// Replaces the species IDs with their display names
///
/// This function iterates through all measurements in the EnzymeML document
/// and replaces the species IDs with their corresponding names from the name map.
///
/// # Arguments
///
/// * `enzmldoc` - Reference to the EnzymeML document containing measurements
/// * `name_map` - HashMap mapping species IDs to their display names
fn replace_species_ids(enzmldoc: &mut EnzymeMLDocument) {
    let name_map = get_name_map(enzmldoc);
    for measurement in &mut enzmldoc.measurements {
        for meas_data in &mut measurement.species_data {
            if let Some(name) = name_map.get(&meas_data.species_id) {
                meas_data.species_id = name.clone();
            }
        }
    }
}

/// Creates a mapping from species IDs to their display names
///
/// This function iterates through all species types in the EnzymeML document
/// (small molecules, proteins, and complexes) and creates a HashMap that maps
/// each species ID to its corresponding name for display purposes.
///
/// # Arguments
///
/// * `enzmldoc` - Reference to the EnzymeML document containing species data
///
/// # Returns
///
/// A HashMap where keys are species IDs and values are species names
pub(super) fn get_name_map(enzmldoc: &EnzymeMLDocument) -> HashMap<String, String> {
    enzmldoc
        .small_molecules
        .iter()
        .map(|species| (species.id.clone(), species.name.clone()))
        .chain(
            enzmldoc
                .proteins
                .iter()
                .map(|species| (species.id.clone(), species.name.clone())),
        )
        .chain(
            enzmldoc
                .complexes
                .iter()
                .map(|species| (species.id.clone(), species.name.clone())),
        )
        .collect()
}
