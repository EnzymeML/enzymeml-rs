use crate::prelude::{EnzymeMLDocument, MeasurementData};

use super::consistency::{get_species_ids, Report, Severity, ValidationResult};

/// Validates measurements in an EnzymeML document by checking:
/// - Initial concentrations match first data point at t=0
/// - Time and data vectors have consistent lengths
/// - Referenced species exist in the document
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document containing measurements to validate
/// * `report` - Validation report to add any validation errors to
///
/// # Details
/// For each measurement and species data point in the document, performs validation
/// checks and adds any validation errors or warnings to the report.
pub(super) fn check_measurements(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    for (meas_idx, measurement) in enzmldoc.measurements.iter().enumerate() {
        let meas_id = &measurement.id;
        for (data_idx, meas_data) in measurement.species_data.iter().enumerate() {
            check_initial_concentrations(report, meas_id, meas_idx, data_idx, meas_data);
            check_time_data_consistency(report, meas_id, meas_idx, data_idx, meas_data);
            check_species_consistency(report, meas_id, meas_idx, data_idx, meas_data, &all_species);
        }
    }
}

/// Validates that initial concentrations match first data point at t=0
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `meas_id` - ID of the measurement
/// * `meas_idx` - Index of the measurement in the document's measurements list
/// * `data_idx` - Index of the species data in the measurement's data list
/// * `meas_data` - The measurement data to validate
///
/// # Details
/// Checks if the initial concentration matches the first data point when t=0.
/// Adds a warning to the report if they don't match.
/// Skips validation if:
/// - Data or time vectors are empty/missing
/// - First time point is not 0
fn check_initial_concentrations(
    report: &mut Report,
    meas_id: &str,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
) {
    let has_data = !meas_data.data.is_empty();
    let has_time = !meas_data.time.is_empty();

    if !has_data || !has_time {
        // Check if the data and time vectors are not empty
        return;
    } else if !has_data {
        return;
    }

    if meas_data.time.first().unwrap() != &0.0 {
        // If the first time point is not 0, we cannot check
        return;
    }

    // Check if the initial concentration matches the first data point
    if let Some(initial) = meas_data.initial {
        if meas_data.data.first().unwrap() != &initial {
            let result = ValidationResult::new(
                format!("/measurements/{meas_idx}/species/{data_idx}"),
                format!(
                "Initial concentration does not match first data point at t=0 for species '{}'.",
                meas_data.species_id
            ),
                Severity::Warning,
                Some(meas_id.to_string()),
            );

            report.add_result(result);
        }
    }
}

/// Validates that time and data vectors have matching lengths
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `meas_idx` - Index of the measurement in the document's measurements list
/// * `data_idx` - Index of the species data in the measurement's data list
/// * `meas_data` - The measurement data to validate
///
/// # Details
/// Checks if the time and data vectors have the same length.
/// Adds an error to the report if the lengths don't match.
fn check_time_data_consistency(
    report: &mut Report,
    meas_id: &str,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
) {
    if !meas_data.data.is_empty() && !meas_data.time.is_empty() {
        let data = &meas_data.data;
        let time = &meas_data.time;

        if data.len() != time.len() {
            let result = ValidationResult::new(
                format!("/measurements/{meas_idx}/species/{data_idx}"),
                format!(
                    "Data and time vectors have different lengths for species '{}'. \
                Got {} data points and {} time points.",
                    meas_data.species_id,
                    data.len(),
                    time.len()
                ),
                Severity::Error,
                Some(meas_id.to_string()),
            );

            report.add_result(result);
        }
    } else if meas_data.data.is_empty() && !meas_data.time.is_empty() {
        let result = ValidationResult::new(
            format!("/measurements/{meas_idx}/species/{data_idx}"),
            format!(
                "Time vector is missing for species '{}'.",
                meas_data.species_id
            ),
            Severity::Error,
            Some(meas_id.to_string()),
        );

        report.add_result(result);
    } else if !meas_data.data.is_empty() && meas_data.time.is_empty() {
        let result = ValidationResult::new(
            format!("/measurements/{meas_idx}/species/{data_idx}"),
            format!(
                "Data vector is missing for species '{}'.",
                meas_data.species_id
            ),
            Severity::Error,
            Some(meas_id.to_string()),
        );

        report.add_result(result);
    }
}

/// Validates that referenced species exist in the document
///
/// # Arguments
/// * `report` - Validation report to add any errors to
/// * `meas_idx` - Index of the measurement in the document's measurements list
/// * `data_idx` - Index of the species data in the measurement's data list
/// * `meas_data` - The measurement data to validate
/// * `all_species` - List of all species IDs defined in the document
///
/// # Details
/// Checks if the species referenced in the measurement data exists in the document.
/// Adds an error to the report if the species is not found.
fn check_species_consistency(
    report: &mut Report,
    meas_id: &str,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
    all_species: &[String],
) {
    if !all_species.contains(&meas_data.species_id) {
        let result = ValidationResult::new(
            format!("/measurements/{meas_idx}/species/{data_idx}"),
            format!(
                "Species '{}' in measurement is not defined in the document.",
                meas_data.species_id
            ),
            Severity::Error,
            Some(meas_id.to_string()),
        );

        report.add_result(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{
        DataTypes, EnzymeMLDocumentBuilder, MeasurementBuilder, MeasurementDataBuilder,
        SmallMoleculeBuilder, UnitDefinitionBuilder,
    };

    /// Test that a valid measurement is valid
    #[test]
    fn test_valid_measurement() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .species_id("S1".to_string())
                            .initial(1.0)
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .data_type(DataTypes::Concentration)
                            .time(vec![0.0, 1.0])
                            .data(vec![1.0, 2.0])
                            .is_simulated(false)
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);
        assert!(report.is_valid);
    }

    /// Test that a measurement with no species ID is invalid
    #[test]
    fn test_invalid_measurement_no_species_id() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .initial(1.0)
                            .data_type(DataTypes::Concentration)
                            .time(vec![0.0, 1.0])
                            .data(vec![1.0, 2.0])
                            .is_simulated(false)
                            .species_id("S1".to_string())
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);
        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a measurement with inconsistent time and data vectors is invalid
    #[test]
    fn test_invalid_measurement_inconsistent_time_and_data_vectors() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .species_id("S1".to_string())
                            .initial(1.0)
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .data_type(DataTypes::Concentration)
                            .is_simulated(false)
                            .time(vec![0.0, 1.0])
                            .data(vec![1.0, 2.0, 3.0])
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a measurement with missing time vector is invalid
    #[test]
    fn test_invalid_measurement_missing_time_vector() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .species_id("S1".to_string())
                            .initial(1.0)
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .data_type(DataTypes::Concentration)
                            .is_simulated(false)
                            .data(vec![1.0, 2.0, 3.0])
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a measurement with missing data vector is invalid
    #[test]
    fn test_invalid_measurement_missing_data_vector() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S1".to_string())
                    .name("S1".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .species_id("S1".to_string())
                            .initial(1.0)
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .data_type(DataTypes::Concentration)
                            .is_simulated(false)
                            .time(vec![0.0, 1.0])
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }

    /// Test that a measurement with inconsistent species id is invalid
    #[test]
    fn test_invalid_measurement_inconsistent_species_id() {
        let mut report = Report::new();
        let enzmldoc = EnzymeMLDocumentBuilder::default()
            .name("test".to_string())
            .to_small_molecules(
                SmallMoleculeBuilder::default()
                    .id("S2".to_string())
                    .name("S2".to_string())
                    .constant(false)
                    .build()
                    .expect("Failed to build small molecule"),
            )
            .to_measurements(
                MeasurementBuilder::default()
                    .id("M1".to_string())
                    .name("M1".to_string())
                    .to_species_data(
                        MeasurementDataBuilder::default()
                            .species_id("S1".to_string())
                            .initial(1.0)
                            .data_unit(UnitDefinitionBuilder::default().build().unwrap())
                            .data_type(DataTypes::Concentration)
                            .is_simulated(false)
                            .build()
                            .expect("Failed to build measurement data"),
                    )
                    .build()
                    .expect("Failed to build measurement"),
            )
            .build()
            .expect("Failed to build document");

        check_measurements(&enzmldoc, &mut report);

        assert!(!report.is_valid);
        assert_eq!(report.errors.len(), 1);
    }
}
