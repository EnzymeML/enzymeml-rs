use crate::enzyme_ml::{EnzymeMLDocument, MeasurementData};

use super::validator::{get_species_ids, Report, Severity, ValidationResult};

/// The `check_measurements` function is used to check the measurements of an `EnzymeMLDocument`.
/// It modifies the provided `Report` with the results of the checks.
pub(super) fn check_measurements(enzmldoc: &EnzymeMLDocument, report: &mut Report) {
    let all_species = get_species_ids(enzmldoc);

    for (meas_idx, measurement) in enzmldoc.measurements.iter().enumerate() {
        for (data_idx, meas_data) in measurement.species_data.iter().enumerate() {
            check_initial_concentrations(report, meas_idx, data_idx, meas_data);
            check_time_data_consistency(report, meas_idx, data_idx, meas_data);
            check_species_consistency(report, meas_idx, data_idx, meas_data, &all_species);
        }
    }
}

/// The `check_measurement_data` function is used to check a single measurement data
/// of an `EnzymeMLDocument`. It modifies the provided `Report` with the results of the check.
fn check_initial_concentrations(
    report: &mut Report,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
) {
    let has_data = meas_data.data.as_ref().is_some();
    let has_time = meas_data.time.as_ref().is_some();

    if !has_data || !has_time {
        // Check if the data and time vectors are not empty
        return;
    }

    if meas_data.time.as_ref().unwrap().first().unwrap() != &0.0 {
        // If the first time point is not 0, we cannot check
        return;
    }

    // Check if the initial concentration matches the first data point
    if meas_data.data.as_ref().unwrap().first().unwrap() != &meas_data.initial {
        let result = ValidationResult::new(
            format!("/measurements/{}/species/{}", meas_idx, data_idx),
            format!(
                "Initial concentration does not match first data point at t=0 for species '{}'.",
                meas_data.species_id
            ),
            Severity::Warning,
        );

        report.add_result(result);
    }
}

fn check_time_data_consistency(
    report: &mut Report,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
) {
    if meas_data.data.is_none() || meas_data.time.is_none() {
        // Check if the data and time vectors are not empty
    } else {
        let data = meas_data.data.as_ref().unwrap();
        let time = meas_data.time.as_ref().unwrap();

        if data.len() != time.len() {
            let result = ValidationResult::new(
                format!("/measurements/{}/species/{}", meas_idx, data_idx),
                format!(
                    "Data and time vectors have different lengths for species '{}'. \
                Got {} data points and {} time points.",
                    meas_data.species_id,
                    meas_data.data.clone().unwrap().len(),
                    meas_data.time.clone().unwrap().len()
                ),
                Severity::Error,
            );

            report.add_result(result);
        }
    }
}

fn check_species_consistency(
    report: &mut Report,
    meas_idx: usize,
    data_idx: usize,
    meas_data: &MeasurementData,
    all_species: &[&String],
) {
    if !all_species.contains(&&meas_data.species_id) {
        let result = ValidationResult::new(
            format!("/measurements/{}/species/{}", meas_idx, data_idx),
            format!(
                "Species '{}' in measurement is not defined in the document.",
                meas_data.species_id
            ),
            Severity::Error,
        );

        report.add_result(result);
    }
}
