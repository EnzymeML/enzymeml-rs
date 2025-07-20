use crate::prelude::Measurement;

/// Checks if a measurement has data for all species
///
/// # Arguments
/// * `measurement` - The measurement to check
///
/// # Returns
/// * `bool` - True if all species have data, false otherwise
pub fn measurement_not_empty(measurement: &Measurement) -> bool {
    measurement.species_data.iter().any(|s| !s.data.is_empty())
}
