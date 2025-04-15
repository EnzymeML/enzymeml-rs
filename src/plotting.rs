use itertools::Itertools;
use plotly::{
    common::{Line, Marker, Mode},
    layout::Axis,
    Layout, Plot, Scatter,
};
use thiserror::Error;

use crate::{
    optim::measurement_not_empty,
    prelude::{EnzymeMLDocument, Measurement},
};

#[cfg(feature = "simulation")]
use crate::{
    prelude::{ODESystem, PlotTraces, SimulationResult, SimulationSetup},
    simulation::{error::SimulationError, init_cond::InitialCondition},
};

#[cfg(feature = "simulation")]
use peroxide::fuga::RK5;

const COLORS: &[&str] = &[
    "green", "blue", "red", "purple", "orange", "yellow", "brown", "pink", "gray", "cyan",
];

impl EnzymeMLDocument {
    /// Creates a plot visualization of an EnzymeML document's measurement data.
    ///
    /// # Arguments
    ///
    /// * `columns` - Optional number of columns for the plot grid layout. Defaults to 2 if not specified.
    /// * `show` - Whether to display the plot immediately. Defaults to false.
    /// * `measurement_ids` - Optional vector of measurement IDs to include in the plot. If not provided, all measurements are plotted.
    ///
    /// # Returns
    ///
    /// A Plot object containing the measurement data visualization in a grid layout.
    pub fn plot_measurement(
        &self,
        measurement_id: &str,
        show: bool,
        show_fit: bool,
        width: Option<usize>,
        height: Option<usize>,
    ) -> Result<Plot, PlotError> {
        // Filter measurements if measurement_ids are provided
        let measurement: &Measurement = self
            .measurements
            .iter()
            .find(|meas| meas.id == measurement_id)
            .ok_or(PlotError::MeasurementNotFound(measurement_id.to_string()))?;

        if !measurement_not_empty(measurement) {
            return Err(PlotError::MeasurementEmpty(measurement_id.to_string()));
        }

        // Create a plot
        let mut plot = Plot::new();

        let traces: Vec<Box<Scatter<f64, f64>>> = measurement.into();
        for (i, trace) in traces.into_iter().enumerate() {
            plot.add_trace(
                trace
                    .clone()
                    .marker(Marker::new().color(COLORS[i % COLORS.len()]).size(10)),
            );
        }

        #[cfg(feature = "simulation")]
        if show_fit {
            let traces =
                get_simulation_traces(self, measurement).map_err(|_| PlotError::SimulationError)?;
            for (i, trace) in traces.into_iter().enumerate() {
                plot.add_trace(
                    trace
                        .clone()
                        .line(Line::new().color(COLORS[i % COLORS.len()]).width(2.0)),
                );
            }
        }

        // Create a layout
        let layout = Layout::new()
            .title(self.name.clone())
            .show_legend(true) // Always show the legend
            .width(width.unwrap_or(800))
            .height(height.unwrap_or(600))
            .x_axis(Axis::new().range(vec![0.0]).auto_range(true).title("Time"))
            .y_axis(
                Axis::new()
                    .range(vec![0.0])
                    .auto_range(true)
                    .title("Concentration"),
            );

        plot.set_layout(layout);

        if show {
            plot.show();
        }

        Ok(plot)
    }
}

/// Implements conversion from Measurement to a vector of Scatter plots.
impl From<&Measurement> for Vec<Box<Scatter<f64, f64>>> {
    /// Converts a Measurement into a vector of Scatter plot traces.
    ///
    /// Creates a line plot for each species in the measurement data that has both time and
    /// concentration data available.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The Measurement to convert into plot traces
    ///
    /// # Returns
    ///
    /// A vector of Scatter plot traces, one for each species with valid data.
    fn from(measurement: &Measurement) -> Self {
        let mut plots = Vec::new();

        for meas_data in measurement
            .species_data
            .iter()
            .sorted_by_key(|d| &d.species_id)
        {
            let plot = Scatter::new(meas_data.time.clone(), meas_data.data.clone())
                .name(&meas_data.species_id)
                .mode(Mode::Markers)
                .x_axis("Time")
                .y_axis("Concentration");
            plots.push(plot);
        }

        plots
    }
}

/// Generates plot traces from simulating a measurement's conditions
///
/// This function takes an EnzymeML document and measurement, simulates the system under the
/// measurement's conditions, and converts the results into plot traces.
///
/// # Arguments
///
/// * `doc` - The EnzymeML document containing the model definition
/// * `meas` - The measurement whose conditions should be simulated
///
/// # Returns
///
/// * `Result<PlotTraces, SimulationError>` - Plot traces from the simulation results if successful,
///   or a SimulationError if the simulation fails
///
/// # Errors
///
/// Returns a SimulationError if:
/// * The simulation fails to run
/// * The measurement conditions cannot be converted to simulation inputs
///
#[cfg(feature = "simulation")]
fn get_simulation_traces(
    doc: &EnzymeMLDocument,
    meas: &Measurement,
) -> Result<PlotTraces, SimulationError> {
    // Check if model has valid equations and parameters
    has_valid_model(doc)?;

    let setup: SimulationSetup = meas.try_into().unwrap();
    let initial_conditions: InitialCondition = meas.into();
    let solver = RK5;
    let system: ODESystem = doc.try_into().unwrap();

    let result = system.integrate::<SimulationResult>(
        &setup,
        initial_conditions,
        None,
        None,
        solver,
        None,
    )?;

    let traces: Vec<Box<Scatter<f64, f64>>> = (&result).into();
    Ok(traces)
}

/// Checks if a model has valid equations and parameters
///
/// # Arguments
///
/// * `doc` - The EnzymeML document to check
///
/// # Returns
///
/// * `bool` - True if the model has valid equations and parameters, false otherwise   
fn has_valid_model(doc: &EnzymeMLDocument) -> Result<(), PlotError> {
    // Check if model has equations, parameters, and all parameters have values
    if doc.equations.is_empty() || doc.parameters.is_empty() {
        return Err(PlotError::MissingModel);
    }

    let mut missing_values = vec![];
    for param in doc.parameters.iter() {
        if param.value.is_none() {
            missing_values.push(param.id.clone());
        }
    }
    if !missing_values.is_empty() {
        return Err(PlotError::MissingParameterValues(missing_values));
    }

    Ok(())
}

#[derive(Error, Debug)]
pub enum PlotError {
    #[error("Measurement with id {0} not found")]
    MeasurementNotFound(String),
    #[error("Simulation failed")]
    SimulationError,
    #[error("Cannot plot fit: Not all parameters have values: {0:?}")]
    MissingParameterValues(Vec<String>),
    #[error("Cannot plot fit: Model has no equations or parameters")]
    MissingModel,
    #[error("Measurement with id {0} has no data")]
    MeasurementEmpty(String),
}

#[cfg(test)]
mod tests {
    use crate::io::load_enzmldoc;

    use super::*;

    #[test]
    fn test_plot() {
        let enzmldoc = load_enzmldoc("tests/data/enzmldoc.json").unwrap();
        enzmldoc
            .plot_measurement("measurement0", true, true, Some(800), Some(600))
            .unwrap();
    }

    #[test]
    fn test_get_simulation_traces() {
        let enzmldoc = load_enzmldoc("tests/data/enzmldoc.json").unwrap();
        let measurement = enzmldoc.measurements.first().unwrap();

        let traces = get_simulation_traces(&enzmldoc, measurement).unwrap();
        assert_eq!(traces.len(), 2);
    }

    #[test]
    fn test_plot_measurement_missing_model() {
        let enzmldoc = load_enzmldoc("tests/data/enzmldoc_no_model.json").unwrap();

        let result = enzmldoc.plot_measurement("measurement0", true, true, Some(800), Some(600));
        assert!(result.is_err());
    }

    #[test]
    fn test_plot_measurement_missing_parameter_values() {
        let enzmldoc = load_enzmldoc("tests/data/enzmldoc_missing_param_value.json").unwrap();

        let result = enzmldoc.plot_measurement("measurement0", true, true, Some(800), Some(600));
        assert!(result.is_err());
    }

    #[test]
    fn test_plot_measurement_empty() {
        let mut enzmldoc = load_enzmldoc("tests/data/enzmldoc.json").unwrap();

        // Clear the data from the measurement
        for meas in enzmldoc.measurements.iter_mut() {
            for species_data in meas.species_data.iter_mut() {
                species_data.data = vec![];
                species_data.time = vec![];
            }
        }

        let result = enzmldoc.plot_measurement("measurement0", true, true, Some(800), Some(600));

        assert!(result.is_err());
    }
}
