use itertools::Itertools;
use plotly::{
    common::{Line, Marker, Mode},
    layout::Axis,
    Layout, Plot, Scatter,
};
use thiserror::Error;

use crate::prelude::{EnzymeMLDocument, Measurement};

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
        measurement_id: String,
        show: bool,
        show_fit: bool,
    ) -> Result<Plot, PlotError> {
        // Filter measurements if measurement_ids are provided
        let measurement: &Measurement = self
            .measurements
            .iter()
            .find(|meas| meas.id == measurement_id)
            .ok_or(PlotError::MeasurementNotFound(measurement_id))?;

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
            .width(800)
            .height(600)
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
    let setup: SimulationSetup = meas.try_into().unwrap();
    let initial_conditions: InitialCondition = meas.into();
    let solver = RK5::default();
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

#[derive(Error, Debug)]
pub enum PlotError {
    #[error("Measurement with id {0} not found")]
    MeasurementNotFound(String),
    #[error("Simulation failed")]
    SimulationError,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::io::load_enzmldoc;

    use super::*;

    #[test]
    fn test_plot() {
        let enzmldoc = load_enzmldoc(&PathBuf::from("tests/data/enzmldoc.json")).unwrap();
        enzmldoc
            .plot_measurement("measurement0".to_string(), true, true)
            .unwrap();
    }

    #[test]
    fn test_get_simulation_traces() {
        let enzmldoc = load_enzmldoc(&PathBuf::from("tests/data/enzmldoc.json")).unwrap();
        let measurement = enzmldoc.measurements.iter().next().unwrap();

        let traces = get_simulation_traces(&enzmldoc, measurement).unwrap();
        assert_eq!(traces.len(), 2);
    }
}
