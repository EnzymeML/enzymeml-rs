use plotly::{
    common::Mode,
    layout::{Axis, GridPattern, LayoutGrid},
    Layout, Plot, Scatter,
};

use crate::prelude::{
    error::SimulationError, init_cond::InitialCondition, result::PlotTraces, system::ODESystem,
    EnzymeMLDocument, Measurement, SimulationResult, SimulationSetup,
};

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
    pub fn plot(
        &self,
        columns: Option<usize>,
        show: bool,
        measurement_ids: Option<Vec<String>>,
        show_fit: bool,
    ) -> Result<Plot, SimulationError> {
        // Determine the number of rows
        let n_meas = self.measurements.len();
        let columns = if n_meas == 1 { 1 } else { columns.unwrap_or(2) };
        let rows = (n_meas as f32 / columns as f32).ceil() as usize;

        // Filter measurements if measurement_ids are provided
        let measurements: Vec<&Measurement> = match measurement_ids {
            Some(ids) => ids
                .into_iter()
                .map(|id| self.measurements.iter().find(|meas| meas.id == id).unwrap())
                .collect(),
            None => self.measurements.iter().collect(),
        };

        // Create a plot
        let mut plot = Plot::new();

        for meas in measurements {
            let traces: Vec<Box<Scatter<f64, f64>>> = meas.into();
            for trace in traces {
                plot.add_trace(trace);
            }

            if show_fit {
                let traces = get_simulation_traces(self, meas)?;
                for trace in traces {
                    plot.add_trace(trace);
                }
            }
        }

        // Create a layout
        let layout = Layout::new()
            .title(self.name.clone())
            .grid(
                LayoutGrid::new()
                    .rows(rows)
                    .columns(columns)
                    .pattern(GridPattern::Independent),
            )
            .show_legend(true) // Always show the legend
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

        for meas_data in measurement.species_data.iter() {
            if let (Some(time), Some(data)) = (&meas_data.time, &meas_data.data) {
                let plot = Scatter::new(time.clone(), data.clone())
                    .name(&meas_data.species_id)
                    .mode(Mode::Markers)
                    .x_axis("Time")
                    .y_axis("Concentration");
                plots.push(plot);
            }
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
fn get_simulation_traces(
    doc: &EnzymeMLDocument,
    meas: &Measurement,
) -> Result<PlotTraces, SimulationError> {
    let setup: SimulationSetup = meas.try_into().unwrap();
    let initial_conditions: InitialCondition = meas.into();

    let system: ODESystem = doc.try_into().unwrap();

    let result =
        system.integrate::<SimulationResult>(&setup, initial_conditions, None, None, None)?;

    let traces: Vec<Box<Scatter<f64, f64>>> = (&result).into();
    Ok(traces)
}
