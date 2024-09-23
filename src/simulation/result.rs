use std::collections::HashMap;

use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::Plot;
use serde::{Deserialize, Serialize};

/// SimulationResult struct represents the result of a simulation.
/// It contains a vector of time points and a hashmap of species data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub time: Vec<f64>,
    pub species: HashMap<String, Vec<f64>>,
    pub assignments: HashMap<String, Vec<f64>>,
}

/// PlotConfig struct represents the configuration for a plot.
/// It includes the title, dimensions, and axis labels for the plot.
pub struct PlotConfig {
    pub title: String,
    pub width: usize,
    pub height: usize,
    pub x_label: String,
    pub y_label: String,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Simulation Results".to_string(),
            width: 800,
            height: 600,
            x_label: "Time".to_string(),
            y_label: "Value".to_string(),
        }
    }
}

impl SimulationResult {
    /// Creates a new SimulationResult with the given time points.
    ///
    /// # Arguments
    ///
    /// * time - A vector of time points.
    ///
    /// # Returns
    ///
    /// Returns a new SimulationResult instance.
    pub fn new(time: Vec<f64>) -> Self {
        Self {
            time,
            species: HashMap::new(),
            assignments: HashMap::new(),
        }
    }

    /// Adds species data to the SimulationResult.
    ///
    /// # Arguments
    ///
    /// * species - The name of the species.
    /// * values - A vector of data points for the species.
    pub fn add_species(&mut self, species: String, values: Vec<f64>) {
        self.species.insert(species, values);
    }

    /// Adds assignment data to the SimulationResult.
    ///
    /// # Arguments
    ///
    /// * assignment - The name of the assignment.
    /// * values - A vector of data points for the assignment.
    pub fn add_assignment(&mut self, assignment: String, values: Vec<f64>) {
        self.assignments.insert(assignment, values);
    }

    /// Plots the simulation results using the provided plot configuration.
    ///
    /// # Arguments
    ///
    /// * `plot_config` - The configuration for the plot, including title, dimensions, and axis labels.
    /// * `show` - A boolean indicating whether to display the plot immediately.
    ///
    /// # Returns
    ///
    /// Returns a `Plot` object representing the simulation results.
    pub fn plot(&self, plot_config: PlotConfig, show: bool) -> Plot {
        // Plot results
        let mut plot: Plot = self.into();

        plot.set_layout(
            plotly::Layout::new()
                .title(plot_config.title)
                .width(plot_config.width)
                .height(plot_config.height)
                .x_axis(Axis::new().title(plot_config.x_label))
                .y_axis(Axis::new().title(plot_config.y_label)),
        );

        if show {
            plot.show();
        }

        plot
    }
}

/// Implements the conversion from a `SimulationResult` to a `Plot`.
impl From<&SimulationResult> for Plot {
    /// Converts a `SimulationResult` into a `Plot`.
    ///
    /// # Arguments
    ///
    /// * `result` - A reference to the `SimulationResult` to be converted.
    ///
    /// # Returns
    ///
    /// Returns a `Plot` object representing the simulation results.
    fn from(result: &SimulationResult) -> Self {
        let mut plot = Plot::new();
        let mut all_species = result.species.clone();
        all_species.extend(result.assignments.clone());

        for (species, values) in result.species.iter() {
            let trace = plotly::Scatter::new(result.time.clone(), values.clone())
                .name(species)
                .mode(Mode::LinesText);

            plot.add_trace(trace);
        }

        plot
    }
}
