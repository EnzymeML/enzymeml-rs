use std::collections::HashMap;

use ndarray::Array2;
use plotly::layout::Axis;
use plotly::Plot;
use plotly::{common::Mode, Scatter};
use serde::{Deserialize, Serialize};

pub type PlotTraces = Vec<Box<Scatter<f64, f64>>>;

/// Represents the result of a simulation.
///
/// Contains time points, species data, and assignment data from the simulation.
///
/// # Fields
///
/// * `time` - Vector of time points at which the simulation was evaluated
/// * `species` - HashMap mapping species names to their concentration values over time
/// * `assignments` - HashMap mapping assignment names to their calculated values over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub time: Vec<f64>,
    pub species: HashMap<String, Vec<f64>>,
    pub assignments: HashMap<String, Vec<f64>>,
}

/// Configuration options for plotting simulation results.
///
/// # Fields
///
/// * `title` - Title of the plot
/// * `width` - Width of the plot in pixels
/// * `height` - Height of the plot in pixels
/// * `x_label` - Label for the x-axis
/// * `y_label` - Label for the y-axis
pub struct PlotConfig {
    pub title: String,
    pub width: usize,
    pub height: usize,
    pub x_label: String,
    pub y_label: String,
}

impl Default for PlotConfig {
    /// Creates a default plot configuration.
    ///
    /// # Returns
    ///
    /// A PlotConfig with default values:
    /// - Title: "Simulation Results"
    /// - Width: 800px
    /// - Height: 600px
    /// - X Label: "Time"
    /// - Y Label: "Value"
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
    /// * `time` - A vector of time points at which the simulation was evaluated
    ///
    /// # Returns
    ///
    /// A new SimulationResult instance with empty species and assignments maps.
    pub fn new(time: Vec<f64>) -> Self {
        Self {
            time,
            species: HashMap::new(),
            assignments: HashMap::new(),
        }
    }

    /// Adds species concentration data to the SimulationResult.
    ///
    /// # Arguments
    ///
    /// * `species` - The name of the species
    /// * `values` - Vector of concentration values for the species over time
    pub fn add_species(&mut self, species: String, values: Vec<f64>) {
        self.species.insert(species, values);
    }

    /// Adds assignment data to the SimulationResult.
    ///
    /// # Arguments
    ///
    /// * `assignment` - The name of the assignment
    /// * `values` - Vector of calculated values for the assignment over time
    pub fn add_assignment(&mut self, assignment: String, values: Vec<f64>) {
        self.assignments.insert(assignment, values);
    }

    /// Creates a plot of the simulation results.
    ///
    /// # Arguments
    ///
    /// * `plot_config` - Configuration options for the plot
    /// * `show` - Whether to display the plot immediately
    ///
    /// # Returns
    ///
    /// A Plot object containing the simulation results visualization.
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

    /// Gets the concentration data for a specific species.
    ///
    /// # Arguments
    ///
    /// * `species_id` - The ID of the species to get data for
    ///
    /// # Returns
    ///
    /// A vector of concentration values for the species over time
    ///
    /// # Panics
    ///
    /// Panics if the species ID is not found in the simulation results
    pub fn get_species_data(&self, species_id: &str) -> &Vec<f64> {
        self.species
            .get(species_id)
            .unwrap_or_else(|| panic!("Species {} not found", species_id))
    }
}

/// Implements conversion from SimulationResult to Plot.
impl From<&SimulationResult> for Plot {
    /// Converts a SimulationResult into a Plot visualization.
    ///
    /// # Arguments
    ///
    /// * `result` - The SimulationResult to convert
    ///
    /// # Returns
    ///
    /// A Plot object containing line traces for each species and assignment.
    fn from(result: &SimulationResult) -> Self {
        let mut plot = Plot::new();
        let traces: Vec<Box<Scatter<f64, f64>>> = result.into();

        for trace in traces {
            plot.add_trace(trace);
        }

        plot
    }
}

/// Implements conversion from SimulationResult to a vector of Scatter plot traces.
///
/// Creates line plots for both species concentrations and assignment values over time.
/// Each species and assignment gets its own trace with a unique name and line style.
///
/// # Arguments
/// * `result` - The SimulationResult to convert into plot traces
///
/// # Returns
/// * A vector of Scatter plot traces, one for each species and assignment
impl From<&SimulationResult> for Vec<Box<Scatter<f64, f64>>> {
    fn from(result: &SimulationResult) -> Self {
        let mut traces = Vec::new();

        // Plot species
        for (species, values) in result.species.iter() {
            let trace = plotly::Scatter::new(result.time.clone(), values.clone())
                .name(format!("{} Fit", species))
                .mode(Mode::LinesText);

            traces.push(trace);
        }

        // Plot assignments
        for (assignment, values) in result.assignments.iter() {
            let trace = plotly::Scatter::new(result.time.clone(), values.clone())
                .name(format!("{} Fit", assignment))
                .mode(Mode::LinesText);
            traces.push(trace);
        }

        traces
    }
}

/// Implements conversion from SimulationResult to a 2D array.
impl From<SimulationResult> for ndarray::Array2<f64> {
    /// Converts a SimulationResult into a 2D array.
    ///
    /// The resulting array has dimensions [time_points × num_species].
    /// Each row represents a time point and each column represents a species.
    /// Species are sorted alphabetically by name.
    ///
    /// # Arguments
    ///
    /// * `result` - The SimulationResult to convert
    ///
    /// # Returns
    ///
    /// A 2D array containing the species concentration data. Shape is [time_points × num_species].
    fn from(result: SimulationResult) -> Self {
        let mut array = Array2::zeros((result.time.len(), result.species.len()));
        let mut species_names = result.species.keys().collect::<Vec<&String>>();
        species_names.sort();

        for i in 0..result.time.len() {
            for (j, species) in species_names.iter().enumerate() {
                array[[i, j]] = result.species[*species][i];
            }
        }

        array
    }
}
