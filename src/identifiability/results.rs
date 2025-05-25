use std::collections::HashMap;

use peroxide::{
    fuga::{CubicSpline, Spline},
    linspace, seq,
};
use plotly::{
    common::{Anchor, ColorScale, ColorScalePalette, Font, Line, Mode, Title},
    layout::{Annotation, Axis, GridPattern, LayoutGrid},
    Contour, Layout, Plot, Scatter,
};

const DEFAULT_HEIGHT: usize = 400;
const DEFAULT_WIDTH: usize = 500;
const UPPER_BOUND: f64 = 1.1;

/// Results from a profile likelihood analysis for a single parameter.
///
/// Contains the profile likelihood data for a single parameter, including
/// the parameter values tested, corresponding likelihood values, and likelihood ratios.
/// Also stores the parameter value with the best likelihood.
///
/// # Notes
///
/// - The `best_value` field stores the parameter value with the best (lowest) likelihood.
/// - The `param_name` field stores the name of the profiled parameter.
/// - The `param_values` field stores the vector of parameter values tested during profiling.
/// - The `likelihoods` field stores the vector of likelihood values corresponding to each parameter value.
/// - The `ratios` field stores the vector of likelihood ratios (normalized to the best likelihood).
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// The parameter value with the best (lowest) likelihood
    pub(crate) best_value: f64,
    /// Name of the profiled parameter
    pub(crate) param_name: String,
    /// Vector of parameter values tested during profiling
    pub(crate) param_values: Vec<f64>,
    /// Vector of likelihood values corresponding to each parameter value
    pub(crate) likelihoods: Vec<f64>,
    /// Vector of likelihood ratios (normalized to the best likelihood)
    pub(crate) ratios: Vec<f64>,
}

impl ProfileResult {
    /// Creates a new ProfileResult instance.
    ///
    /// # Arguments
    ///
    /// * `best_value` - The parameter value with the best likelihood
    /// * `param_name` - Name of the profiled parameter
    /// * `param_values` - Vector of parameter values tested
    /// * `likelihoods` - Vector of likelihood values for each parameter value
    /// * `ratios` - Vector of likelihood ratios
    pub fn new(
        best_value: f64,
        param_name: String,
        param_values: Vec<f64>,
        likelihoods: Vec<f64>,
        ratios: Vec<f64>,
    ) -> Self {
        Self {
            best_value,
            param_name,
            param_values,
            likelihoods,
            ratios,
        }
    }

    /// Returns the parameter value with the best likelihood.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Returns the name of the profiled parameter.
    pub fn param_name(&self) -> &str {
        &self.param_name
    }

    /// Returns the vector of parameter values tested during profiling.
    pub fn param_values(&self) -> &[f64] {
        &self.param_values
    }

    /// Returns the vector of likelihood values for each parameter value.
    pub fn likelihoods(&self) -> &[f64] {
        &self.likelihoods
    }

    /// Returns the vector of likelihood ratios (normalized to the best likelihood).
    pub fn ratios(&self) -> &[f64] {
        &self.ratios
    }
}

/// Collection of profile likelihood results for multiple parameters.
///
/// Provides convenient access to individual profile results and methods
/// for visualization and conversion to other formats.
///
/// # Notes
///
/// - This struct is a collection of `ProfileResult` objects, one for each parameter profiled.
/// - It provides methods for accessing individual profile results and converting the results
///   to other formats, such as a `HashMap` or a `Plot`.
#[derive(Debug, Clone)]
pub struct ProfileResults(pub Vec<ProfileResult>);

impl ProfileResults {
    /// Returns the number of parameter profiles in the collection.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Checks if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a reference to the first profile result, if any.
    pub fn first(&self) -> Option<&ProfileResult> {
        self.0.first()
    }

    /// Returns a reference to the last profile result, if any.
    pub fn last(&self) -> Option<&ProfileResult> {
        self.0.last()
    }

    /// Returns a reference to the profile result for the given index.
    pub fn get(&self, index: usize) -> Option<&ProfileResult> {
        self.0.get(index)
    }
}

impl From<ProfileResults> for HashMap<String, ProfileResult> {
    /// Converts the profile results into a HashMap keyed by parameter name.
    ///
    /// This allows easy lookup of profile results by parameter name.
    fn from(results: ProfileResults) -> HashMap<String, ProfileResult> {
        results
            .0
            .into_iter()
            .map(|r| (r.param_name.clone(), r))
            .collect()
    }
}

impl IntoIterator for ProfileResults {
    type Item = ProfileResult;
    type IntoIter = std::vec::IntoIter<ProfileResult>;

    /// Allows iterating over the profile results.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Plots a pair of parameters against each other using a contour plot.
///
/// This function creates a contour plot showing the likelihood ratio between two parameters.
/// The plot is created using the `plotly` crate.
///
/// # Arguments
///
/// * `results` - The profile results to plot
///
/// # Returns
///
/// A `Plot` object containing the contour plot
pub fn plot_pair_contour(results: &ProfileResults) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();
    let mut n_plots = 1;

    // Get every pair of results (i, j) where i < j
    for i in 0..results.len() {
        for j in i + 1..results.len() {
            let result1 = results.get(i).unwrap();
            let result2 = results.get(j).unwrap();
            let combined_name = format!(
                "{} vs {}",
                to_html_sub(&result1.param_name),
                to_html_sub(&result2.param_name)
            );

            let trace = Contour::new(
                result1.param_values.clone(),
                result2.param_values.clone(),
                result1.ratios.clone(),
            )
            .name(&combined_name)
            .x_axis(&format!("x{}", n_plots))
            .y_axis(&format!("y{}", n_plots))
            .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

            layout.add_annotation(
                Annotation::new()
                    .y_ref(format!("y{} domain", n_plots))
                    .y_anchor(Anchor::Bottom)
                    .y(1)
                    .text(format!("<b>{}</b>", combined_name))
                    .x_ref(format!("x{} domain", n_plots))
                    .x_anchor(Anchor::Center)
                    .x(0.5)
                    .show_arrow(false),
            );

            plot.add_trace(trace);
            n_plots += 1;
        }
    }

    let n_rows = n_plots / 2;
    layout = layout.grid(
        LayoutGrid::new()
            .columns(if n_plots == 2 { 1 } else { 2 })
            .rows(n_rows)
            .pattern(GridPattern::Independent),
    );

    plot.set_layout(layout);

    plot
}

/// Converts ProfileResults into a multi-panel plotly Plot for visualization.
///
/// Creates a grid of line plots showing likelihood ratios for all profiled parameters.
/// This implementation allows easy visualization of multiple parameter profiles at once.
///
/// # Notes
///
/// - This implementation creates a grid of line plots, one for each parameter profiled.
/// - Each plot shows the likelihood ratio vs parameter value.
/// - The plots are arranged in a grid, with two columns and a number of rows equal to the
///   number of parameters profiled.
/// - Each plot includes a title, x-axis label, and y-axis label.
/// - The x-axis label is the parameter name.
/// - The y-axis label is "Likelihood Ratio".
impl From<ProfileResults> for Plot {
    fn from(results: ProfileResults) -> Self {
        let mut plot = Plot::new();
        let mut layout = Layout::new();

        for (i, result) in results.0.iter().enumerate() {
            let sup_name = to_html_sub(&result.param_name);
            let markers_trace = Scatter::new(result.param_values.clone(), result.ratios.clone())
                .name(format!("Likelihood Ratio of {}", &sup_name))
                .mode(Mode::Markers)
                .x_axis(format!("x{}", i + 1))
                .y_axis(format!("y{}", i + 1));

            let ratios = result
                .ratios
                .clone()
                .into_iter()
                .map(|r| if r < 0.1 { 0.0 } else { r })
                .collect::<Vec<f64>>();
            let (query_values, interpolated_ratios) =
                interpolate_ratios(&ratios, &result.param_values);
            let line_trace = Scatter::new(query_values, interpolated_ratios)
                .name(format!("Interpolated Likelihood Ratio of {}", &sup_name))
                .mode(Mode::Lines)
                .x_axis(format!("x{}", i + 1))
                .y_axis(format!("y{}", i + 1))
                .line(
                    Line::new()
                        .width(1.3)
                        .dash(plotly::common::DashType::DashDot)
                        .color("gray"),
                );

            plot.add_trace(line_trace);
            plot.add_trace(markers_trace);

            layout.add_annotation(
                Annotation::new()
                    .y_ref(format!("y{} domain", i + 1))
                    .y_anchor(Anchor::Bottom)
                    .y(1)
                    .text(&sup_name)
                    .x_ref(format!("x{} domain", i + 1))
                    .x_anchor(Anchor::Center)
                    .x(0.5)
                    .font(Font::new().size(18))
                    .show_arrow(false),
            );

            let x_title = format!("Parameter Value of {}", &sup_name);
            let y_title = format!("Likelihood Ratio of {}", &sup_name);

            layout = match i + 1 {
                1 => layout
                    .x_axis(Axis::new().title(Title::from(x_title)))
                    .y_axis(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                2 => layout
                    .x_axis2(Axis::new().title(Title::from(x_title)))
                    .y_axis2(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                3 => layout
                    .x_axis3(Axis::new().title(Title::from(x_title)))
                    .y_axis3(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                4 => layout
                    .x_axis4(Axis::new().title(Title::from(x_title)))
                    .y_axis4(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                5 => layout
                    .x_axis5(Axis::new().title(Title::from(x_title)))
                    .y_axis5(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                6 => layout
                    .x_axis6(Axis::new().title(Title::from(x_title)))
                    .y_axis6(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                7 => layout
                    .x_axis7(Axis::new().title(Title::from(x_title)))
                    .y_axis7(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                8 => layout
                    .x_axis8(Axis::new().title(Title::from(x_title)))
                    .y_axis8(
                        Axis::new()
                            .title(Title::from(y_title))
                            .range(vec![0.0, UPPER_BOUND]),
                    ),
                _ => layout,
            };
        }

        // Determine the number of columns based on the number of parameters
        // There should be two columns, and the number of rows should be the number of parameters
        let n_params = results.0.len();
        let n_cols = 2;
        let n_rows = n_params.div_ceil(n_cols);

        layout = layout
            .grid(
                LayoutGrid::new()
                    .columns(if n_params == 1 { 1 } else { 2 })
                    .rows(n_rows)
                    .pattern(GridPattern::Independent),
            )
            .height(DEFAULT_HEIGHT * n_rows)
            .width(DEFAULT_WIDTH * n_cols);

        plot.set_layout(layout);

        plot
    }
}

fn interpolate_ratios(ratios: &[f64], param_values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (sorted_param_values, sorted_ratios) = sort_ratios_and_param_values(ratios, param_values);
    let cs = CubicSpline::from_nodes(&sorted_param_values, &sorted_ratios).unwrap();
    let query_values = linspace!(
        sorted_param_values[0],
        sorted_param_values[sorted_param_values.len() - 1],
        50
    );
    let interpolated_ratios = cs
        .eval_vec(&query_values)
        .into_iter()
        .map(|x| x.max(0.01))
        .collect::<Vec<f64>>();
    (query_values, interpolated_ratios)
}

fn sort_ratios_and_param_values(ratios: &[f64], param_values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Create indices for sorting
    let mut indices: Vec<usize> = (0..param_values.len()).collect();

    // Sort indices by parameter values in ascending order
    indices.sort_by(|&a, &b| {
        param_values[a]
            .partial_cmp(&param_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Create sorted vectors using the sorted indices
    let sorted_param_values: Vec<f64> = indices.iter().map(|&i| param_values[i]).collect();
    let sorted_ratios: Vec<f64> = indices.iter().map(|&i| ratios[i]).collect();

    (sorted_param_values, sorted_ratios)
}

/// Converts parameter names with underscores to HTML with subscripts
///
/// For example, "k_cat" becomes "k<sub>cat</sub>"
///
/// # Arguments
///
/// * `s` - The parameter name to convert
///
/// # Returns
///
/// The converted parameter name
///
/// # Notes
///
/// - This function converts parameter names with underscores to HTML with subscripts.
/// - For example, "k_cat" becomes "k<sub>cat</sub>".
/// - If the parameter name does not contain an underscore, it is returned unchanged.
fn to_html_sub(s: &str) -> String {
    if !s.contains('_') {
        return s.to_string();
    }

    let mut parts = s.splitn(2, '_');
    let base = parts.next().unwrap_or("");
    let superscript = parts.next().unwrap_or("");

    if superscript.is_empty() {
        base.to_string()
    } else {
        format!("{}<sub>{}</sub>", base, superscript)
    }
}
