//! MCMC Diagnostics Module
//!
//! This module provides comprehensive diagnostic tools for Markov Chain Monte Carlo (MCMC) sampling,
//! including convergence diagnostics, effective sample size calculations, and summary statistics.
//!
//! The main components are:
//! - `Diagnostics`: Main struct containing all diagnostic information
//! - `ParameterDiagnostics`: Per-parameter diagnostic statistics
//! - Convergence diagnostics (R-hat statistic)
//! - Effective sample size (ESS) calculations
//! - Highest Density Probability Intervals (HDPI)

use std::collections::HashMap;
use std::fmt::{self, Display};

use polars::prelude::*;
use tabled::{builder::Builder, settings::Style};

use crate::mcmc::output::SampleOutput;

/// Main diagnostics container holding all MCMC diagnostic information
///
/// This struct aggregates diagnostic statistics across all parameters and chains,
/// providing a comprehensive view of sampling performance and convergence.
#[derive(Debug, Clone)]
pub struct Diagnostics {
    /// Number of MCMC chains used in sampling
    num_chains: usize,
    /// Number of draws per chain (as f64 for calculations)
    num_draws: f64,
    /// Number of divergences
    divergences: Option<usize>,
    /// Map of parameter names to their diagnostic statistics
    parameter_diagnostics: HashMap<String, ParameterDiagnostics>,
}

/// Diagnostic statistics for a single parameter across all chains
///
/// Contains all relevant statistics needed to assess parameter estimation quality,
/// including central tendency, dispersion, convergence, and efficiency metrics.
#[derive(Debug, Clone)]
pub struct ParameterDiagnostics {
    /// Posterior mean estimate
    pub mean: f64,
    /// Posterior median estimate
    pub median: f64,
    /// Posterior standard deviation
    pub std: f64,
    /// R-hat convergence diagnostic (should be < 1.1 for convergence)
    pub rhat: f64,
    /// Effective sample size (measure of sampling efficiency)
    pub ess: f64,
    /// Monte Carlo Standard Error (MCSE)
    pub mcse: f64,
    /// Lower bound of 95% Highest Density Probability Interval
    pub hdpi_low: f64,
    /// Upper bound of 95% Highest Density Probability Interval
    pub hdpi_high: f64,
}

impl Diagnostics {
    /// Create diagnostics from MCMC output
    ///
    /// Analyzes all chains and parameters from the provided output to compute
    /// comprehensive diagnostic statistics.
    ///
    /// # Arguments
    /// * `output` - MCMC output implementing SampleOutput trait
    ///
    /// # Returns
    /// Complete diagnostics for all parameters and chains
    pub fn from_output<O>(output: &O, divergences: Option<usize>) -> Self
    where
        O: SampleOutput,
    {
        // Extract chains from output - these contain the actual MCMC samples
        let chains = output.get_chains().unwrap();
        let num_chains = chains.len();
        let num_draws = chains.iter().map(|c| c.height() as f64).sum::<f64>();

        // Get parameter names from the first chain (all chains should have same structure)
        let parameter_names: Vec<String> = chains[0]
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();

        let mut parameter_diagnostics = HashMap::new();

        // Calculate diagnostics for each parameter across all chains
        for param_name in &parameter_names {
            let param_diagnostics = calculate_parameter_diagnostics(&chains, param_name);
            parameter_diagnostics.insert(param_name.clone(), param_diagnostics);
        }

        Self {
            num_chains,
            num_draws,
            divergences,
            parameter_diagnostics,
        }
    }

    /// Get diagnostic statistics for a specific parameter
    ///
    /// # Arguments
    /// * `param_name` - Name of the parameter to retrieve diagnostics for
    ///
    /// # Returns
    /// Optional reference to parameter diagnostics (None if parameter not found)
    pub fn get_parameter_diagnostics(&self, param_name: &str) -> Option<&ParameterDiagnostics> {
        self.parameter_diagnostics.get(param_name)
    }

    /// Get the number of chains used in sampling
    pub fn get_num_chains(&self) -> usize {
        self.num_chains
    }

    /// Get the number of draws per chain
    pub fn get_num_draws(&self) -> f64 {
        self.num_draws
    }

    /// Get all parameter names that have diagnostics computed
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameter_diagnostics.keys().cloned().collect()
    }

    /// Check if all chains have converged based on R-hat criterion
    ///
    /// Convergence is assessed using the R-hat statistic with threshold < 1.1
    /// This is the standard criterion used in MCMC literature.
    ///
    /// # Returns
    /// true if all parameters have R-hat < 1.1, false otherwise
    pub fn is_converged(&self) -> bool {
        self.parameter_diagnostics
            .values()
            .all(|diag| diag.rhat < 1.1)
    }

    /// Print a formatted summary of all diagnostics to stdout
    ///
    /// This is a convenience method that displays the complete diagnostic table
    pub fn print_summary(&self) {
        println!("{self}");
    }
}

/// Creates a formatted table displaying parameter diagnostics
///
/// Generates a nicely formatted table with all diagnostic statistics for easy reading.
/// Parameters are sorted alphabetically for consistent display.
///
/// # Arguments
/// * `parameter_diagnostics` - Map of parameter names to their diagnostic statistics
///
/// # Returns
/// Formatted string containing the diagnostic table
fn create_parameter_table(parameter_diagnostics: &HashMap<String, ParameterDiagnostics>) -> String {
    let mut builder = Builder::default();

    // Add header row with all diagnostic column names
    builder.push_record(vec![
        "Parameter".to_string(),
        "Mean".to_string(),
        "Std".to_string(),
        "Median".to_string(),
        "R-hat".to_string(),
        "ESS".to_string(),
        "MCSE".to_string(),
        "MCSE ≤ 5% Std".to_string(),
        "HDPI (95%)".to_string(),
        "Converged".to_string(),
    ]);

    // Sort parameters by name for consistent display across runs
    let mut sorted_params: Vec<_> = parameter_diagnostics.iter().collect();
    sorted_params.sort_by(|a, b| a.0.cmp(b.0));

    // Add parameter data rows with formatted values
    for (param_name, diag) in sorted_params {
        let mcse_ok = if diag.mcse <= diag.std * 0.05 {
            "✅"
        } else {
            "❌"
        };
        builder.push_record(vec![
            param_name.clone(),
            format!("{:.4}", diag.mean),
            format!("{:.4}", diag.std),
            format!("{:.4}", diag.median),
            format!("{:.4}", diag.rhat),
            format!("{:.1}", diag.ess),
            format!("{:.4}", diag.mcse),
            format!("{mcse_ok}"),
            format!("[{:.4}, {:.4}]", diag.hdpi_low, diag.hdpi_high),
            // Visual indicator for convergence status
            if diag.rhat < 1.1 {
                "✅".to_string()
            } else {
                "❌".to_string()
            },
        ]);
    }

    // Apply rounded table style for better appearance
    let mut table = builder.build();
    table.with(Style::rounded());
    table.to_string()
}

impl Display for Diagnostics {
    /// Format diagnostics as a comprehensive summary table
    ///
    /// Creates a multi-section display with overall statistics and detailed parameter diagnostics
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Create the main table with overall statistics
        let mut builder = Builder::default();

        // Title section
        builder.push_record(vec!["MCMC Diagnostics Summary"]);
        builder.push_record(vec![""]);

        // Overall statistics section
        builder.push_record(vec!["Overall Statistics"]);
        builder.push_record(vec![format!("Chains: {}", self.num_chains)]);
        builder.push_record(vec![format!("Total draws: {}", self.num_draws)]);
        if let Some(divergences) = self.divergences {
            builder.push_record(vec![format!("Divergences: {}", divergences)]);
        }
        builder.push_record(vec![format!(
            "Converged: {}",
            if self.is_converged() {
                "✅ Yes"
            } else {
                "⚠️ No"
            }
        )]);

        builder.push_record(vec![""]);

        // Parameter-specific diagnostics table section
        if !self.parameter_diagnostics.is_empty() {
            builder.push_record(vec!["Parameter-Specific Diagnostics"]);
            builder.push_record(vec![create_parameter_table(&self.parameter_diagnostics)]);
        }

        // Apply sharp table style for the main container
        let mut table = builder.build();
        table.with(Style::sharp());
        write!(f, "{table}")
    }
}

impl Display for ParameterDiagnostics {
    /// Format parameter diagnostics as a compact summary line
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mean: {:.4}, Std: {:.4}, R-hat: {:.4}, ESS: {:.1}, HDPI: [{:.4}, {:.4}]",
            self.mean, self.std, self.rhat, self.ess, self.hdpi_low, self.hdpi_high
        )
    }
}

/// Calculate comprehensive diagnostic statistics for a single parameter
///
/// This function computes all diagnostic metrics for a parameter across all chains,
/// including central tendency, dispersion, convergence, and efficiency measures.
///
/// # Arguments
/// * `chains` - Vector of DataFrames, each representing one MCMC chain
/// * `param_name` - Name of the parameter to analyze
///
/// # Returns
/// Complete ParameterDiagnostics struct with all computed statistics
fn calculate_parameter_diagnostics(chains: &[DataFrame], param_name: &str) -> ParameterDiagnostics {
    let mut chain_means = Vec::new();
    let mut chain_vars = Vec::new();
    let mut all_values = Vec::new();
    let mut n_draws = 0;

    // Extract parameter values from each chain and compute per-chain statistics
    for chain in chains {
        let values = extract_parameter_values(chain, param_name);
        n_draws = values.len(); // Number of draws per chain

        // Calculate chain-specific mean and variance for R-hat computation
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        chain_means.push(mean);
        chain_vars.push(variance);
        all_values.extend(values);
    }

    // Calculate R-hat convergence diagnostic (Gelman-Rubin statistic)
    let rhat = calculate_rhat(&chain_means, &chain_vars, n_draws);

    // Calculate effective sample size (measure of sampling efficiency)
    let ess = calculate_effective_sample_size(chains, param_name);

    // Calculate overall posterior statistics across all chains
    let overall_mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let overall_var = all_values
        .iter()
        .map(|x| (x - overall_mean).powi(2))
        .sum::<f64>()
        / (all_values.len() - 1) as f64;
    let overall_std = overall_var.sqrt();

    // Calculate median (50th percentile)
    let mut sorted_values = all_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_values.len() % 2 == 0 {
        let mid = sorted_values.len() / 2;
        (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };

    // Calculate Monte Carlo Standard Error (MCSE)
    let mcse = if ess > 0.0 {
        overall_std / ess.sqrt()
    } else {
        f64::NAN
    };

    // Calculate 95% HDPI (Highest Density Probability Interval)
    // This is the narrowest interval containing 95% of the posterior mass
    let (hdpi_low, hdpi_high) = calculate_hdpi(&sorted_values, 0.95);

    ParameterDiagnostics {
        mean: overall_mean,
        median,
        std: overall_std,
        rhat,
        ess,
        mcse,
        hdpi_low,
        hdpi_high,
    }
}

/// Extract parameter values from a DataFrame chain
///
/// Helper function to safely extract f64 values for a specific parameter from a chain.
///
/// # Arguments
/// * `chain` - DataFrame containing MCMC samples
/// * `param_name` - Name of the parameter column to extract
///
/// # Returns
/// Vector of f64 values for the parameter
///
/// # Panics
/// Panics if parameter is not found or contains null values
fn extract_parameter_values(chain: &DataFrame, param_name: &str) -> Vec<f64> {
    match chain.column(param_name) {
        Ok(column) => column
            .f64()
            .expect("Parameter column should be f64")
            .into_iter()
            .map(|opt| opt.expect("No null values expected in parameter column"))
            .collect(),
        Err(_) => {
            panic!("Parameter '{param_name}' not found in chain");
        }
    }
}

/// Calculate R-hat statistic (Gelman-Rubin diagnostic) for convergence assessment
///
/// The R-hat statistic compares within-chain and between-chain variance to assess
/// whether chains have converged to the same distribution. Values close to 1.0
/// indicate convergence, while values > 1.1 suggest lack of convergence.
///
/// Formula: R-hat = sqrt(((n-1)/n * W + (1/n) * B) / W)
/// where W = within-chain variance, B = between-chain variance
///
/// # Arguments
/// * `chain_means` - Mean of each chain
/// * `chain_vars` - Variance of each chain  
/// * `n_draws` - Number of draws per chain
///
/// # Returns
/// R-hat statistic value
fn calculate_rhat(chain_means: &[f64], chain_vars: &[f64], n_draws: usize) -> f64 {
    let m = chain_means.len() as f64; // number of chains
    let n = n_draws as f64; // number of draws per chain

    // Between-chain variance B - measures how much chain means differ
    let overall_mean = chain_means.iter().sum::<f64>() / m;
    let b = chain_means
        .iter()
        .map(|&mean| (mean - overall_mean).powi(2))
        .sum::<f64>()
        * n
        / (m - 1.0);

    // Within-chain variance W - average of individual chain variances
    let w = chain_vars.iter().sum::<f64>() / m;

    // Pooled variance estimate - weighted combination of within and between variance
    let var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b;

    // R-hat statistic - ratio of pooled variance to within-chain variance
    (var_plus / w).sqrt()
}

/// Calculate effective sample size using autocorrelation-based method
///
/// ESS measures how many independent samples the MCMC chains are equivalent to,
/// accounting for autocorrelation. Higher ESS indicates more efficient sampling.
///
/// This implementation uses the improved algorithm from Vehtari et al. (2021)
/// with rank normalization and automatic windowing.
///
/// # Arguments
/// * `chains` - Vector of DataFrames containing MCMC samples
/// * `param` - Parameter name to calculate ESS for
///
/// # Returns
/// Effective sample size (bounded by total number of samples)
fn calculate_effective_sample_size(chains: &[DataFrame], param: &str) -> f64 {
    // Split each chain in half to increase number of sequences for better ESS estimation
    let draws = split_chains_in_half(chains, param);

    // Apply rank normalization for robustness to non-normal distributions
    let normalized_draws = apply_rank_normalization(draws);

    // Calculate ESS using autocorrelation-based method
    calculate_ess_from_autocorrelation(&normalized_draws)
}

/// Split each MCMC chain in half to create more sequences for ESS estimation
///
/// This technique improves the reliability of ESS calculations by increasing
/// the number of independent sequences available for analysis.
///
/// # Arguments
/// * `chains` - Vector of DataFrames containing MCMC samples
/// * `param` - Parameter name to extract values for
///
/// # Returns
/// Vector of parameter value sequences (2 per original chain)
fn split_chains_in_half(chains: &[DataFrame], param: &str) -> Vec<Vec<f64>> {
    let mut draws: Vec<Vec<f64>> = Vec::with_capacity(chains.len() * 2);

    for chain in chains {
        let values = extract_parameter_values(chain, param);
        let n = values.len();

        if n < 4 {
            // Not enough samples for reliable ESS calculation
            return vec![values];
        }

        let half = n / 2;
        draws.push(values[..half].to_vec());
        draws.push(values[half..].to_vec());
    }

    draws
}

/// Apply rank normalization to make ESS calculation robust to non-normal distributions
///
/// Converts parameter values to normal scores using the inverse normal CDF.
/// This transformation preserves the rank order while making the distribution
/// approximately normal, which improves ESS estimation accuracy.
///
/// # Arguments
/// * `draws` - Vector of parameter value sequences to normalize
///
/// # Returns
/// Vector of rank-normalized sequences
fn apply_rank_normalization(mut draws: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    for sequence in &mut draws {
        let n = sequence.len();
        let mut indices: Vec<_> = (0..n).collect();

        // Sort indices by their corresponding values
        indices.sort_by(|&i, &j| sequence[i].total_cmp(&sequence[j]));

        // Replace values with normal scores Φ⁻¹(rank / (N+1))
        for (rank, &i) in indices.iter().enumerate() {
            let p = (rank + 1) as f64 / (n as f64 + 1.0);
            sequence[i] = statrs::function::erf::erf_inv(2.0 * p - 1.0) * 2f64.sqrt();
        }
    }

    draws
}

/// Calculate ESS using autocorrelation-based method with automatic windowing
///
/// Implements the improved algorithm from Vehtari et al. (2021) with
/// Initial Positive Sequence (IPS) truncation for robust ESS estimation.
///
/// # Arguments
/// * `draws` - Vector of rank-normalized parameter sequences
///
/// # Returns
/// Effective sample size (bounded by total number of samples)
fn calculate_ess_from_autocorrelation(draws: &[Vec<f64>]) -> f64 {
    let m = draws.len(); // number of sequences
    let n = draws[0].len(); // length of each sequence

    // Calculate variance for each sequence
    let variances = calculate_sequence_variances(draws);

    // Calculate autocorrelation sum using IPS truncation
    let gamma_sum = calculate_autocorrelation_sum(draws, &variances);

    // Calculate ESS using the autocorrelation sum
    let ess = (m as f64 * n as f64) / (1.0 + 2.0 * gamma_sum);
    ess.min(m as f64 * n as f64) // ESS cannot exceed total number of draws
}

/// Calculate variance for each parameter sequence
///
/// # Arguments
/// * `draws` - Vector of parameter sequences
///
/// # Returns
/// Vector of variances (one per sequence)
fn calculate_sequence_variances(draws: &[Vec<f64>]) -> Vec<f64> {
    draws
        .iter()
        .map(|sequence| {
            let n = sequence.len();
            let mean = sequence.iter().sum::<f64>() / n as f64;
            let variance =
                sequence.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            variance.max(1e-12) // guard against variance → 0
        })
        .collect()
}

/// Calculate sum of autocorrelation pairs with IPS truncation
///
/// Uses Initial Positive Sequence truncation: stops summing when
/// autocorrelation pairs become non-positive, which indicates
/// the end of meaningful autocorrelation structure.
///
/// # Arguments
/// * `draws` - Vector of parameter sequences
/// * `variances` - Pre-calculated variances for each sequence
///
/// # Returns
/// Sum of autocorrelation pairs for ESS calculation
fn calculate_autocorrelation_sum(draws: &[Vec<f64>], variances: &[f64]) -> f64 {
    let n = draws[0].len();
    let mut rho_prev = 1.0; // ρ₀ = 1 (autocorrelation at lag 0)
    let mut gamma_sum = 0.0; // Σ (ρ_{2k}+ρ_{2k+1}) - sum of autocorrelation pairs

    for lag in 1..n {
        // Calculate mean autocorrelation over all sequences at this lag
        let rho_t = calculate_mean_autocorrelation_at_lag(draws, variances, lag);

        if lag % 2 == 0 {
            // Pair up consecutive autocorrelations (ρ_{2k-1}, ρ_{2k}) → γ_k
            let gamma = rho_prev + rho_t;
            if gamma <= 0.0 {
                break; // Initial Positive Sequence (IPS) truncation
            }
            gamma_sum += gamma;
        }
        rho_prev = rho_t;
    }

    gamma_sum
}

/// Calculate mean autocorrelation across all sequences at a specific lag
///
/// # Arguments
/// * `draws` - Vector of parameter sequences
/// * `variances` - Pre-calculated variances for each sequence
/// * `lag` - Time lag for autocorrelation calculation
///
/// # Returns
/// Mean autocorrelation at the specified lag
fn calculate_mean_autocorrelation_at_lag(draws: &[Vec<f64>], variances: &[f64], lag: usize) -> f64 {
    let m = draws.len();
    let n = draws[0].len();

    let mut rho_sum = 0.0;

    for (sequence, &variance) in draws.iter().zip(variances) {
        let mut numerator = 0.0;
        for i in 0..(n - lag) {
            numerator += sequence[i] * sequence[i + lag];
        }
        rho_sum += numerator / ((n - lag) as f64 * variance);
    }

    rho_sum / m as f64
}

/// Calculate Highest Density Probability Interval (HDPI)
///
/// HDPI is the shortest interval that contains a specified probability mass.
/// Unlike equal-tailed intervals, HDPI is optimal for skewed distributions
/// as it contains the most probable values.
///
/// # Arguments
/// * `sorted_values` - Parameter values sorted in ascending order
/// * `prob` - Probability mass to include (e.g., 0.95 for 95% interval)
///
/// # Returns
/// Tuple of (lower_bound, upper_bound) for the HDPI
fn calculate_hdpi(sorted_values: &[f64], prob: f64) -> (f64, f64) {
    let n = sorted_values.len();
    let exclude = ((1.0 - prob) * n as f64) as usize; // number of points to exclude
    let include = n - exclude; // number of points to include

    if include == 0 {
        // Edge case: include all values
        return (sorted_values[0], sorted_values[n - 1]);
    }

    // Find the shortest interval containing the required probability mass
    let mut best_width = f64::INFINITY;
    let mut best_low = 0;
    let mut best_high = include - 1;

    // Try all possible intervals of the required size
    for i in 0..=exclude {
        let low = i;
        let high = i + include - 1;

        if high < n {
            let width = sorted_values[high] - sorted_values[low];
            if width < best_width {
                best_width = width;
                best_low = low;
                best_high = high;
            }
        }
    }

    (sorted_values[best_low], sorted_values[best_high])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::output::DataFrameOutput;

    #[test]
    fn test_rhat_calculation_converged_chains() {
        // Create synthetic data for testing - two chains with very similar means and sufficient variability
        let mut output = DataFrameOutput::new(vec!["param1", "param2"]).unwrap();
        output.add_chains(&["chain_0", "chain_1"]).unwrap();

        // Use random values with similar means but sufficient within-chain variance
        // to create truly converged chains
        let base_values: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                (
                    1.0 + 0.3 * t.sin() + 0.1 * (2.0 * t).cos(), // More variation for param1
                    2.0 + 0.3 * t.cos() + 0.1 * (1.5 * t).sin(), // More variation for param2
                )
            })
            .collect();

        for (val1, val2) in base_values {
            // Very small differences between chains (much smaller than within-chain variation)
            let chain_0_sample = [val1, val2];
            let chain_1_sample = [val1 + 0.01, val2 + 0.01]; // Tiny difference: 0.01

            output.add_draw(&chain_0_sample, "chain_0").unwrap();
            output.add_draw(&chain_1_sample, "chain_1").unwrap();
        }

        let diagnostics = Diagnostics::from_output(&output, None);

        // For converged chains, R-hat should be close to 1.0
        assert!(diagnostics.is_converged(), "Chains should be converged");

        // Check parameter-specific diagnostics
        let param1_diag = diagnostics.get_parameter_diagnostics("param1").unwrap();
        assert!(param1_diag.rhat < 1.1, "param1 R-hat should be < 1.1");
        assert!(
            param1_diag.mean > 0.8 && param1_diag.mean < 1.3,
            "param1 mean should be around 1.0"
        );

        let param2_diag = diagnostics.get_parameter_diagnostics("param2").unwrap();
        assert!(param2_diag.rhat < 1.1, "param2 R-hat should be < 1.1");
        assert!(
            param2_diag.mean > 1.8 && param2_diag.mean < 2.3,
            "param2 mean should be around 2.0"
        );

        println!("Converged chains test:");
        diagnostics.print_summary();
    }

    #[test]
    fn test_rhat_calculation_diverged_chains() {
        // Create synthetic data for testing - two chains with very different means
        let mut output = DataFrameOutput::new(vec!["param1"]).unwrap();
        output.add_chains(&["chain_0", "chain_1"]).unwrap();

        // Add samples that should NOT converge (very different means between chains)
        for i in 0..100 {
            let chain_0_sample = [1.0 + 0.1 * (i as f64).sin()];
            let chain_1_sample = [5.0 + 0.1 * (i as f64).sin()]; // Very different mean

            output.add_draw(&chain_0_sample, "chain_0").unwrap();
            output.add_draw(&chain_1_sample, "chain_1").unwrap();
        }

        let diagnostics = Diagnostics::from_output(&output, None);

        // For diverged chains, R-hat should be much larger than 1.0
        assert!(
            !diagnostics.is_converged(),
            "Chains should not be considered converged"
        );

        println!("Diverged chains test:");
        diagnostics.print_summary();
    }

    #[test]
    fn test_parameter_diagnostics_calculation() {
        // Create simple synthetic data
        let mut output = DataFrameOutput::new(vec!["test_param"]).unwrap();
        output.add_chains(&["chain_0", "chain_1"]).unwrap();

        // Add known values to verify mean calculation
        let known_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for &val in &known_values {
            output.add_draw(&[val], "chain_0").unwrap();
            output.add_draw(&[val + 0.1], "chain_1").unwrap(); // Slightly different
        }

        let diagnostics = Diagnostics::from_output(&output, None);
        let param_diag = diagnostics.get_parameter_diagnostics("test_param").unwrap();

        // Check that the mean is approximately correct
        let expected_mean = (known_values.iter().sum::<f64>()
            + known_values.iter().map(|x| x + 0.1).sum::<f64>())
            / 10.0;
        assert!(
            (param_diag.mean - expected_mean).abs() < 0.01,
            "Mean calculation should be accurate"
        );

        println!("Parameter diagnostics test:");
        diagnostics.print_summary();
    }
}
