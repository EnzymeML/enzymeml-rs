//! Likelihood functions for Bayesian inference in EnzymeML.
//!
//! This module provides likelihood functions that can be used with MCMC sampling
//! and other Bayesian inference methods. All likelihood functions implement both
//! the `Likelihood` and `ObjectiveFunction` traits, making them compatible with
//! both optimization and sampling algorithms.
//!
//! # Available Likelihood Functions
//!
//! - [`NormalLikelihood`]: Assumes normally distributed residuals (Gaussian errors)
//! - [`LaplaceLikelihood`]: Assumes Laplace distributed residuals (robust to outliers)
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use enzymeml::mcmc::likelihood::{NormalLikelihood, Likelihood};
//! use ndarray::Array2;
//!
//! // Create a normal likelihood with sigma = 1.0
//! let likelihood = NormalLikelihood::new(1.0);
//!
//! // Compute log-likelihood for some residuals
//! let residuals = Array2::from_shape_vec((2, 2), vec![0.1, -0.2, 0.3, -0.1]).unwrap();
//! let log_like = likelihood.log_likelihood(&residuals, &[1.0]);
//! ```

use std::fmt;

use crate::{objective::error::ObjectiveError, prelude::ObjectiveFunction};
use ndarray::{Array1, Array2, Array3, Axis};
use regex::Regex;
use variantly::Variantly;

/// Generic vectorized chain rule implementation for likelihood gradients.
///
/// This function efficiently computes the gradient of log-likelihood with respect to
/// model parameters using vectorized ndarray operations. It applies the chain rule:
/// ∂L/∂θ = Σ(∂L/∂r × ∂r/∂θ)
///
/// # Performance Benefits
///
/// - Uses vectorized operations instead of nested loops
/// - Leverages SIMD instructions for element-wise operations
/// - Optimal memory access patterns
/// - Potentially uses BLAS operations under the hood
///
/// # Arguments
///
/// * `residuals` - 2D array of residuals with shape (n_timepoints, n_species)
/// * `sensitivities` - 3D array of sensitivities with shape (n_timepoints, n_species, n_parameters)
/// * `dl_dr_fn` - Function that computes ∂L/∂r for all residuals
///
/// # Returns
///
/// 1D array of gradients with shape (n_parameters,)
///
/// # Mathematical Details
///
/// The computation involves:
/// 1. Compute ∂L/∂r for all residuals using the provided function
/// 2. Broadcast ∂L/∂r to match sensitivities dimensions
/// 3. Element-wise multiply: ∂L/∂r × ∂r/∂θ
/// 4. Sum over timepoints and species dimensions
///
/// This is equivalent to the nested loop approach but much faster:
/// ```text
/// gradient[k] = Σ_i Σ_j (∂L/∂r[i,j] × ∂r[i,j]/∂θ[k])
/// ```
#[inline(always)]
fn apply_chain_rule<F>(
    residuals: &Array2<f64>,
    sensitivities: &Array3<f64>,
    dl_dr_fn: F,
) -> Array1<f64>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    // Step 1: Compute ∂L/∂r for all residuals (vectorized)
    let dl_dr = dl_dr_fn(residuals);

    // Step 2: Apply chain rule using vectorized operations
    // Broadcast dl_dr to match sensitivities dimensions: (n_timepoints, n_species, 1)
    let dl_dr_broadcasted = dl_dr.insert_axis(Axis(2));

    // Step 3: Element-wise multiply and sum over timepoints and species
    // This computes: Σ_i Σ_j (∂L/∂r[i,j] × ∂r[i,j]/∂θ[k]) for each k
    (&dl_dr_broadcasted * sensitivities)
        .sum_axis(Axis(0)) // Sum over timepoints
        .sum_axis(Axis(0)) // Sum over species
}

/// A trait for objective functions that can be used as likelihood functions
/// in Bayesian inference.
///
/// This trait extends `ObjectiveFunction` to provide likelihood-specific functionality
/// for MCMC samplers and other Bayesian methods. The key requirement is that the
/// `cost` method returns the log-likelihood (or log-probability density) of the
/// data given the model parameters.
///
/// # Usage in Bayesian Inference
///
/// Likelihood functions are used in Bayesian inference to:
/// - Calculate the probability of observed data given model parameters
/// - Combine with prior distributions to form posterior distributions
/// - Guide MCMC sampling algorithms toward high-probability parameter regions
///
/// # Implementation Requirements
///
/// Types implementing this trait must:
/// - Return log-likelihood values from their `cost` method
/// - Handle residuals (differences between model predictions and observations)
/// - Be suitable for use in probabilistic inference contexts
/// - Provide a `log_likelihood` method that can use dynamic parameters
/// - Provide analytical gradients via `log_likelihood_gradient`
///
/// # Dynamic vs Fixed Parameters
///
/// The trait provides two ways to compute log-likelihood:
/// 1. `log_likelihood`: Uses dynamic parameters passed as a slice
/// 2. `ObjectiveFunction::cost`: Uses fixed parameters stored in the struct
///
/// This flexibility allows the same likelihood function to be used in both
/// optimization contexts (fixed parameters) and sampling contexts (dynamic parameters).
pub trait Likelihood: ObjectiveFunction {
    /// Computes the log-likelihood for the given residuals and parameters.
    ///
    /// This method allows for dynamic parameter values, which is useful when
    /// parameters are being sampled or optimized during MCMC or other inference
    /// procedures.
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals (model predictions - observations)
    /// * `parameters` - Slice of parameter values (e.g., scale parameters)
    ///
    /// # Returns
    ///
    /// The total log-likelihood summed over all residuals.
    ///
    /// # Note
    ///
    /// The number and interpretation of parameters depends on the specific
    /// likelihood implementation. Use `n_parameters()` and `parameter_names()`
    /// to determine the expected parameter structure.
    fn log_likelihood(&self, residuals: &ndarray::Array2<f64>) -> f64;

    /// Computes the gradient of the log-likelihood with respect to model parameters.
    ///
    /// This method uses the chain rule to compute analytical gradients:
    /// ∂L/∂θ = (∂L/∂r) × (∂r/∂θ)
    ///
    /// Where:
    /// - L is the log-likelihood
    /// - r are the residuals (model predictions - observations)
    /// - θ are the model parameters
    /// - ∂r/∂θ are the sensitivities from the ODE system
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals with shape (n_timepoints, n_species)
    /// * `sensitivities` - 3D array of sensitivities with shape (n_timepoints, n_species, n_parameters)
    ///
    /// # Returns
    ///
    /// 1D array of gradients with shape (n_parameters,) representing ∂L/∂θ
    ///
    /// # Mathematical Details
    ///
    /// The gradient computation involves:
    /// 1. Computing ∂L/∂r for each residual
    /// 2. Applying the chain rule: ∂L/∂θ = Σ(∂L/∂r × ∂r/∂θ)
    /// 3. Summing over all timepoints and species
    ///
    /// This provides exact analytical gradients for gradient-based MCMC methods
    /// like Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS).
    fn log_likelihood_gradient(
        &self,
        residuals: &Array2<f64>,
        sensitivities: &Array3<f64>,
    ) -> Array1<f64>;
}

/// Normal (Gaussian) likelihood function for Bayesian inference.
///
/// This likelihood assumes that residuals (model predictions minus observations)
/// are normally distributed with mean 0 and standard deviation `sigma`. It's
/// commonly used when measurement errors are expected to be symmetric and
/// bell-shaped around the true values.
///
/// # Mathematical Form
///
/// For residuals r_i, the log-likelihood is:
/// ```text
/// log L = -N/2 * log(2π) - N * log(σ) - (1/2σ²) * Σ(r_i²)
/// ```
///
/// Where N is the total number of residuals.
///
/// # Gradient Form
///
/// The gradient with respect to model parameters θ is:
/// ```text
/// ∂L/∂θ = -1/σ² * Σ(r_i * ∂r_i/∂θ)
/// ```
///
/// # When to Use
///
/// - Measurement errors are expected to be symmetric around zero
/// - Errors follow a bell-shaped distribution
/// - No significant outliers are expected
/// - Classical least-squares assumptions hold
///
/// # Parameters
///
/// - `sigma`: Standard deviation parameter controlling the width of the normal distribution
///
/// # Usage
///
/// ```rust,ignore
/// use enzymeml::mcmc::likelihood::{NormalLikelihood, Likelihood};
/// use ndarray::Array2;
///
/// let likelihood = NormalLikelihood::new(1.0);
/// let residuals = Array2::from_shape_vec((2, 2), vec![0.1, -0.2, 0.3, -0.1]).unwrap();
/// let log_like = likelihood.log_likelihood(&residuals, &[1.0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NormalLikelihood {
    /// Standard deviation parameter of the normal distribution
    pub sigma: f64,
}

impl NormalLikelihood {
    /// Creates a new normal likelihood with the specified standard deviation.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Standard deviation parameter (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `sigma` is not positive.
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0, "Standard deviation must be positive");
        Self { sigma }
    }
}

impl Likelihood for NormalLikelihood {
    /// Computes the log-likelihood using dynamic parameters.
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals
    /// * `parameters` - Slice containing the standard deviation as the first element
    ///
    /// # Returns
    ///
    /// The total log-likelihood summed over all residuals.
    ///
    /// # Panics
    ///
    /// May panic if the parameters slice is empty or if the first parameter
    /// is not positive.
    #[inline(always)]
    fn log_likelihood(&self, residuals: &ndarray::Array2<f64>) -> f64 {
        // Use the first parameter as the standard deviation
        let sigma = self.sigma;
        let z = residuals / sigma;
        let log_norm_const = (sigma * (2.0 * std::f64::consts::PI).sqrt()).ln();
        let log_likelihood = -0.5 * &z * &z - log_norm_const;
        log_likelihood.sum()
    }

    /// Computes the gradient of the log-likelihood with respect to model parameters using vectorized operations.
    ///
    /// For normal likelihood: ∂L/∂θ = -1/σ² * Σ(r_i * ∂r_i/∂θ)
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals with shape (n_timepoints, n_species)
    /// * `sensitivities` - 3D array of sensitivities with shape (n_timepoints, n_species, n_parameters)
    ///
    /// # Returns
    ///
    /// 1D array of gradients with shape (n_parameters,)
    ///
    /// # Performance
    ///
    /// This implementation uses vectorized ndarray operations for maximum performance,
    /// avoiding nested loops and leveraging SIMD instructions.
    #[inline(always)]
    fn log_likelihood_gradient(
        &self,
        residuals: &Array2<f64>,
        sensitivities: &Array3<f64>,
    ) -> Array1<f64> {
        let sigma_squared = self.sigma;

        // Use vectorized chain rule: ∂L/∂r = -r/σ²
        apply_chain_rule(residuals, sensitivities, |r| -r / sigma_squared)
    }
}

impl ObjectiveFunction for NormalLikelihood {
    /// Computes the log-likelihood for normally distributed residuals using the fixed sigma.
    ///
    /// This implementation uses the `sigma` value stored in the struct, making it
    /// suitable for optimization contexts where the likelihood parameters are fixed.
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals (model predictions - observations)
    /// * `_` - Number of data points (unused in this implementation)
    ///
    /// # Returns
    ///
    /// The total log-likelihood summed over all residuals.
    ///
    /// # Mathematical Details
    ///
    /// For each residual r_ij, computes:
    /// - Standardized residual: z_ij = r_ij / σ
    /// - Log normalization constant: log(σ√(2π))
    /// - Log probability density: -0.5 * z_ij² - log(σ√(2π))
    ///
    /// Returns the sum of all log probability densities.
    #[inline(always)]
    fn cost(
        &self,
        residuals: &ndarray::Array2<f64>,
        _: usize,
    ) -> Result<f64, crate::objective::error::ObjectiveError> {
        // Vectorized computation of ln_pdf for all residuals at once (mu = 0)
        let z = residuals / self.sigma;
        let log_norm_const = (self.sigma * (2.0 * std::f64::consts::PI).sqrt()).ln();
        let log_likelihood = -0.5 * &z * &z - log_norm_const;

        Ok(log_likelihood.sum())
    }

    fn gradient(
        &self,
        _: Array2<f64>,
        _: &ndarray::Array3<f64>,
        _: usize,
    ) -> Result<ndarray::Array1<f64>, ObjectiveError> {
        todo!()
    }
}

impl fmt::Display for NormalLikelihood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N(mu=0, sigma={0})", self.sigma)
    }
}

/// Laplace (double exponential) likelihood function for Bayesian inference.
///
/// This likelihood assumes that residuals are Laplace distributed with location
/// parameter 0 and scale parameter `b`. The Laplace distribution is useful when
/// residuals are expected to have heavier tails than a normal distribution,
/// making it more robust to outliers.
///
/// # Mathematical Form
///
/// For residuals r_i, the log-likelihood is:
/// ```text
/// log L = -N * log(2b) - (1/b) * Σ|r_i|
/// ```
///
/// Where N is the total number of residuals.
///
/// # Gradient Form
///
/// The gradient with respect to model parameters θ is:
/// ```text
/// ∂L/∂θ = -1/b * Σ(sign(r_i) * ∂r_i/∂θ)
/// ```
///
/// # When to Use
///
/// - Data contains outliers that should not dominate the fit
/// - Errors are expected to have heavier tails than normal distribution
/// - Robust regression is desired
/// - L1-norm penalty is preferred over L2-norm
///
/// # Comparison with Normal Likelihood
///
/// - **Robustness**: Laplace is more robust to outliers
/// - **Tail behavior**: Laplace has exponentially decaying tails vs. Gaussian's quadratic decay
/// - **Computational**: Laplace involves absolute values, Normal involves squares
///
/// # Parameters
///
/// - `b`: Scale parameter controlling the width of the Laplace distribution
#[derive(Debug, Clone, Copy)]
pub struct LaplaceLikelihood {
    /// Scale parameter of the Laplace distribution
    pub b: f64,
}

impl LaplaceLikelihood {
    /// Creates a new Laplace likelihood with the specified scale parameter.
    ///
    /// # Arguments
    ///
    /// * `b` - Scale parameter (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `b` is not positive.
    pub fn new(b: f64) -> Self {
        assert!(b > 0.0, "Scale parameter must be positive");
        Self { b }
    }
}

impl Likelihood for LaplaceLikelihood {
    /// Computes the log-likelihood using dynamic parameters.
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals
    /// * `parameters` - Slice containing the scale parameter as the first element
    ///
    /// # Returns
    ///
    /// The total log-likelihood summed over all residuals.
    ///
    /// # Panics
    ///
    /// May panic if the parameters slice is empty or if the first parameter
    /// is not positive.
    #[inline(always)]
    fn log_likelihood(&self, residuals: &ndarray::Array2<f64>) -> f64 {
        // Use the first parameter as the scale parameter
        let b = self.b;
        let log_norm_const = (2.0 * b).ln();
        let log_likelihood = -residuals.mapv(|x| x.abs()) / b - log_norm_const;
        log_likelihood.sum()
    }

    /// Computes the gradient of the log-likelihood with respect to model parameters using vectorized operations.
    ///
    /// For Laplace likelihood: ∂L/∂θ = -1/b * Σ(sign(r_i) * ∂r_i/∂θ)
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals with shape (n_timepoints, n_species)
    /// * `sensitivities` - 3D array of sensitivities with shape (n_timepoints, n_species, n_parameters)
    ///
    /// # Returns
    ///
    /// 1D array of gradients with shape (n_parameters,)
    ///
    /// # Performance
    ///
    /// This implementation uses vectorized ndarray operations for maximum performance,
    /// avoiding nested loops and leveraging SIMD instructions.
    ///
    /// # Non-differentiable Points
    ///
    /// At r = 0, the Laplace distribution is not differentiable. We use the subgradient
    /// approach and choose 0 as the most conservative choice.
    #[inline(always)]
    fn log_likelihood_gradient(
        &self,
        residuals: &Array2<f64>,
        sensitivities: &Array3<f64>,
    ) -> Array1<f64> {
        let b = self.b;

        // Use vectorized chain rule: ∂L/∂r = -sign(r)/b with handling for r = 0
        apply_chain_rule(residuals, sensitivities, |r| {
            r.mapv(|x| if x == 0.0 { 0.0 } else { -x.signum() / b })
        })
    }
}

impl ObjectiveFunction for LaplaceLikelihood {
    /// Computes the log-likelihood for Laplace distributed residuals using the fixed scale parameter.
    ///
    /// This implementation uses the `b` value stored in the struct, making it
    /// suitable for optimization contexts where the likelihood parameters are fixed.
    ///
    /// # Arguments
    ///
    /// * `residuals` - 2D array of residuals (model predictions - observations)
    /// * `_` - Number of data points (unused in this implementation)
    ///
    /// # Returns
    ///
    /// The total log-likelihood summed over all residuals.
    ///
    /// # Mathematical Details
    ///
    /// For each residual r_ij, computes:
    /// - Absolute residual: |r_ij|
    /// - Log normalization constant: log(2b)
    /// - Log probability density: -|r_ij|/b - log(2b)
    ///
    /// Returns the sum of all log probability densities.
    #[inline(always)]
    fn cost(
        &self,
        residuals: &ndarray::Array2<f64>,
        _: usize,
    ) -> Result<f64, crate::objective::error::ObjectiveError> {
        let log_norm_const = (2.0 * self.b).ln();
        let log_likelihood = -residuals.mapv(|x| x.abs()) / self.b - log_norm_const;
        Ok(log_likelihood.sum())
    }

    fn gradient(
        &self,
        _: Array2<f64>,
        _: &ndarray::Array3<f64>,
        _: usize,
    ) -> Result<ndarray::Array1<f64>, ObjectiveError> {
        todo!()
    }
}

impl fmt::Display for LaplaceLikelihood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Laplace(b={0})", self.b)
    }
}

/// Enumeration of supported likelihood functions for Bayesian inference.
///
/// This enum provides a unified interface for working with different likelihood functions
/// in MCMC sampling and Bayesian parameter estimation. Each variant encapsulates the
/// parameters needed to fully specify a likelihood function and supports both
/// programmatic construction and string parsing.
///
/// # Design Principles
///
/// - **Type Safety**: Each likelihood variant has its own parameter structure
/// - **Validation**: All parameters are validated upon construction
/// - **Parsing**: Human-readable string format with error handling
/// - **Extensibility**: New likelihood functions can be added as additional variants
/// - **Integration**: Seamless conversion to concrete likelihood types
///
/// # Usage Patterns
///
/// This enum is commonly used in three scenarios:
///
/// 1. **Configuration files**: Users specify likelihood functions as strings that get parsed
/// 2. **Programmatic construction**: Direct instantiation with validated parameters
/// 3. **MCMC setup**: Converting to concrete likelihood types for sampling
///
/// # Parameter Conventions
///
/// All likelihood functions follow consistent parameter naming conventions:
/// - `sigma`: Standard deviation parameter for normal distributions
/// - `b`: Scale parameter for Laplace distributions
/// - All scale parameters must be strictly positive
///
/// # Supported Likelihood Functions
///
/// - **Normal**: Gaussian likelihood for symmetric, bell-shaped error distributions
/// - **Laplace**: Double exponential likelihood for robust handling of outliers
#[derive(Debug, Clone, Copy, Variantly)]
pub enum LikelihoodFunction {
    /// Normal (Gaussian) likelihood function for symmetric error distributions.
    ///
    /// This likelihood assumes that residuals follow a normal distribution with mean 0
    /// and standard deviation σ. It's the most common choice when measurement errors
    /// are expected to be symmetric and bell-shaped around the true values.
    ///
    /// # Mathematical Properties
    /// - Log-likelihood: -N/2 * log(2π) - N * log(σ) - (1/2σ²) * Σ(r²)
    /// - Gradient: ∂L/∂θ = -1/σ² * Σ(r * ∂r/∂θ)
    /// - Support: All real numbers
    /// - Shape: Symmetric, bell-shaped distribution
    ///
    /// # When to Use
    /// - Measurement errors are symmetric around zero
    /// - No significant outliers are expected
    /// - Classical least-squares assumptions hold
    /// - Fast convergence is desired
    ///
    /// # Parameters
    /// - `sigma`: Standard deviation parameter (must be positive)
    ///
    /// # Statistical Properties
    /// - Mean = 0 (by construction for residuals)
    /// - Variance = σ²
    /// - 68% of data within ±1σ, 95% within ±2σ
    Normal(NormalLikelihood),

    /// Laplace (double exponential) likelihood function for robust error handling.
    ///
    /// This likelihood assumes that residuals follow a Laplace distribution with location
    /// parameter 0 and scale parameter b. It provides better robustness to outliers
    /// compared to the normal likelihood due to its heavier tails.
    ///
    /// # Mathematical Properties
    /// - Log-likelihood: -N * log(2b) - (1/b) * Σ|r|
    /// - Gradient: ∂L/∂θ = -1/b * Σ(sign(r) * ∂r/∂θ)
    /// - Support: All real numbers
    /// - Shape: Symmetric, exponentially decaying tails
    ///
    /// # When to Use
    /// - Data contains outliers that should not dominate the fit
    /// - Robust regression is desired
    /// - L1-norm penalty is preferred over L2-norm
    /// - Heavy-tailed error distributions are expected
    ///
    /// # Parameters
    /// - `b`: Scale parameter (must be positive)
    ///
    /// # Statistical Properties
    /// - Mean = 0 (by construction for residuals)
    /// - Variance = 2b²
    /// - Heavier tails than normal distribution
    /// - More robust to outliers
    Laplace(LaplaceLikelihood),
}

/// Internal representation of a parsed likelihood specification.
///
/// This struct serves as an intermediate representation during string parsing,
/// separating the likelihood function name from its parameters before validation
/// and construction of the final `LikelihoodFunction` enum variant.
///
/// # Fields
/// - `name`: Lowercase likelihood name (e.g., "normal", "n", "laplace", "l")
/// - `params`: Vector of parsed numeric parameters in the order they appeared
///
/// # Design
/// This struct is not exposed in the public API and is only used internally
/// by the parsing pipeline to maintain separation of concerns between parsing
/// and validation logic.
#[derive(Debug)]
struct LikelihoodSpec {
    name: String,
    params: Vec<f64>,
}

impl LikelihoodFunction {
    pub const AVAILABLE_LIKELIHOODS: [&str; 2] = ["Normal", "Laplace"];

    /// Parse a likelihood specification string into structured components.
    ///
    /// This is the primary parsing method that handles the regex-based extraction
    /// of likelihood function names and parameters from user input strings. It supports
    /// a flexible parentheses-based format with case-insensitive function names.
    ///
    /// # Format Support
    /// - **Case insensitive**: `Normal`, `NORMAL`, `normal` all work
    /// - **Short names**: `N`, `L` for common likelihood functions
    /// - **Flexible whitespace**: Spaces around parentheses and commas are ignored
    /// - **Scientific notation**: Parameters can use `1e-3`, `2.5E+4` notation
    ///
    /// # Parsing Pipeline
    /// 1. Trim input and extract likelihood name and parameter string
    /// 2. Convert likelihood name to lowercase for consistency
    /// 3. Parse comma-separated parameters as floating-point numbers
    /// 4. Return structured `LikelihoodSpec` for further validation
    ///
    /// # Arguments
    /// * `s` - Input string in format `LikelihoodName(param1, param2, ...)`
    ///
    /// # Returns
    /// * `Ok(LikelihoodSpec)` - Successfully parsed specification
    /// * `Err(String)` - Descriptive error message for invalid input
    ///
    /// # Error Conditions
    /// - Invalid syntax (missing parentheses, malformed structure)
    /// - Unparseable numeric parameters
    /// - Empty likelihood name
    fn parse_likelihood(s: &str) -> Result<LikelihoodSpec, String> {
        let s = s.trim();

        // Parse parentheses format: LikelihoodName(param1, param2, ...)
        let paren_regex = Regex::new(r"^([a-zA-Z]+)\s*\(\s*([^)]*)\s*\)$")
            .map_err(|e| format!("Regex compilation error: {e}"))?;

        if let Some(captures) = paren_regex.captures(s) {
            let name = captures[1].to_lowercase();
            let params_str = &captures[2];

            let params = if params_str.trim().is_empty() {
                Vec::new()
            } else {
                Self::parse_parameters(params_str)?
            };

            return Ok(LikelihoodSpec { name, params });
        }

        Err(format!(
            "Invalid format: '{s}'. Use 'LikelihoodName(param1, param2, ...)'"
        ))
    }

    /// Extract and parse numeric parameters from a comma-separated string.
    ///
    /// This helper method handles the conversion of parameter strings into
    /// floating-point numbers with comprehensive error handling. It supports
    /// various numeric formats including scientific notation and handles
    /// whitespace gracefully.
    ///
    /// # Behavior
    /// - Splits on commas and trims whitespace from each parameter
    /// - Parses each parameter as a 64-bit floating-point number
    /// - Supports standard notation (1.5), scientific notation (1e-3), and negative numbers
    /// - Preserves parameter order for position-dependent likelihood functions
    ///
    /// # Arguments
    /// * `params_str` - Comma-separated parameter string (e.g., "1.0" or "1e-3,2.5E+4")
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Successfully parsed parameters in order
    /// * `Err(String)` - Error message indicating which parameter failed to parse
    fn parse_parameters(params_str: &str) -> Result<Vec<f64>, String> {
        params_str
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse parameters '{params_str}': {e}"))
    }

    /// Construct a validated `LikelihoodFunction` instance from a parsed specification.
    ///
    /// This method performs the final stage of parsing by converting a structured
    /// `LikelihoodSpec` into a concrete `LikelihoodFunction` enum variant. It handles
    /// likelihood-specific parameter validation and ensures all constraints are satisfied.
    ///
    /// # Validation Logic
    /// - **Parameter count validation**: Each likelihood requires specific parameter counts
    /// - **Positivity constraints**: Scale parameters must be strictly positive
    /// - **Range validation**: Parameters must satisfy mathematical constraints
    ///
    /// # Supported Likelihood Functions
    /// - `normal`, `n`: Requires exactly 1 parameter (sigma > 0)
    /// - `laplace`, `l`: Requires exactly 1 parameter (b > 0)
    ///
    /// # Arguments
    /// * `spec` - Parsed likelihood specification with name and parameters
    ///
    /// # Returns
    /// * `Ok(LikelihoodFunction)` - Successfully validated and constructed likelihood
    /// * `Err(String)` - Detailed error message describing validation failure
    fn from_likelihood_spec(spec: LikelihoodSpec) -> Result<Self, String> {
        match spec.name.as_str() {
            "normal" | "n" => {
                Self::validate_param_count(&spec.name, &spec.params, 1)?;
                Self::validate_positive(spec.params[0], "sigma")?;
                Ok(Self::Normal(NormalLikelihood::new(spec.params[0])))
            }
            "laplace" | "l" => {
                Self::validate_param_count(&spec.name, &spec.params, 1)?;
                Self::validate_positive(spec.params[0], "b")?;
                Ok(Self::Laplace(LaplaceLikelihood::new(spec.params[0])))
            }
            _ => Err(format!("Unknown likelihood function: '{}'", spec.name)),
        }
    }

    /// Validate that the correct number of parameters was provided for a likelihood function.
    ///
    /// Different likelihood functions require different numbers of parameters
    /// to be fully specified. This validation ensures users provide exactly the
    /// required count and generates helpful error messages when they don't.
    ///
    /// # Parameter Requirements
    /// - Normal: 1 parameter (sigma)
    /// - Laplace: 1 parameter (b)
    ///
    /// # Arguments
    /// * `name` - Likelihood function name for error message generation
    /// * `params` - Slice of provided parameters to validate
    /// * `expected` - Expected number of parameters for this likelihood function
    ///
    /// # Returns
    /// * `Ok(())` - Parameter count is correct
    /// * `Err(String)` - Error message with expected vs actual count
    fn validate_param_count(name: &str, params: &[f64], expected: usize) -> Result<(), String> {
        if params.len() != expected {
            return Err(format!(
                "{} likelihood requires exactly {} parameter{}, got {}",
                name,
                expected,
                if expected == 1 { "" } else { "s" },
                params.len()
            ));
        }
        Ok(())
    }

    /// Validate that a parameter value satisfies positivity constraints.
    ///
    /// All likelihood function scale parameters must be strictly positive to be
    /// mathematically valid. This includes standard deviations and scale parameters.
    /// This validation ensures such constraints are enforced.
    ///
    /// # Mathematical Requirements
    /// - Standard deviations must be positive (σ > 0)
    /// - Scale parameters must be positive (b > 0)
    ///
    /// # Arguments
    /// * `value` - Parameter value to validate
    /// * `param_name` - Human-readable parameter name for error messages
    ///
    /// # Returns
    /// * `Ok(())` - Parameter value is strictly positive
    /// * `Err(String)` - Error message indicating the constraint violation
    fn validate_positive(value: f64, param_name: &str) -> Result<(), String> {
        if value <= 0.0 {
            return Err(format!("{param_name} must be positive"));
        }
        Ok(())
    }
}

impl Likelihood for LikelihoodFunction {
    fn log_likelihood(&self, residuals: &ndarray::Array2<f64>) -> f64 {
        match self {
            LikelihoodFunction::Normal(ll) => ll.log_likelihood(residuals),
            LikelihoodFunction::Laplace(ll) => ll.log_likelihood(residuals),
        }
    }

    fn log_likelihood_gradient(
        &self,
        residuals: &ndarray::Array2<f64>,
        sensitivities: &ndarray::Array3<f64>,
    ) -> ndarray::Array1<f64> {
        match self {
            LikelihoodFunction::Normal(ll) => ll.log_likelihood_gradient(residuals, sensitivities),
            LikelihoodFunction::Laplace(ll) => ll.log_likelihood_gradient(residuals, sensitivities),
        }
    }
}

impl ObjectiveFunction for LikelihoodFunction {
    fn cost(
        &self,
        residuals: &ndarray::Array2<f64>,
        _: usize,
    ) -> Result<f64, crate::objective::error::ObjectiveError> {
        Ok(self.log_likelihood(residuals))
    }

    fn gradient(
        &self,
        _: ndarray::Array2<f64>,
        _: &ndarray::Array3<f64>,
        _: usize,
    ) -> Result<ndarray::Array1<f64>, crate::objective::error::ObjectiveError> {
        todo!()
    }
}

impl fmt::Display for LikelihoodFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LikelihoodFunction::Normal(ll) => write!(f, "Normal({})", ll.sigma),
            LikelihoodFunction::Laplace(ll) => write!(f, "Laplace({})", ll.b),
        }
    }
}

/// Implementation of string parsing for `LikelihoodFunction` specifications.
///
/// This implementation enables the convenient `.parse()` method for converting
/// strings into `LikelihoodFunction` instances. It coordinates the entire parsing
/// pipeline from raw string input to validated likelihood function objects.
///
/// # Parsing Pipeline
/// 1. **Syntax parsing**: Extract likelihood name and parameters using regex
/// 2. **Parameter parsing**: Convert parameter strings to floating-point numbers
/// 3. **Validation**: Ensure parameter constraints are satisfied
/// 4. **Construction**: Create the appropriate `LikelihoodFunction` enum variant
///
/// # Error Handling
/// Errors can occur at any stage of parsing and are propagated with descriptive
/// messages to help users correct their input. Common error categories include:
/// - Syntax errors (malformed input format)
/// - Type errors (non-numeric parameters)
/// - Validation errors (constraint violations)
/// - Specification errors (unknown likelihood function names)
///
/// # Usage
/// The `FromStr` trait enables multiple convenient ways to parse likelihood functions:
/// - Direct parsing: `"Normal(1.0)".parse::<LikelihoodFunction>()`
/// - Turbofish syntax: `s.parse::<LikelihoodFunction>()`
/// - Type annotation: `let likelihood: LikelihoodFunction = s.parse()?`
impl std::str::FromStr for LikelihoodFunction {
    type Err = String;

    /// Parse a string representation into a `LikelihoodFunction`.
    ///
    /// This method orchestrates the complete parsing process from string input
    /// to validated `LikelihoodFunction` instance. It combines syntax parsing,
    /// parameter extraction, and validation into a single convenient interface.
    ///
    /// # Arguments
    /// * `s` - String representation of the likelihood function
    ///
    /// # Returns
    /// * `Ok(LikelihoodFunction)` - Successfully parsed and validated likelihood function
    /// * `Err(String)` - Detailed error message describing the parsing failure
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let spec = Self::parse_likelihood(s)?;
        Self::from_likelihood_spec(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_likelihood_parentheses_format() {
        let likelihood: LikelihoodFunction = "Normal(1.0)".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "N(2.5)".parse().unwrap();
        assert!(likelihood.is_normal());
    }

    #[test]
    fn test_laplace_likelihood_parentheses_format() {
        let likelihood: LikelihoodFunction = "Laplace(0.5)".parse().unwrap();
        assert!(likelihood.is_laplace());

        let likelihood: LikelihoodFunction = "L(1.5)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_whitespace_handling() {
        let likelihood: LikelihoodFunction = "  Normal  (  1.0  )  ".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "Laplace( 2.0 )".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_case_insensitive() {
        let likelihood: LikelihoodFunction = "NORMAL(1.0)".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "lApLaCe(0.5)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_invalid_likelihood_name() {
        let result: Result<LikelihoodFunction, _> = "InvalidLikelihood(1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown likelihood function"));
    }

    #[test]
    fn test_invalid_format() {
        let result: Result<LikelihoodFunction, _> = "Normal[1.0]".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));

        let result: Result<LikelihoodFunction, _> = "normal:1.0".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));
    }

    #[test]
    fn test_missing_closing_parenthesis() {
        let result: Result<LikelihoodFunction, _> = "Normal(1.0".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));
    }

    #[test]
    fn test_invalid_parameter_count() {
        // Normal requires 1 parameter
        let result: Result<LikelihoodFunction, _> = "Normal()".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 1"));

        let result: Result<LikelihoodFunction, _> = "Normal(1.0, 2.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 1"));

        // Laplace requires 1 parameter
        let result: Result<LikelihoodFunction, _> = "Laplace(1.0, 2.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 1"));
    }

    #[test]
    fn test_invalid_parameter_values() {
        // Negative sigma
        let result: Result<LikelihoodFunction, _> = "Normal(-1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Zero sigma
        let result: Result<LikelihoodFunction, _> = "Normal(0.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Negative b
        let result: Result<LikelihoodFunction, _> = "Laplace(-0.5)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Zero b
        let result: Result<LikelihoodFunction, _> = "Laplace(0.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));
    }

    #[test]
    fn test_invalid_number_format() {
        let result: Result<LikelihoodFunction, _> = "Normal(abc)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));

        let result: Result<LikelihoodFunction, _> = "Laplace(xyz)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));
    }

    #[test]
    fn test_scientific_notation() {
        let likelihood: LikelihoodFunction = "Normal(1e-3)".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "Laplace(2.5E+1)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_decimal_numbers() {
        let likelihood: LikelihoodFunction = "Normal(0.123)".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "Laplace(3.14159)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_short_names() {
        // Test all short name variants
        let likelihood: LikelihoodFunction = "N(1.0)".parse().unwrap();
        assert!(likelihood.is_normal());

        let likelihood: LikelihoodFunction = "L(0.5)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }

    #[test]
    fn test_edge_cases() {
        // Very small positive values
        let likelihood: LikelihoodFunction = "Normal(1e-10)".parse().unwrap();
        assert!(likelihood.is_normal());

        // Very large values
        let likelihood: LikelihoodFunction = "Laplace(1e6)".parse().unwrap();
        assert!(likelihood.is_laplace());
    }
}
