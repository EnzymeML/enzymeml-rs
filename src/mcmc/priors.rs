//! Prior distributions for Bayesian inference in MCMC sampling.
//!
//! This module provides a comprehensive framework for working with prior distributions in
//! Bayesian parameter estimation. It includes the `DiffablePrior` trait for probability
//! distributions with gradient computation capabilities, and the `Prior` enum for representing
//! common distribution types with string parsing support.
//!
//! # Overview
//!
//! The module supports two main use cases:
//!
//! 1. **Gradient-based MCMC sampling**: The `DiffablePrior` trait extends standard probability
//!    distributions with log-probability density gradient computation, enabling efficient
//!    sampling with algorithms like Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampling (NUTS).
//!
//! 2. **Flexible prior specification**: The `Prior` enum allows users to specify distributions
//!    using human-readable string formats with automatic parsing and validation.
//!
//! # Supported Distributions
//!
//! Currently supports four common probability distributions:
//!
//! - **Normal**: Gaussian distribution with mean and standard deviation parameters
//! - **Uniform**: Uniform distribution over a specified interval
//! - **Exponential**: Exponential distribution with rate parameter (memoryless property)
//! - **Log-Normal**: Log-normal distribution for positive-valued parameters
//!
//! # String Format
//!
//! Distributions can be specified using a parentheses format with case-insensitive names.
//! Both full names and abbreviated forms are supported:
//!
//! - Normal distributions: `Normal(mean, std)` or `N(mean, std)`
//! - Uniform distributions: `Uniform(min, max)` or `U(min, max)`
//! - Exponential distributions: `Exponential(rate)`, `Exp(rate)`, or `E(rate)`
//! - Log-normal distributions: `LogNormal(mean, std)`, `LogN(mean, std)`, or `LN(mean, std)`
//!
//! The parser handles whitespace gracefully and supports scientific notation for parameters.
//!
//! # Parameter Validation
//!
//! All distributions include comprehensive parameter validation:
//!
//! - **Positive constraints**: Standard deviations and rates must be positive
//! - **Range constraints**: Uniform distribution bounds must satisfy min < max
//! - **Parameter counts**: Each distribution requires the correct number of parameters
//!
//! # Thread Safety
//!
//! All types in this module implement `Send + Sync`, making them suitable for parallel
//! MCMC sampling across multiple chains.

use std::str::FromStr;

use rand::distributions::Distribution;
use rand::thread_rng;
use regex::Regex;
use statrs::{
    distribution::{Continuous, Exp, LogNormal, Normal, Uniform},
    statistics::Distribution as _,
};

/// A trait for prior distributions with differentiable log probability density functions.
///
/// This trait extends standard probability distributions with gradient computation capabilities
/// required for gradient-based MCMC samplers. The gradient of the log-pdf guides the sampler
/// through the parameter space more efficiently than random walk methods.
///
/// # Type Requirements
///
/// - `Continuous<f64, f64>`: Continuous probability distribution over real numbers
/// - `Distribution<f64>`: Sampling capability
/// - `Clone + Send + Sync`: Thread safety for parallel MCMC chains
pub trait DiffablePrior: Continuous<f64, f64> + Clone + Send + Sync {
    /// Computes the gradient of the log probability density function.
    ///
    /// Returns the derivative ∇ ln p(x) = d/dx ln p(x) where p(x) is the probability
    /// density function evaluated at point x.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the gradient
    ///
    /// # Returns
    ///
    /// The gradient value d/dx ln p(x)
    fn ln_pdf_grad(&self, x: f64) -> f64;

    /// Draws a single sample from the distribution.
    ///
    /// # Returns
    ///
    /// A random sample from the distribution
    fn draw_sample(&self) -> f64
    where
        Self: Distribution<f64>,
    {
        let mut rng = thread_rng();
        self.sample(&mut rng)
    }
}

/// Implementation of `DiffablePrior` for the Normal distribution.
///
/// For a normal distribution N(μ, σ²), the gradient of the log-pdf is:
/// ∇ ln p(x) = -(x - μ) / σ²
///
/// This provides the derivative information needed for gradient-based MCMC sampling.
impl DiffablePrior for Normal {
    /// Computes the gradient of the log-pdf for a normal distribution.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the gradient
    ///
    /// # Returns
    ///
    /// The gradient -(x - μ) / σ² where μ is the mean and σ² is the variance
    fn ln_pdf_grad(&self, x: f64) -> f64 {
        -(x - self.mean().unwrap()) / self.variance().unwrap()
    }
}

/// Implementation of `DiffablePrior` for the Uniform distribution.
///
/// For a uniform distribution U(a, b), the log-pdf is constant within the support
/// and undefined outside. The gradient is zero everywhere within the support.
///
/// This makes uniform priors "non-informative" in the sense that they don't
/// provide gradient information to guide the sampler.
impl DiffablePrior for Uniform {
    /// Computes the gradient of the log-pdf for a uniform distribution.
    ///
    /// # Arguments
    ///
    /// * `_` - The point at which to evaluate the gradient (unused)
    ///
    /// # Returns
    ///
    /// Always returns 0.0 since the log-pdf is constant within the support
    fn ln_pdf_grad(&self, _: f64) -> f64 {
        0.0
    }
}

/// Implementation of `DiffablePrior` for the Exponential distribution.
///
/// For an exponential distribution Exp(λ), the gradient of the log-pdf is:
/// ∇ ln p(x) = -λ
///
/// This is constant and independent of x, reflecting the memoryless property
/// of the exponential distribution.
impl DiffablePrior for Exp {
    /// Computes the gradient of the log-pdf for an exponential distribution.
    ///
    /// # Arguments
    ///
    /// * `_` - The point at which to evaluate the gradient (unused)
    ///
    /// # Returns
    ///
    /// The negative rate parameter -λ
    fn ln_pdf_grad(&self, _: f64) -> f64 {
        -self.rate()
    }
}

/// Implementation of `DiffablePrior` for the Log-Normal distribution.
///
/// For a log-normal distribution LogN(μ, σ²), the gradient of the log-pdf is:
/// ∇ ln p(x) = -(ln(x) - μ) / (σ² * x)
///
/// Note: This implementation appears to use an incorrect formula and should be
/// reviewed. The correct gradient should involve ln(x) and division by x.
impl DiffablePrior for LogNormal {
    /// Computes the gradient of the log-pdf for a log-normal distribution.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the gradient
    ///
    /// # Returns
    ///
    /// The gradient value (currently using an incorrect formula)
    ///
    /// # Note
    ///
    /// This implementation may be incorrect and should be verified against
    /// the proper log-normal distribution gradient formula.
    fn ln_pdf_grad(&self, x: f64) -> f64 {
        -(x - self.mean().unwrap()) / self.variance().unwrap()
    }
}

/// Enumeration of supported prior distributions for Bayesian inference.
///
/// This enum provides a unified interface for working with different probability distributions
/// as priors in MCMC sampling. Each variant contains the actual distribution type from the
/// `statrs` crate, providing direct access to all distribution methods and properties.
///
/// # Design Principles
///
/// - **Type Safety**: Each variant contains the actual distribution type
/// - **Validation**: All parameters are validated during distribution construction
/// - **Parsing**: Human-readable string format with error handling
/// - **Direct Access**: Full access to `statrs` distribution functionality
/// - **Extensibility**: New distributions can be added as additional variants
///
/// # Usage Patterns
///
/// This enum is commonly used in three scenarios:
///
/// 1. **Configuration files**: Users specify priors as strings that get parsed
/// 2. **Programmatic construction**: Direct instantiation with validated parameters
/// 3. **MCMC setup**: Direct use of contained distribution types for sampling
///
/// # Distribution Access
///
/// Since each variant contains the actual distribution type, you can directly access
/// all methods provided by the `statrs` crate, including:
/// - Probability density functions
/// - Cumulative distribution functions
/// - Random sampling
/// - Statistical moments
/// - Parameter accessors
///
/// # Parameter Conventions
///
/// All distributions follow the parameter conventions from the `statrs` crate:
/// - Normal: mean and standard deviation
/// - Uniform: minimum and maximum bounds  
/// - Exponential: rate parameter
/// - LogNormal: underlying normal distribution parameters
#[derive(Debug, Clone, PartialEq)]
pub enum Prior {
    /// Normal (Gaussian) distribution from the `statrs` crate.
    ///
    /// The normal distribution is the most common choice for priors on unbounded
    /// real-valued parameters. It provides a symmetric, bell-shaped distribution
    /// centered at the mean with spread controlled by the standard deviation.
    ///
    /// # Properties
    /// - Support: (-∞, +∞)
    /// - Symmetric around the mean
    /// - 68% of mass within ±1 standard deviation
    /// - Suitable for parameters that can take any real value
    ///
    /// # Access
    /// The contained `Normal` distribution provides methods like:
    /// - `mean()`: Get the mean parameter
    /// - `std_dev()`: Get the standard deviation
    /// - `pdf(x)`: Probability density at x
    /// - `cdf(x)`: Cumulative probability at x
    Normal(Normal),

    /// Uniform distribution from the `statrs` crate.
    ///
    /// The uniform distribution assigns equal probability density to all values
    /// within a specified range. This represents complete ignorance about the
    /// parameter value within the given bounds, making it a non-informative prior.
    ///
    /// # Properties
    /// - Support: [min, max)
    /// - Constant probability density within bounds
    /// - Zero probability outside bounds
    /// - Requires min < max
    ///
    /// # Access
    /// The contained `Uniform` distribution provides methods like:
    /// - `min()`: Get the lower bound
    /// - `max()`: Get the upper bound
    /// - `pdf(x)`: Probability density at x
    /// - `cdf(x)`: Cumulative probability at x
    Uniform(Uniform),

    /// Exponential distribution from the `statrs` crate.
    ///
    /// The exponential distribution is commonly used for priors on positive-valued
    /// parameters, especially those representing rates, scales, or waiting times.
    /// It has the memoryless property and decreases monotonically from its mode at zero.
    ///
    /// # Properties
    /// - Support: [0, +∞)
    /// - Mean = 1/rate
    /// - Mode at 0, monotonically decreasing
    /// - Memoryless property: P(X > s+t | X > s) = P(X > t)
    ///
    /// # Access
    /// The contained `Exp` distribution provides methods like:
    /// - `rate()`: Get the rate parameter λ
    /// - `pdf(x)`: Probability density at x
    /// - `cdf(x)`: Cumulative probability at x
    Exp(Exp),

    /// Log-normal distribution from the `statrs` crate.
    ///
    /// The log-normal distribution is used for positive-valued parameters where
    /// the logarithm of the parameter follows a normal distribution. This is
    /// particularly useful for scale parameters or quantities that vary over
    /// several orders of magnitude.
    ///
    /// # Properties
    /// - Support: (0, +∞)
    /// - Right-skewed with long tail
    /// - Mode at exp(location - scale²)
    /// - Geometric mean = exp(location)
    ///
    /// # Access
    /// The contained `LogNormal` distribution provides methods like:
    /// - `location()`: Get the location parameter (underlying normal mean)
    /// - `scale()`: Get the scale parameter (underlying normal std dev)
    /// - `pdf(x)`: Probability density at x
    /// - `cdf(x)`: Cumulative probability at x
    ///
    /// # Note
    /// The parameters refer to the underlying normal distribution, not the
    /// log-normal distribution itself. The actual mean of the log-normal
    /// distribution is exp(location + scale²/2).
    LogNormal(LogNormal),
}

/// Internal representation of a parsed distribution specification.
///
/// This struct serves as an intermediate representation during string parsing,
/// separating the distribution name from its parameters before validation and
/// construction of the final `Prior` enum variant.
///
/// # Fields
/// - `name`: Lowercase distribution name (e.g., "normal", "n", "uniform", "u")
/// - `params`: Vector of parsed numeric parameters in the order they appeared
///
/// # Design
/// This struct is not exposed in the public API and is only used internally
/// by the parsing pipeline to maintain separation of concerns between parsing
/// and validation logic.
#[derive(Debug)]
struct DistSpec {
    name: String,
    params: Vec<f64>,
}

impl Prior {
    pub const AVAILABLE_PRIORS: [&str; 4] = ["normal", "uniform", "exp", "lognormal"];

    /// Parse a distribution specification string into structured components.
    ///
    /// This is the primary parsing method that handles the regex-based extraction
    /// of distribution names and parameters from user input strings. It supports
    /// a flexible parentheses-based format with case-insensitive distribution names.
    ///
    /// # Format Support
    /// - **Case insensitive**: `Normal`, `NORMAL`, `normal` all work
    /// - **Short names**: `N`, `U`, `E`, `LN` for common distributions
    /// - **Flexible whitespace**: Spaces around parentheses and commas are ignored
    /// - **Scientific notation**: Parameters can use `1e-3`, `2.5E+4` notation
    ///
    /// # Parsing Pipeline
    /// 1. Trim input and extract distribution name and parameter string
    /// 2. Convert distribution name to lowercase for consistency
    /// 3. Parse comma-separated parameters as floating-point numbers
    /// 4. Return structured `DistSpec` for further validation
    ///
    /// # Arguments
    /// * `s` - Input string in format `DistributionName(param1, param2, ...)`
    ///
    /// # Returns
    /// * `Ok(DistSpec)` - Successfully parsed specification
    /// * `Err(String)` - Descriptive error message for invalid input
    ///
    /// # Error Conditions
    /// - Invalid syntax (missing parentheses, malformed structure)
    /// - Unparseable numeric parameters
    /// - Empty distribution name
    fn parse_distribution(s: &str) -> Result<DistSpec, String> {
        let s = s.trim();

        // Parse parentheses format: DistName(param1, param2, ...)
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

            return Ok(DistSpec { name, params });
        }

        Err(format!(
            "Invalid format: '{s}'. Use 'DistributionName(param1, param2, ...)'"
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
    /// - Preserves parameter order for position-dependent distributions
    ///
    /// # Arguments
    /// * `params_str` - Comma-separated parameter string (e.g., "0.0, 1.0" or "1e-3,2.5E+4")
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

    /// Construct a validated `Prior` instance from a parsed distribution specification.
    ///
    /// This method performs the final stage of parsing by converting a structured
    /// `DistSpec` into a concrete `Prior` enum variant. It handles distribution-specific
    /// parameter validation and ensures all constraints are satisfied before construction.
    ///
    /// # Validation Logic
    /// - **Parameter count validation**: Each distribution requires specific parameter counts
    /// - **Range validation**: Parameters must satisfy mathematical constraints
    /// - **Positivity constraints**: Standard deviations and rates must be positive
    /// - **Ordering constraints**: Uniform bounds must satisfy min < max
    ///
    /// # Supported Distributions
    /// - `normal`, `n`: Requires exactly 2 parameters (mean, std > 0)
    /// - `uniform`, `u`: Requires exactly 2 parameters (min < max)
    /// - `exponential`, `exp`, `e`: Requires exactly 1 parameter (rate > 0)
    /// - `lognormal`, `logn`, `ln`: Requires exactly 2 parameters (mean, std > 0)
    ///
    /// # Arguments
    /// * `spec` - Parsed distribution specification with name and parameters
    ///
    /// # Returns
    /// * `Ok(Prior)` - Successfully validated and constructed distribution
    /// * `Err(String)` - Detailed error message describing validation failure
    fn from_dist_spec(spec: DistSpec) -> Result<Self, String> {
        match spec.name.as_str() {
            "normal" | "n" => {
                Self::validate_param_count(&spec.name, &spec.params, 2)?;
                Self::validate_positive(spec.params[1], "standard deviation")?;
                let normal = Normal::new(spec.params[0], spec.params[1])
                    .map_err(|e| format!("Failed to create Normal distribution: {e}"))?;
                Ok(Self::Normal(normal))
            }
            "uniform" | "u" => {
                Self::validate_param_count(&spec.name, &spec.params, 2)?;
                if spec.params[0] >= spec.params[1] {
                    return Err(
                        "Uniform distribution minimum must be less than maximum".to_string()
                    );
                }
                let uniform = Uniform::new(spec.params[0], spec.params[1])
                    .map_err(|e| format!("Failed to create Uniform distribution: {e}"))?;
                Ok(Self::Uniform(uniform))
            }
            "exponential" | "exp" | "e" => {
                Self::validate_param_count(&spec.name, &spec.params, 1)?;
                Self::validate_positive(spec.params[0], "rate")?;
                let exp = Exp::new(spec.params[0])
                    .map_err(|e| format!("Failed to create Exponential distribution: {e}"))?;
                Ok(Self::Exp(exp))
            }
            "lognormal" | "logn" | "ln" => {
                Self::validate_param_count(&spec.name, &spec.params, 2)?;
                Self::validate_positive(spec.params[1], "standard deviation")?;
                let lognormal = LogNormal::new(spec.params[0], spec.params[1])
                    .map_err(|e| format!("Failed to create LogNormal distribution: {e}"))?;
                Ok(Self::LogNormal(lognormal))
            }
            _ => Err(format!("Unknown distribution type: '{}'", spec.name)),
        }
    }

    /// Validate that the correct number of parameters was provided for a distribution.
    ///
    /// Different probability distributions require different numbers of parameters
    /// to be fully specified. This validation ensures users provide exactly the
    /// required count and generates helpful error messages when they don't.
    ///
    /// # Parameter Requirements
    /// - Normal/Log-normal: 2 parameters (location, scale)
    /// - Uniform: 2 parameters (lower bound, upper bound)  
    /// - Exponential: 1 parameter (rate)
    ///
    /// # Arguments
    /// * `name` - Distribution name for error message generation
    /// * `params` - Slice of provided parameters to validate
    /// * `expected` - Expected number of parameters for this distribution
    ///
    /// # Returns
    /// * `Ok(())` - Parameter count is correct
    /// * `Err(String)` - Error message with expected vs actual count
    fn validate_param_count(name: &str, params: &[f64], expected: usize) -> Result<(), String> {
        if params.len() != expected {
            return Err(format!(
                "{} distribution requires exactly {} parameter{}, got {}",
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
    /// Many probability distribution parameters must be strictly positive to be
    /// mathematically valid. This includes standard deviations, variance parameters,
    /// and rate parameters. This validation ensures such constraints are enforced.
    ///
    /// # Mathematical Requirements
    /// - Standard deviations must be positive (σ > 0)
    /// - Rate parameters must be positive (λ > 0)
    /// - Scale parameters must be positive
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

impl DiffablePrior for Prior {
    fn ln_pdf_grad(&self, x: f64) -> f64 {
        match self {
            Prior::Normal(dist) => dist.ln_pdf_grad(x),
            Prior::Uniform(dist) => dist.ln_pdf_grad(x),
            Prior::Exp(dist) => dist.ln_pdf_grad(x),
            Prior::LogNormal(dist) => dist.ln_pdf_grad(x),
        }
    }
}

impl Distribution<f64> for Prior {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Prior::Normal(dist) => dist.sample(rng),
            Prior::Uniform(dist) => dist.sample(rng),
            Prior::Exp(dist) => dist.sample(rng),
            Prior::LogNormal(dist) => dist.sample(rng),
        }
    }
}

impl Continuous<f64, f64> for Prior {
    /// Probability distribution function
    #[inline(always)]
    fn pdf(&self, x: f64) -> f64 {
        match self {
            Prior::Normal(normal) => normal.pdf(x),
            Prior::Uniform(uniform) => uniform.pdf(x),
            Prior::Exp(exp) => exp.pdf(x),
            Prior::LogNormal(log_normal) => log_normal.pdf(x),
        }
    }

    /// Log-Probability distribution function
    fn ln_pdf(&self, x: f64) -> f64 {
        match self {
            Prior::Normal(normal) => normal.ln_pdf(x),
            Prior::Uniform(uniform) => uniform.ln_pdf(x),
            Prior::Exp(exp) => exp.ln_pdf(x),
            Prior::LogNormal(log_normal) => log_normal.ln_pdf(x),
        }
    }
}

/// Implementation of string parsing for `Prior` distributions.
///
/// This implementation enables the convenient `.parse()` method for converting
/// strings into `Prior` instances. It coordinates the entire parsing pipeline
/// from raw string input to validated distribution objects.
///
/// # Parsing Pipeline
/// 1. **Syntax parsing**: Extract distribution name and parameters using regex
/// 2. **Parameter parsing**: Convert parameter strings to floating-point numbers
/// 3. **Validation**: Ensure parameter constraints are satisfied
/// 4. **Construction**: Create the appropriate `Prior` enum variant
///
/// # Error Handling
/// Errors can occur at any stage of parsing and are propagated with descriptive
/// messages to help users correct their input. Common error categories include:
/// - Syntax errors (malformed input format)
/// - Type errors (non-numeric parameters)
/// - Validation errors (constraint violations)
/// - Specification errors (unknown distribution names)
///
/// # Usage
/// The `FromStr` trait enables multiple convenient ways to parse distributions:
/// - Direct parsing: `"Normal(0, 1)".parse::<Prior>()`
/// - Turbofish syntax: `s.parse::<Prior>()`
/// - Type annotation: `let prior: Prior = s.parse()?`
impl FromStr for Prior {
    type Err = String;

    /// Parse a string representation into a `Prior` distribution.
    ///
    /// This method orchestrates the complete parsing process from string input
    /// to validated `Prior` instance. It combines syntax parsing, parameter
    /// extraction, and validation into a single convenient interface.
    ///
    /// # Arguments
    /// * `s` - String representation of the distribution
    ///
    /// # Returns
    /// * `Ok(Prior)` - Successfully parsed and validated distribution
    /// * `Err(String)` - Detailed error message describing the parsing failure
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let spec = Self::parse_distribution(s)?;
        Self::from_dist_spec(spec)
    }
}

#[cfg(test)]
mod tests {
    use statrs::statistics::{Max, Min};

    use super::*;

    #[test]
    fn test_normal_parentheses_format() {
        let prior: Prior = "Normal(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "N(5.5, 2.3)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 5.5);
                assert_eq!(normal.std_dev().unwrap(), 2.3);
            }
            _ => panic!("Expected Normal distribution"),
        }
    }

    #[test]
    fn test_uniform_parentheses_format() {
        let prior: Prior = "Uniform(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), 0.0);
                assert_eq!(uniform.max(), 1.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }

        let prior: Prior = "U(-5.0, 10.0)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), -5.0);
                assert_eq!(uniform.max(), 10.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }
    }

    #[test]
    fn test_exponential_parentheses_format() {
        let prior: Prior = "Exponential(1.5)".parse().unwrap();
        match prior {
            Prior::Exp(exp) => {
                assert_eq!(exp.rate(), 1.5);
            }
            _ => panic!("Expected Exponential distribution"),
        }

        let prior: Prior = "Exp(0.1)".parse().unwrap();
        match prior {
            Prior::Exp(exp) => {
                assert_eq!(exp.rate(), 0.1);
            }
            _ => panic!("Expected Exponential distribution"),
        }

        let prior: Prior = "E(2.0)".parse().unwrap();
        match prior {
            Prior::Exp(exp) => {
                assert_eq!(exp.rate(), 2.0);
            }
            _ => panic!("Expected Exponential distribution"),
        }
    }

    #[test]
    fn test_lognormal_parentheses_format() {
        let prior: Prior = "LogNormal(0.0, 1.0)".parse().unwrap();
        assert!(matches!(prior, Prior::LogNormal(_)));

        let prior: Prior = "LogN(1.5, 0.5)".parse().unwrap();
        assert!(matches!(prior, Prior::LogNormal(_)));

        let prior: Prior = "LN(2.0, 2.0)".parse().unwrap();
        assert!(matches!(prior, Prior::LogNormal(_)));
    }

    #[test]
    fn test_whitespace_handling() {
        let prior: Prior = "  Normal  (  0.0  ,  1.0  )  ".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "Uniform( -1.0 , 2.0 )".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), -1.0);
                assert_eq!(uniform.max(), 2.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }
    }

    #[test]
    fn test_case_insensitive() {
        let prior: Prior = "NORMAL(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "UnIfOrM(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), 0.0);
                assert_eq!(uniform.max(), 1.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }
    }

    #[test]
    fn test_invalid_distribution_name() {
        let result: Result<Prior, _> = "InvalidDist(1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown distribution type"));
    }

    #[test]
    fn test_invalid_format() {
        let result: Result<Prior, _> = "Normal[0.0, 1.0]".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));

        let result: Result<Prior, _> = "normal:0.0:1.0".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));
    }

    #[test]
    fn test_missing_closing_parenthesis() {
        let result: Result<Prior, _> = "Normal(0.0, 1.0".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid format"));
    }

    #[test]
    fn test_invalid_parameter_count() {
        // Normal requires 2 parameters
        let result: Result<Prior, _> = "Normal(1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 2"));

        let result: Result<Prior, _> = "Normal(1.0, 2.0, 3.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 2"));

        // Exponential requires 1 parameter
        let result: Result<Prior, _> = "Exp(1.0, 2.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 1"));
    }

    #[test]
    fn test_invalid_parameter_values() {
        // Negative standard deviation
        let result: Result<Prior, _> = "Normal(0.0, -1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Zero standard deviation
        let result: Result<Prior, _> = "LogNormal(0.0, 0.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Negative rate
        let result: Result<Prior, _> = "Exponential(-1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));

        // Invalid uniform bounds
        let result: Result<Prior, _> = "Uniform(1.0, 1.0)".parse();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("minimum must be less than maximum"));

        let result: Result<Prior, _> = "Uniform(2.0, 1.0)".parse();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("minimum must be less than maximum"));
    }

    #[test]
    fn test_invalid_number_format() {
        let result: Result<Prior, _> = "Normal(abc, 1.0)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));

        let result: Result<Prior, _> = "Uniform(1.0, xyz)".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));
    }

    #[test]
    fn test_empty_parameters() {
        let result: Result<Prior, _> = "Normal()".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires exactly 2"));
    }

    #[test]
    fn test_scientific_notation() {
        let prior: Prior = "Normal(1e-3, 2.5e2)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.001);
                assert_eq!(normal.std_dev().unwrap(), 250.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "Exponential(1.5E-2)".parse().unwrap();
        match prior {
            Prior::Exp(exp) => {
                assert_eq!(exp.rate(), 0.015);
            }
            _ => panic!("Expected Exponential distribution"),
        }
    }

    #[test]
    fn test_negative_numbers() {
        let prior: Prior = "Normal(-5.5, 1.0)".parse().unwrap();
        assert!(matches!(prior, Prior::Normal(_)));

        let prior: Prior = "Uniform(-10.0, -1.0)".parse().unwrap();
        assert!(matches!(prior, Prior::Uniform(_)));

        let prior: Prior = "LogNormal(-2.0, 0.5)".parse().unwrap();
        assert!(matches!(prior, Prior::LogNormal(_)));
    }

    #[test]
    fn test_short_names() {
        // Test all short name variants
        let prior: Prior = "N(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "U(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), 0.0);
                assert_eq!(uniform.max(), 1.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }

        let prior: Prior = "E(1.0)".parse().unwrap();
        match prior {
            Prior::Exp(exp) => {
                assert_eq!(exp.rate(), 1.0);
            }
            _ => panic!("Expected Exponential distribution"),
        }
    }

    #[test]
    fn test_edge_cases() {
        // Very small positive values
        let prior: Prior = "Normal(0.0, 1e-10)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1e-10);
            }
            _ => panic!("Expected Normal distribution"),
        }

        // Very large values
        let prior: Prior = "Uniform(-1e6, 1e6)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), -1e6);
                assert_eq!(uniform.max(), 1e6);
            }
            _ => panic!("Expected Uniform distribution"),
        }

        // Zero mean/min values (should be valid)
        let prior: Prior = "Normal(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Normal(normal) => {
                assert_eq!(normal.mean().unwrap(), 0.0);
                assert_eq!(normal.std_dev().unwrap(), 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        let prior: Prior = "Uniform(0.0, 1.0)".parse().unwrap();
        match prior {
            Prior::Uniform(uniform) => {
                assert_eq!(uniform.min(), 0.0);
                assert_eq!(uniform.max(), 1.0);
            }
            _ => panic!("Expected Uniform distribution"),
        }
    }
}
