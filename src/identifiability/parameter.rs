use std::str::FromStr;

use regex::Regex;

use crate::optim::OptimizeError;

/// Configuration for a parameter to be profiled.
///
/// Specifies the parameter name and the range of values to test during profiling.
/// Can be created using the builder pattern or parsed from a string.
///
/// # Notes
///
/// - The `name` field specifies the name of the parameter to profile.
/// - The `from` field specifies the lower bound of the parameter range.
/// - The `to` field specifies the upper bound of the parameter range.
/// - The `from` value must be less than the `to` value.
///
/// This structure is used in profile likelihood analysis to define which parameters
/// should be profiled and over what ranges. The profile likelihood method evaluates
/// how the likelihood function changes as each parameter varies across its specified range.
#[derive(Debug, Clone, bon::Builder)]
#[allow(clippy::duplicated_attributes)]
#[builder(on(String, into), on(f64, into))]
pub struct ProfileParameter {
    /// Name of the profiled parameter
    ///
    /// This should match the parameter name in your model that you want to profile.
    pub(crate) name: String,

    /// From value (lower bound of the range)
    ///
    /// The minimum value in the parameter range to be profiled. The profile likelihood
    /// analysis will start from this value and increase toward the upper bound.
    pub(crate) from: f64,

    /// To value (upper bound of the range)
    ///
    /// The maximum value in the parameter range to be profiled. The profile likelihood
    /// analysis will evaluate points up to this value.
    pub(crate) to: f64,
}

impl ProfileParameter {
    /// Creates a new ProfileParameter instance.
    ///
    /// Constructs a new parameter configuration for profile likelihood analysis
    /// with validation to ensure the parameter range is valid (from < to).
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter to be profiled
    /// * `from` - The starting value (lower bound) of the parameter range
    /// * `to` - The ending value (upper bound) of the parameter range
    ///
    /// # Returns
    ///
    /// A Result containing either:
    /// - Ok(ProfileParameter): A new valid ProfileParameter instance
    /// - Err(OptimizeError): An error if the range is invalid (from >= to)
    pub fn new(name: String, from: f64, to: f64) -> Result<Self, OptimizeError> {
        if from >= to {
            return Err(OptimizeError::ProfileParameterParseError(format!(
                "From value {from} must be less than to value {to}"
            )));
        }

        Ok(Self { name, from, to })
    }
}

impl FromStr for ProfileParameter {
    type Err = OptimizeError;

    /// Parses a ProfileParameter from a string.
    ///
    /// The string should be in the format "name=from:to", where:
    /// - name is the parameter name (must be alphanumeric with underscores)
    /// - from is the lower bound of the range (must be a valid floating point number)
    /// - to is the upper bound of the range (must be a valid floating point number)
    ///
    /// This implementation allows ProfileParameter configurations to be easily
    /// specified in configuration files or command line arguments.
    ///
    /// # Returns
    ///
    /// A Result containing either:
    /// - Ok(ProfileParameter): The parsed ProfileParameter
    /// - Err(OptimizeError): An error if the string format is invalid or the range is invalid
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Example: "k_cat=1.0:2.0"
        let pattern = Regex::new(r"^(\w+)=([0-9.]+):([0-9.]+)$").unwrap();
        let caps = pattern
            .captures(s)
            .ok_or(OptimizeError::ProfileParameterParseError(s.to_string()))?;

        if caps.len() != 4 {
            return Err(OptimizeError::ProfileParameterParseError(s.to_string()));
        }

        let from = caps[2]
            .parse::<f64>()
            .map_err(|e| OptimizeError::ProfileParameterParseError(e.to_string()))?;
        let to = caps[3]
            .parse::<f64>()
            .map_err(|e| OptimizeError::ProfileParameterParseError(e.to_string()))?;

        Self::new(caps[1].to_string(), from, to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_parameter_parse() {
        let input = "K_M=50.0:100.0";
        let param = ProfileParameter::from_str(input).expect("Should not fail");
        assert_eq!(param.name, "K_M");
        assert_eq!(param.from, 50.0);
        assert_eq!(param.to, 100.0);

        let input = "K_1M=50.0:100.0";
        let param = ProfileParameter::from_str(input).expect("Should not fail");
        assert_eq!(param.name, "K_1M");
        assert_eq!(param.from, 50.0);
        assert_eq!(param.to, 100.0);
    }

    #[test]
    fn test_profile_parameter_parse_error() {
        let input = "K_M=50.0:100.0:200.0";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {input} should fail");

        let input = "K_M=50,0:100,0";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {input} should fail");

        let input = "K_M=50.0,100.0:";
        let case = ProfileParameter::from_str(input);
        assert!(case.is_err(), "Case {input} should fail");
    }
}
