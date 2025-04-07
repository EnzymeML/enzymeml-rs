use ndarray::Array2;
use peroxide::fuga::ODEIntegrator;

use crate::prelude::{EnzymeMLDocument, Parameter};

use super::{OptimizeError, Problem};

/// Represents bounds on an optimization parameter.
///
/// This struct defines the valid range for a parameter by specifying
/// its lower and upper bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Bound {
    /// Name of the parameter being bounded
    param: String,
    /// Lower bound/minimum allowed value for the parameter
    lower: f64,
    /// Upper bound/maximum allowed value for the parameter
    upper: f64,
}

impl Bound {
    /// Creates a new parameter bound with the specified name and range.
    ///
    /// # Arguments
    ///
    /// * `param` - Name of the parameter to bound
    /// * `lower` - Lower bound/minimum allowed value
    /// * `upper` - Upper bound/maximum allowed value
    pub fn new(param: String, lower: f64, upper: f64) -> Self {
        Self {
            param,
            lower,
            upper,
        }
    }

    /// Returns the parameter name.
    pub fn param(&self) -> &str {
        &self.param
    }

    /// Returns the lower bound of the parameter.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Returns the upper bound of the parameter.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Sets the lower bound of the parameter.
    ///
    /// # Arguments
    ///
    /// * `lower` - The new lower bound value
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the lower bound is valid
    /// * `Err(OptimizeError)` - If the lower bound is invalid
    pub fn set_lower(&mut self, lower: f64) {
        self.lower = lower;
    }

    /// Sets the upper bound of the parameter.
    ///
    /// # Arguments
    ///
    /// * `upper` - The new upper bound value
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the upper bound is valid
    /// * `Err(OptimizeError)` - If the upper bound is invalid
    pub fn set_upper(&mut self, upper: f64) {
        self.upper = upper;
    }

    /// Validates the bounds of the parameter.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the bounds are valid
    /// * `Err(OptimizeError)` - If the bounds are invalid
    pub fn validate(&self) -> Result<(), OptimizeError> {
        if self.lower > self.upper {
            return Err(OptimizeError::InvalidBounds {
                expected: vec![self.param.clone()],
                found: vec![self.lower.to_string(), self.upper.to_string()],
            });
        }
        Ok(())
    }
}

/// Converts a vector of bounds into a 2D array format required by optimization algorithms.
///
/// # Arguments
///
/// * `problem` - The optimization problem containing system parameters
/// * `bounds` - Vector of parameter bounds
///
/// # Returns
///
/// * `Ok(Array2<f64>)` - 2D array where each row contains [lower_bound, upper_bound] for a parameter
/// * `Err(OptimizeError)` - Error if bounds are invalid or missing parameters
pub(crate) fn bounds_to_array2<S: ODEIntegrator + Copy>(
    problem: &Problem<S>,
    bounds: &[Bound],
) -> Result<Array2<f64>, OptimizeError> {
    let bound_params = bounds.iter().map(|b| b.param.clone()).collect::<Vec<_>>();
    let system_params = problem
        .ode_system()
        .get_params_mapping()
        .keys()
        .cloned()
        .collect::<Vec<String>>();

    if !has_all_params(&bound_params, &system_params) {
        return Err(OptimizeError::InvalidBounds {
            expected: system_params,
            found: bound_params.iter().map(|p| p.to_string()).collect(),
        });
    }

    // Sort bounds by system parameters
    let sorted_params = problem.ode_system().get_sorted_params();
    let sorted_bounds = sort_by_system_params(bounds, &sorted_params);

    // Create array with bounds
    let mut array = Array2::zeros((bounds.len(), 2));
    for (i, bound) in sorted_bounds.iter().enumerate() {
        array[[i, 0]] = bound.lower;
        array[[i, 1]] = bound.upper;
    }
    Ok(array)
}

/// Checks if all system parameters have corresponding bounds.
///
/// # Arguments
///
/// * `bound_params` - List of parameter names that have bounds defined
/// * `system_params` - List of parameter names required by the system
///
/// # Returns
///
/// `true` if all system parameters have bounds, `false` otherwise
fn has_all_params(bound_params: &[String], system_params: &[String]) -> bool {
    system_params
        .iter()
        .all(|p| bound_params.contains(&p.to_string()))
}

/// Sorts bounds to match the order of parameters in the system.
///
/// # Arguments
///
/// * `bounds` - Vector of parameter bounds to sort
/// * `sorted_params` - Reference parameter order to sort by
///
/// # Returns
///
/// New vector of bounds sorted to match the parameter order
fn sort_by_system_params(bounds: &[Bound], sorted_params: &[String]) -> Vec<Bound> {
    let mut sorted_bounds = bounds.to_vec();
    sorted_bounds.sort_by_key(|b| sorted_params.iter().position(|p| p == &b.param).unwrap());
    sorted_bounds
}

impl TryFrom<&EnzymeMLDocument> for Vec<Bound> {
    type Error = OptimizeError;

    /// Attempts to create bounds from an EnzymeML document.
    ///
    /// Extracts parameter bounds from the document's parameters section.
    fn try_from(doc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        Ok(doc
            .parameters
            .iter()
            .map(|p| p.try_into().unwrap())
            .collect())
    }
}

impl TryFrom<&Parameter> for Bound {
    type Error = OptimizeError;

    /// Attempts to create a bound from a parameter definition.
    ///
    /// # Errors
    ///
    /// Returns an error if either the lower or upper bound is missing from the parameter.
    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        let lower = param.lower_bound.ok_or(OptimizeError::MissingLowerBound {
            param: param.id.clone(),
        })?;
        let upper = param.upper_bound.ok_or(OptimizeError::MissingUpperBound {
            param: param.id.clone(),
        })?;

        Ok(Bound::new(param.id.clone(), lower, upper))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_by_system_params() {
        let bounds = vec![
            Bound::new("k1".to_string(), 0.0, 1.0),
            Bound::new("k2".to_string(), 0.0, 1.0),
        ];
        let sorted_params = vec!["k1".to_string(), "k2".to_string()];
        let sorted_bounds = sort_by_system_params(&bounds, &sorted_params);
        assert_eq!(
            sorted_bounds,
            vec![
                Bound {
                    param: "k1".to_string(),
                    lower: 0.0,
                    upper: 1.0
                },
                Bound {
                    param: "k2".to_string(),
                    lower: 0.0,
                    upper: 1.0
                },
            ]
        );
    }
}
