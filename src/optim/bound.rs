use ndarray::Array2;
use peroxide::fuga::ODEIntegrator;

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
}

pub(crate) fn bounds_to_array2<S: ODEIntegrator + Copy>(
    problem: &Problem<S>,
    bounds: &Vec<Bound>,
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
    let sorted_bounds = sort_by_system_params(bounds.clone(), &sorted_params);

    // Create array with bounds
    let mut array = Array2::zeros((bounds.len(), 2));
    for (i, bound) in sorted_bounds.iter().enumerate() {
        array[[i, 0]] = bound.lower;
        array[[i, 1]] = bound.upper;
    }
    Ok(array)
}

fn has_all_params(bound_params: &[String], system_params: &[String]) -> bool {
    system_params
        .iter()
        .all(|p| bound_params.contains(&p.to_string()))
}

fn sort_by_system_params(bounds: Vec<Bound>, sorted_params: &[String]) -> Vec<Bound> {
    let mut sorted_bounds = bounds.clone();
    sorted_bounds.sort_by_key(|b| sorted_params.iter().position(|p| p == &b.param).unwrap());
    sorted_bounds
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
        let sorted_bounds = sort_by_system_params(bounds, &sorted_params);
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
