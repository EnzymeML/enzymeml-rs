use std::{iter::Map, slice::Iter};

use peroxide::{linspace, seq};
use rayon::prelude::*;

use super::parameter::ProfileParameter;

/// Type alias for the iterator returned by `ProfileGrid::iter()`
pub type ProfileGridValuesIter<'a> =
    Map<Iter<'a, ProfileGridPoint>, fn(&'a ProfileGridPoint) -> &'a [f64]>;

/// A grid of parameter values for profile likelihood analysis.
///
/// Contains the parameter values for each parameter to be profiled.
/// Stores a collection of grid points that represent different parameter combinations
/// to be evaluated during profile likelihood analysis.
///
/// This structure is used to systematically explore the parameter space by creating
/// a grid of points that cover the ranges of parameters being profiled.
#[derive(Debug, Clone)]
pub struct ProfileGrid {
    /// Collection of grid points representing parameter combinations
    points: Vec<ProfileGridPoint>,
}

impl ProfileGrid {
    /// Creates a grid of all possible combinations of parameter values for profile likelihood analysis.
    ///
    /// This method generates a grid by taking the Cartesian product of parameter values across
    /// their specified ranges. For each parameter, a uniform grid of values is created between
    /// the lower and upper bounds, and then all combinations of these values are generated.
    ///
    /// # Arguments
    ///
    /// * `parameters` - List of parameters to profile with their ranges (lower and upper bounds)
    /// * `n_points` - Number of equally spaced points to evaluate within each parameter range
    ///
    /// # Returns
    ///
    /// A `ProfileGrid` containing all combinations of parameter values
    pub fn from_profile_parameters(parameters: &[ProfileParameter], n_points: usize) -> Self {
        if parameters.is_empty() {
            return Self { points: Vec::new() };
        }

        // First generate the linspaces for each parameter
        let mut parameter_grids: Vec<(String, Vec<f64>)> = Vec::new();
        for parameter in parameters {
            let values = linspace!(parameter.from, parameter.to, n_points);
            parameter_grids.push((parameter.name.clone(), values));
        }

        // Generate all possible combinations
        let points = Self::generate_combinations(&parameter_grids, 0, vec![], vec![]);

        Self { points }
    }

    /// Helper function to recursively generate all combinations of parameter values.
    ///
    /// This method implements a recursive algorithm to generate the Cartesian product
    /// of all parameter value sets. It builds up combinations by adding one parameter
    /// at a time, recursively generating all possible combinations.
    ///
    /// # Arguments
    ///
    /// * `grids` - Vector of parameter names and their value ranges
    /// * `current_index` - Current index in the grids vector, representing the current parameter
    /// * `current_params` - Current parameter names being accumulated in this recursion branch
    /// * `current_values` - Current parameter values being accumulated in this recursion branch
    ///
    /// # Returns
    ///
    /// Vector of ProfileGridPoint containing all combinations of parameter values
    fn generate_combinations(
        grids: &[(String, Vec<f64>)],
        current_index: usize,
        current_params: Vec<String>,
        current_values: Vec<f64>,
    ) -> Vec<ProfileGridPoint> {
        if current_index == grids.len() {
            return vec![ProfileGridPoint {
                parameters: current_params,
                values: current_values,
            }];
        }

        let (param_name, values) = &grids[current_index];
        let mut result = Vec::new();

        for &value in values {
            let mut params_copy = current_params.clone();
            let mut values_copy = current_values.clone();

            params_copy.push(param_name.clone());
            values_copy.push(value);

            let points =
                Self::generate_combinations(grids, current_index + 1, params_copy, values_copy);
            result.extend(points);
        }

        result
    }

    /// Get a reference to the grid points.
    ///
    /// # Returns
    ///
    /// A slice containing all the grid points in this ProfileGrid
    pub fn points(&self) -> &[ProfileGridPoint] {
        &self.points
    }

    /// Returns an iterator over the grid points.
    ///
    /// This method provides an iterator that yields slices of
    /// parameter values for each grid point.
    ///
    /// # Returns
    ///
    /// An iterator over the grid points' values
    pub fn iter(&self) -> ProfileGridValuesIter<'_> {
        self.points.iter().map(|point| point.values.as_slice())
    }

    /// Returns the length of the grid.
    ///
    /// # Returns
    ///
    /// The number of grid points in this ProfileGrid
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the grid is empty.
    ///
    /// # Returns
    ///
    /// True if the grid is empty, false otherwise
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl<'a> IntoIterator for &'a ProfileGrid {
    type Item = &'a [f64];
    type IntoIter = ProfileGridValuesIter<'a>;

    /// Converts a reference to ProfileGrid into an iterator.
    ///
    /// This implementation allows ProfileGrid to be used in for loops.
    /// The iterator returns parameter values.
    ///
    /// # Returns
    ///
    /// An iterator over the grid points, yielding parameter values
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoParallelIterator for &'a ProfileGrid {
    type Iter = rayon::iter::Map<
        rayon::slice::Iter<'a, ProfileGridPoint>,
        fn(&'a ProfileGridPoint) -> Self::Item,
    >;
    type Item = (&'a [String], &'a [f64]);

    fn into_par_iter(self) -> Self::Iter {
        self.points
            .par_iter()
            .map(|point| (point.parameters.as_slice(), point.values.as_slice()))
    }
}

/// A point in the profile grid.
///
/// Represents a single combination of parameter values for profile likelihood analysis.
/// Each point contains the parameter names and their corresponding values.
///
/// This structure is used within the ProfileGrid to store individual parameter combinations
/// that will be evaluated during profile likelihood analysis.
#[derive(Debug, Clone)]
pub struct ProfileGridPoint {
    /// Names of the parameters for this grid point
    pub parameters: Vec<String>,
    /// Values of the parameters for this grid point
    pub values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_empty_parameters() {
        let grid = ProfileGrid::from_profile_parameters(&[], 5);
        assert!(grid.points().is_empty());
    }

    #[test]
    fn test_single_parameter() {
        let param = ProfileParameter::new("k_cat".to_string(), 1.0, 5.0).unwrap();
        let grid = ProfileGrid::from_profile_parameters(&[param], 5);

        assert_eq!(grid.points().len(), 5);
        assert_eq!(grid.points()[0].parameters[0], "k_cat");
        assert_eq!(grid.points()[0].values[0], 1.0);
        assert_eq!(grid.points()[4].values[0], 5.0);
    }

    #[test]
    fn test_two_parameters() {
        let param1 = ProfileParameter::new("k_cat".to_string(), 1.0, 3.0).unwrap();
        let param2 = ProfileParameter::new("K_M".to_string(), 10.0, 20.0).unwrap();
        let grid = ProfileGrid::from_profile_parameters(&[param1, param2], 3);

        // Should have 3×3=9 points
        assert_eq!(grid.points().len(), 9);

        // Check first point
        assert_eq!(grid.points()[0].parameters, vec!["k_cat", "K_M"]);
        assert_eq!(grid.points()[0].values, vec![1.0, 10.0]);

        // Check last point
        assert_eq!(grid.points()[8].parameters, vec!["k_cat", "K_M"]);
        assert_eq!(grid.points()[8].values, vec![3.0, 20.0]);
    }

    #[test]
    fn test_three_parameters() {
        let param1 = ProfileParameter::from_str("k_cat=1.0:2.0").unwrap();
        let param2 = ProfileParameter::from_str("K_M=10.0:20.0").unwrap();
        let param3 = ProfileParameter::from_str("k_ie=0.1:0.2").unwrap();
        let grid = ProfileGrid::from_profile_parameters(&[param1, param2, param3], 2);

        // Should have 2×2×2=8 points
        assert_eq!(grid.points().len(), 8);

        // First point should have minimum values
        assert_eq!(grid.points()[0].parameters, vec!["k_cat", "K_M", "k_ie"]);
        assert_eq!(grid.points()[0].values, vec![1.0, 10.0, 0.1]);

        // Last point should have maximum values
        assert_eq!(grid.points()[7].parameters, vec!["k_cat", "K_M", "k_ie"]);
        assert_eq!(grid.points()[7].values, vec![2.0, 20.0, 0.2]);
    }

    #[test]
    fn test_iter() {
        let param1 = ProfileParameter::from_str("k_cat=1.0:2.0").unwrap();
        let param2 = ProfileParameter::from_str("K_M=10.0:20.0").unwrap();
        let param3 = ProfileParameter::from_str("k_ie=0.1:0.2").unwrap();
        let grid = ProfileGrid::from_profile_parameters(&[param1, param2, param3], 2);

        let mut iter = grid.iter();
        assert_eq!(iter.next(), Some(vec![1.0, 10.0, 0.1].as_slice()));
        assert_eq!(iter.next(), Some(vec![1.0, 10.0, 0.2].as_slice()));
        assert_eq!(iter.next(), Some(vec![1.0, 20.0, 0.1].as_slice()));
        assert_eq!(iter.next(), Some(vec![1.0, 20.0, 0.2].as_slice()));
        assert_eq!(iter.next(), Some(vec![2.0, 10.0, 0.1].as_slice()));
        assert_eq!(iter.next(), Some(vec![2.0, 10.0, 0.2].as_slice()));
        assert_eq!(iter.next(), Some(vec![2.0, 20.0, 0.1].as_slice()));
        assert_eq!(iter.next(), Some(vec![2.0, 20.0, 0.2].as_slice()));
        assert_eq!(iter.next(), None);
    }
}
