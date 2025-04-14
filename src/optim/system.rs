//! This module provides optimization functionality for EnzymeML documents, including cost function
//! calculation and gradient computation for parameter optimization.

use argmin::core::{CostFunction, Gradient, Hessian};
use finitediff::FiniteDiff;
use ndarray::{Array1, Array2, Array3, Axis, ShapeError};
use peroxide::fuga::ODEIntegrator;

use crate::{
    objective::objfun::ObjectiveFunction,
    prelude::{MatrixResult, Mode},
};

use super::{problem::Problem, OptimizeError};

/// Implementation of the CostFunction trait for Problem to enable parameter optimization.
/// This allows using the Problem with optimization algorithms that minimize a cost function.
impl<S: ODEIntegrator + Copy, L: ObjectiveFunction> CostFunction for Problem<S, L> {
    type Param = Array1<f64>;
    type Output = f64;

    /// Calculates the cost between simulated and measured data.
    /// The cost is computed by running simulations with the provided parameters and comparing
    /// the results to experimental measurements.
    ///
    /// # Arguments
    /// * `params` - Vector of parameter values to test in the simulation
    ///
    /// # Returns
    /// * `Result<f64, argmin::core::Error>` - The calculated error according to the objective function
    ///
    /// # Details
    /// The function performs the following steps:
    /// 1. Maps the parameter vector to named parameters
    /// 2. Simulates the system with the given parameters in parallel chunks
    /// 3. Computes the error according to the objective function
    fn cost(&self, params: &Self::Param) -> Result<f64, argmin::core::Error> {
        // Parallelize simulation and residual calculation in a single pass
        let results = self.ode_system().bulk_integrate::<MatrixResult>(
            self.simulation_setup(),
            self.initials(),
            params.as_slice(),
            Some(self.evaluation_times()),
            self.solver(),
            Some(Mode::Regular),
        )?;

        let species = concatenate_results(results)?;

        // Extract the columns of the species and sensitivities arrays
        // that correspond to the observable species
        let species = species.select(Axis(1), self.observable_species());

        let residuals = species - self.measurement_buffer();
        let cost = self.objective().cost(&residuals, self.n_points()).unwrap();

        if cost.is_nan() {
            return Err(argmin::core::Error::msg(OptimizeError::CostNaN.to_string()));
        }

        Ok(cost)
    }
}

/// Implementation of the Gradient trait for Problem to enable gradient-based optimization.
/// The gradient is computed using central finite differences for better accuracy.
impl<S: ODEIntegrator + Copy, L: ObjectiveFunction> Gradient for Problem<S, L> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    /// Computes the gradient of the cost function with respect to the parameters using central differences.
    ///
    /// # Arguments
    /// * `params` - Vector of parameter values at which to evaluate the gradient
    ///
    /// # Returns
    /// * `Result<Array1<f64>, argmin::core::Error>` - The computed gradient vector or an error
    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let gradient = params.central_diff(&|x| self.cost(x).expect("Failed to compute cost"));
        Ok(gradient)
    }
}

/// Implementation of the Hessian trait for Problem to enable second-order optimization methods.
/// The Hessian is computed using central finite differences of the gradient for better accuracy.
///
/// # Arguments
/// * `params` - Vector of parameter values at which to evaluate the Hessian
///
/// # Returns
/// * `Result<Array2<f64>, argmin::core::Error>` - The computed Hessian matrix or an error
impl<S: ODEIntegrator + Copy, L: ObjectiveFunction> Hessian for Problem<S, L> {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, params: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let hessian =
            params.central_hessian(&|x| self.gradient(x).expect("Failed to compute gradient"));
        Ok(hessian)
    }
}

fn concatenate_results(results: Vec<MatrixResult>) -> Result<Array2<f64>, ShapeError> {
    let (species, _): (Vec<Array2<f64>>, Vec<Option<Array3<f64>>>) = results
        .into_iter()
        .map(|r| (r.species, r.parameter_sensitivities))
        .unzip();

    let species_views: Vec<_> = species.iter().map(|x| x.view()).collect();
    let concatenated_species = ndarray::concatenate(ndarray::Axis(0), species_views.as_slice())?;
    Ok(concatenated_species)

    // Leaving this here for now, but it's not used
    // let sensitivities: Vec<Array3<f64>> = sensitivities.into_iter().flatten().collect();
    // let sensitivities_views: Vec<_> = sensitivities.iter().map(|x| x.view()).collect();
    // let concatenated_sensitivities = ndarray::concatenate(ndarray::Axis(0), &sensitivities_views)?;
}
