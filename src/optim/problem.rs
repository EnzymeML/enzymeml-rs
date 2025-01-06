use std::collections::HashMap;

use argmin::{
    core::Hessian,
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use derive_builder::Builder;
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

use crate::prelude::{
    EnzymeMLDocument, Equation, EquationBuilder, EquationType, Initials, SimulationSetup,
};

use super::{error::OptimizeError, objective::ObjectiveFunction};

/// Type alias for the More-Thuente line search algorithm used in optimization
type LineSearchType = MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>;

/// Represents an optimization problem for enzyme kinetics parameter estimation
///
/// Contains all necessary components to set up and run a parameter optimization problem,
/// including the model definition, experimental data, initial conditions, simulation settings,
/// solver configuration and optional parameter transformations.
///
/// # Fields
/// * `doc` - EnzymeML document containing the model definition, parameters and experimental data
/// * `initials` - Optional user-provided initial conditions, otherwise derived from the document
/// * `simulation_setup` - Configuration for numerical integration of the model
/// * `solver` - Optimization algorithm configuration
/// * `transformations` - Optional parameter transformations to enforce constraints
#[derive(Debug, Clone, Builder)]
pub struct Problem {
    pub doc: EnzymeMLDocument,
    #[builder(default)]
    pub initials: Option<Initials>,
    pub simulation_setup: SimulationSetup,
    pub solver: Solver,
    #[builder(default, setter(each(name = "transform", into)))]
    pub transformations: Vec<Transformation>,
    #[builder(default = "ObjectiveFunction::MSE")]
    pub objective: ObjectiveFunction,
}

impl Problem {
    /// Gets the initial conditions for the optimization
    ///
    /// Returns user-provided initial conditions if available, otherwise attempts to
    /// derive them from the model definition in the EnzymeML document.
    ///
    /// # Returns
    /// * `Initials` - Initial conditions for all model species and parameters
    pub fn initials(&self) -> Initials {
        self.initials
            .clone()
            .unwrap_or_else(|| (&self.doc).try_into().unwrap())
    }

    /// Gets the configured optimization solver
    ///
    /// # Returns
    /// * `Solver` - Clone of the configured optimization algorithm
    pub fn solver(&self) -> Solver {
        self.solver.clone()
    }

    /// Sets up parameter transformations in the model equations
    ///
    /// For each defined transformation:
    /// 1. Creates a new equation implementing the transformation
    /// 2. Updates all existing equations to reference the transformed parameter
    /// 3. Adds the transformation equation to the model
    ///
    /// This allows enforcing constraints like positivity or bounds during optimization
    /// while maintaining the original parameter meanings.
    ///
    /// # Returns
    /// * `Result<(), OptimizeError>` - Success or error if transformation setup fails
    pub fn setup_transformations(&mut self) -> Result<(), OptimizeError> {
        for transformation in &self.transformations {
            let assignment = transformation.equation()?;
            for eq in self.doc.equations.iter_mut() {
                eq.equation = eq.equation.replace(
                    &transformation.symbol(),
                    &assignment.get_species_id().clone().unwrap(),
                );
            }

            self.doc.equations.push(assignment.clone());
        }
        Ok(())
    }

    /// Applies parameter transformations to optimization results
    ///
    /// Takes the raw parameter values from the optimizer and applies any defined
    /// transformations to convert them back to their original scale/meaning.
    ///
    /// # Arguments
    /// * `params` - Raw parameter values from the optimizer
    ///
    /// # Returns
    /// * `Result<HashMap<String, f64>, OptimizeError>` - Mapping of parameter names to their final values
    pub fn apply_transformations(
        &self,
        params: &Array1<f64>,
    ) -> Result<HashMap<String, f64>, OptimizeError> {
        let mut param_names = self
            .doc
            .parameters
            .iter()
            .map(|p| p.symbol.clone())
            .collect::<Vec<_>>();

        param_names.sort_unstable();

        let mut transformed_params = HashMap::new();

        for (i, param_name) in param_names.iter().enumerate() {
            if let Some(transformation) = self
                .transformations
                .iter()
                .find(|t| t.symbol() == *param_name)
            {
                let transformed_value = transformation.apply(params[i]);
                transformed_params.insert(param_name.clone(), transformed_value);
            } else {
                transformed_params.insert(param_name.clone(), params[i]);
            }
        }

        Ok(transformed_params)
    }

    /// Resets all parameter transformations in the model equations
    ///
    /// This function:
    /// 1. Replaces transformed parameter symbols with their original symbols in all equations
    /// 2. Removes any transformation equations that were added to the model
    ///
    /// This effectively reverts the model back to its pre-transformation state, which is useful
    /// after optimization is complete and transformed parameters are no longer needed.
    ///
    /// # Details
    /// For each transformation:
    /// - Scans through all equations and replaces transformed symbols (e.g. "k1_transformed")
    ///   with original parameter symbols (e.g. "k1")
    /// - Removes any equations that were added specifically for the transformation
    ///   (identified by having the transformed parameter as their species_id)
    pub fn reset_transformations(&mut self) {
        for transformation in &self.transformations {
            for eq in self.doc.equations.iter_mut() {
                eq.equation = eq.equation.replace(
                    &transformation.transform_symbol(),
                    &transformation.symbol().clone(),
                );
            }

            // Remove the transformation equation from the model
            self.doc
                .equations
                .retain(|eq| eq.species_id != Some(transformation.transform_symbol()));
        }
    }

    /// Returns the total number of parameters in the optimization problem
    ///
    /// # Returns
    /// * `usize` - Number of parameters in the model
    pub fn get_n_params(&self) -> usize {
        self.doc.parameters.len()
    }

    /// Returns the total number of data points across all measurements
    ///
    /// Counts the number of data points by summing up the length of data arrays
    /// for each species in each measurement that has data.
    ///
    /// # Returns
    /// * `usize` - Total number of data points
    pub fn get_n_points(&self) -> usize {
        let mut n_points = 0;
        for m in &self.doc.measurements {
            for s in &m.species_data {
                if let Some(data) = &s.data {
                    n_points += data.len();
                }
            }
        }
        n_points
    }

    /// Computes the inverse of the Hessian matrix at the given parameter values
    ///
    /// The inverse Hessian provides information about parameter uncertainties and correlations.
    /// It is computed by:
    /// 1. Calculating the Hessian matrix using finite differences
    /// 2. Converting the ndarray Hessian to a nalgebra matrix
    /// 3. Computing the matrix inverse
    ///
    /// # Arguments
    /// * `params` - Parameter values at which to evaluate the Hessian
    ///
    /// # Returns
    /// * `Result<DMatrix<f64>, OptimizeError>` - The inverse Hessian matrix or an error
    ///
    /// # Errors
    /// Returns OptimizeError if:
    /// * Hessian computation fails
    /// * Matrix inversion fails
    pub fn inverse_hessian(&self, params: &Array1<f64>) -> Result<DMatrix<f64>, OptimizeError> {
        let hessian = self.hessian(params).map_err(OptimizeError::ArgMinError)?;

        // Convert hessian to nalgebra array
        let hessian_nalgebra = Self::array2_to_dmatrix(&hessian);
        let inverse_hessian = hessian_nalgebra.try_inverse().unwrap();
        Ok(inverse_hessian)
    }

    /// Converts an ndarray Array2 to a nalgebra DMatrix
    ///
    /// This helper function handles the conversion between array types by:
    /// 1. Getting a contiguous slice of the input array
    /// 2. Creating a new DMatrix with the same dimensions and data
    ///
    /// # Arguments
    /// * `arr` - The ndarray Array2 to convert
    ///
    /// # Returns
    /// * `DMatrix<f64>` - The converted nalgebra matrix
    ///
    /// # Panics
    /// Panics if the input array is not contiguous in memory
    fn array2_to_dmatrix(arr: &Array2<f64>) -> DMatrix<f64> {
        // Make sure that `arr` is contiguous (via `.as_slice().ok_or(...)` if needed).
        let slice = arr
            .as_slice()
            .expect("Array must be contiguous in memory to use `as_slice()`");

        // Create a DMatrix from the row-major slice.
        DMatrix::from_row_slice(arr.nrows(), arr.ncols(), slice)
    }
}

/// Parameter transformations for enforcing constraints during optimization
///
/// These transformations modify parameters during optimization to enforce constraints
/// or improve convergence behavior, while preserving the original parameter meanings
/// in the model:
///
/// - SoftPlus: Ensures positivity by transforming through ln(1 + exp(x))
/// - MultScale: Scales parameter by a constant factor for better numerical properties
/// - Pow: Raises parameter to a power to adjust sensitivity
/// - Logit: Maps parameter to (0,1) interval for bounded optimization
/// - Abs: Ensures positivity through absolute value
///
/// # Variants
/// * `SoftPlus(String)` - Softplus transformation of the named parameter
/// * `MultScale(String, f64)` - Scale the named parameter by the given factor
/// * `Pow(String, f64)` - Raise named parameter to the specified power
/// * `Logit(String)` - Logit transformation of the named parameter
/// * `Abs(String)` - Absolute value of the named parameter
#[derive(Debug, Clone)]
pub enum Transformation {
    SoftPlus(String),
    MultScale(String, f64),
    Pow(String, f64),
    Logit(String),
    Abs(String),
}

impl Transformation {
    /// Creates an equation representing this transformation
    ///
    /// Generates an initial assignment equation that implements the transformation
    /// in the model. This allows the transformation to be applied during simulation
    /// while keeping the original parameter meanings.
    ///
    /// # Returns
    /// * `Result<Equation, OptimizeError>` - The transformation equation or an error if creation fails
    ///
    /// # Errors
    /// Returns OptimizeError if the equation cannot be constructed
    pub fn equation(&self) -> Result<Equation, OptimizeError> {
        let variable = match self {
            Transformation::SoftPlus(s) => s,
            Transformation::MultScale(s, _) => s,
            Transformation::Pow(s, _) => s,
            Transformation::Logit(s) => s,
            Transformation::Abs(s) => s,
        };

        let equation_string = match self {
            Transformation::SoftPlus(s) => format!("ln(1 + exp({s}))", s = s),
            Transformation::MultScale(s, scale) => format!("{s} * {scale}", s = s, scale = scale),
            Transformation::Pow(s, power) => format!("{s}^{power}", s = s, power = power),
            Transformation::Logit(s) => format!("ln(1 + exp({s}))", s = s),
            Transformation::Abs(s) => format!("abs({s})", s = s),
        };

        EquationBuilder::default()
            .species_id(format!("{variable}_transformed"))
            .equation(equation_string.clone())
            .equation_type(EquationType::InitialAssignment)
            .build()
            .map_err(|e| OptimizeError::TransformationError {
                variable: variable.to_string(),
                transformation: equation_string,
                message: e.to_string(),
            })
    }

    /// Applies the transformation to a single parameter value
    ///
    /// Computes the transformed value for a parameter according to the
    /// transformation type. This is used to convert optimizer results
    /// back to their original scale/meaning.
    ///
    /// # Arguments
    /// * `value` - Parameter value to transform
    ///
    /// # Returns
    /// * `f64` - The transformed parameter value
    pub fn apply(&self, value: f64) -> f64 {
        match self {
            Transformation::SoftPlus(_) => (value.exp() - 1.0).ln(),
            Transformation::MultScale(_, scale) => value * scale,
            Transformation::Pow(_, power) => value.powi(*power as i32),
            Transformation::Logit(_) => (value / (1.0 - value)).ln(),
            Transformation::Abs(_) => value.abs(),
        }
    }

    /// Gets the original parameter symbol/name for this transformation
    ///
    /// Returns the untransformed parameter name that this transformation is applied to.
    /// This is used to identify which parameter in the equations needs to be replaced
    /// with its transformed version.
    ///
    /// # Returns
    /// * `String` - The original parameter symbol/name
    pub fn symbol(&self) -> String {
        match self {
            Transformation::SoftPlus(s) => s.clone(),
            Transformation::MultScale(s, _) => s.clone(),
            Transformation::Pow(s, _) => s.clone(),
            Transformation::Logit(s) => s.clone(),
            Transformation::Abs(s) => s.clone(),
        }
    }

    /// Gets the transformed parameter symbol/name for this transformation
    ///
    /// Returns the transformed parameter name by appending "_transformed" to the original name.
    /// This is used to identify the transformed parameter in the equations after applying
    /// the transformation.
    ///
    /// # Returns
    /// * `String` - The transformed parameter symbol/name
    pub fn transform_symbol(&self) -> String {
        match self {
            Transformation::SoftPlus(s) => format!("{s}_transformed"),
            Transformation::MultScale(s, _) => format!("{s}_transformed"),
            Transformation::Pow(s, _) => format!("{s}_transformed"),
            Transformation::Logit(s) => format!("{s}_transformed"),
            Transformation::Abs(s) => format!("{s}_transformed"),
        }
    }
}

/// Available optimization algorithms for parameter estimation
///
/// Currently supported algorithms:
/// * LBFGS - Limited-memory BFGS quasi-Newton method with More-Thuente line search
///   for efficient optimization of smooth functions
#[derive(Debug, Clone)]
pub enum Solver {
    LBFGSMoreThuente(LBFGMoreThuenteArgs),
}

impl Default for Solver {
    /// Creates default solver configuration using L-BFGS with standard settings
    fn default() -> Self {
        Solver::LBFGSMoreThuente(LBFGMoreThuenteArgs::default())
    }
}

impl Solver {
    /// Initializes the configured optimization algorithm
    ///
    /// Sets up the solver with all specified parameters and prepares it
    /// for optimization.
    ///
    /// # Returns
    /// * `Result<LBFGS<LineSearchType, Array1<f64>, Array1<f64>, f64>, OptimizeError>` -
    ///   Configured solver instance or error if initialization fails
    pub fn setup(
        self,
    ) -> Result<LBFGS<LineSearchType, Array1<f64>, Array1<f64>, f64>, OptimizeError> {
        match self {
            Solver::LBFGSMoreThuente(args) => args.setup(),
        }
    }
}

/// Configuration for the L-BFGS optimization algorithm
///
/// Specifies all parameters controlling the L-BFGS algorithm behavior:
///
/// # Fields
/// * `linesearch` - More-Thuente line search settings for step size selection
/// * `max_iters` - Maximum number of optimization iterations
/// * `tolerance_cost` - Convergence threshold for the objective function
/// * `memory_size` - Number of previous iterations stored for Hessian approximation
#[derive(Debug, Clone, Builder)]
pub struct LBFGMoreThuenteArgs {
    pub max_iters: usize,
    pub tolerance_cost: f64,
    pub memory_size: usize,
}

impl LBFGMoreThuenteArgs {
    /// Creates and configures an L-BFGS solver instance
    ///
    /// Initializes the L-BFGS algorithm with all specified settings including
    /// line search parameters, convergence criteria, and memory size.
    ///
    /// # Returns
    /// * `Result<LBFGS<LineSearchType, Array1<f64>, Array1<f64>, f64>, OptimizeError>` -
    ///   Configured L-BFGS solver or error if initialization fails
    pub fn setup(
        self,
    ) -> Result<LBFGS<LineSearchType, Array1<f64>, Array1<f64>, f64>, OptimizeError> {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
        LBFGS::new(linesearch, self.memory_size)
            .with_tolerance_cost(self.tolerance_cost)
            .map_err(OptimizeError::ArgMinError)
    }
}

impl Default for LBFGMoreThuenteArgs {
    /// Creates L-BFGS configuration with standard settings
    ///
    /// Default values:
    /// * Line search: More-Thuente with c1=1e-4, c2=0.9 (Wolfe conditions)
    /// * Maximum iterations: 100
    /// * Cost tolerance: 1e-8
    /// * Memory size: 7 previous iterations
    fn default() -> Self {
        LBFGMoreThuenteArgsBuilder::default()
            .max_iters(100)
            .tolerance_cost(1e-8)
            .memory_size(7)
            .build()
            .unwrap()
    }
}
