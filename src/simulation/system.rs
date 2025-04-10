//! Simulation module for solving ODE systems.
//!
//! This module provides functionality for simulating reaction networks described by systems of
//! ordinary differential equations (ODEs). The main struct [`ODESystem`] handles:
//!
//! - Numerical integration of ODEs using Dormand-Prince (DOPRI5) method
//! - Parameter sensitivity analysis
//! - Assignment rules that must be satisfied during simulation
//! - Initial assignments evaluated only at t=0
//! - Efficient computation using JIT-compiled equations
//!
//! The system maintains mappings between variable names (species, parameters, etc.) and their
//! indices in the state vector, along with buffers for efficient computation.

use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
};

use evalexpr_jit::prelude::*;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};
use ndarray::Array1;
use peroxide::fuga::{BasicODESolver, ODEIntegrator, ODEProblem, ODESolver};

use crate::prelude::{EnzymeMLDocument, Equation, EquationType};

use super::{
    error::SimulationError, interpolation::interpolate, output::OutputFormat, SimulationSetup,
};

/// Type alias for the output of the ODE system.
/// Uses a vector of DMatrix<f64> to represent the output.
pub type StepperOutput = Vec<Vec<f64>>;

/// Represents an ODE system for simulating reaction networks.
///
/// This struct contains all the information needed to simulate a system of ODEs,
/// including parameters, initial conditions, equations, and mappings between
/// species names and their indices in the state vector.
///
/// The system supports three types of equations:
/// - ODEs: Differential equations describing species concentrations over time
/// - Assignment rules: Equations that must be satisfied at all times
/// - Initial assignments: Equations that are only evaluated at t=0
///
/// It also maintains buffers for efficient computation and provides methods for
/// parameter/species updates and gradient calculations.
#[derive(Clone)]
pub struct ODESystem {
    // Variable mapping
    var_map: HashMap<String, u32>,
    species_mapping: HashMap<String, u32>,
    params_mapping: HashMap<String, u32>,
    initial_assignments_mapping: HashMap<String, u32>,
    assignment_rules_mapping: HashMap<String, u32>,
    sorted_vars: Vec<String>,

    // Equation systems for all three types of equations
    ode_jit: EquationSystem,
    assignment_jit: EquationSystem,
    initial_assignment_jit: EquationSystem,

    // Gradient functions
    grad_params: EquationSystem,
    grad_species: EquationSystem,

    // Buffers used for intermediate calculations
    initial_assignment_buffer: RefCell<Vec<f64>>,
    assignment_buffer: RefCell<Vec<f64>>,
    species_buffer: RefCell<Vec<f64>>,
    params_buffer: RefCell<Vec<f64>>,

    // Mode
    mode: RefCell<Mode>,

    /// Buffer for system evaluation input vector
    input_buffer: RefCell<Vec<f64>>,
    /// Buffer for assignment rule evaluations
    assignments_result_buffer: RefCell<Vec<f64>>,

    /// Starts and ends of the parts of the input vector
    species_range: (usize, usize),
    parameters_range: (usize, usize),
    initial_assignments_range: (usize, usize),
    assignment_rules_range: (usize, usize),
}

impl ODESystem {
    /// Creates a new ODESystem instance.
    ///
    /// This function initializes all components of the ODE system, including:
    /// - Creating variable mappings and ordering
    /// - Compiling equations into efficient JIT representations
    /// - Setting up gradient calculations
    /// - Initializing buffers for computation
    ///
    /// # Arguments
    ///
    /// * `params` - A hashmap mapping parameter names to their initial values
    /// * `equations` - A struct containing ODEs, assignment rules, and initial assignments
    ///
    /// # Returns
    ///
    /// Returns a Result containing either:
    /// - Ok(ODESystem): A fully initialized ODE system ready for simulation
    /// - Err(SimulationError): An error describing what went wrong during initialization
    ///
    /// # Errors
    ///
    /// Can return errors for:
    /// - Invalid equation syntax
    /// - Failed equation compilation
    /// - Failed gradient calculation
    pub(crate) fn new(
        odes: HashMap<String, String>,
        initial_assignments: HashMap<String, String>,
        assignments: HashMap<String, String>,
        params: HashMap<String, f64>,
        mode: Option<Mode>,
    ) -> Result<Self, SimulationError> {
        // We need to create an index mapping for all variables
        // 0. Time
        // 1. Species
        // 2. Parameters
        // 3. Initial assignments
        // 4. Assignment rules
        //
        // These are then used as the var_map for all the EquationSystem instances

        // Extract all variables and sort them
        let mut species: Vec<String> = odes.keys().map(|x| x.to_string()).collect();
        let mut parameters: Vec<String> = params.keys().map(|x| x.to_string()).collect();
        let mut initial_assignment_rules: Vec<String> =
            initial_assignments.keys().map(|x| x.to_string()).collect();
        let mut assignment_rules: Vec<String> = assignments.keys().map(|x| x.to_string()).collect();

        species.sort();
        parameters.sort();
        initial_assignment_rules.sort();
        assignment_rules.sort();

        // Create the sorted list of all variables
        let mut sorted_vars = vec!["t".to_string()];
        sorted_vars.extend(species.clone());
        sorted_vars.extend(parameters.clone());
        sorted_vars.extend(assignment_rules.clone());
        sorted_vars.extend(initial_assignment_rules.clone());

        // Create the variable mapping
        let var_map = HashMap::from_iter(
            sorted_vars
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i as u32)),
        );

        // Derive the mappings for species, parameters, initial assignments
        // and assignment rules from the var_map master mapping
        let species_mapping = Self::derive_mapping(&var_map, &species)?;
        let params_mapping = Self::derive_mapping(&var_map, &parameters)?;
        let initial_assignments_mapping =
            Self::derive_mapping(&var_map, &initial_assignment_rules)?;
        let assignment_rules_mapping = Self::derive_mapping(&var_map, &assignment_rules)?;

        // Parse equations into EquationSystem instances
        let ode_jit = Self::parse_equations_to_systems(&odes, &var_map)?;
        let assignment_jit = Self::parse_equations_to_systems(&assignments, &var_map)?;
        let initial_assignment_jit =
            Self::parse_equations_to_systems(&initial_assignments, &var_map)?;

        // Get the gradient wrt to parameters and species
        let grad_params = ode_jit
            .jacobian_wrt(&parameters.iter().map(|x| x.as_str()).collect::<Vec<&str>>())
            .map_err(SimulationError::EquationError)?;
        let grad_species = ode_jit
            .jacobian_wrt(&species.iter().map(|x| x.as_str()).collect::<Vec<&str>>())
            .map_err(SimulationError::EquationError)?;

        // Create buffers that can be used in other functions
        let initial_assignment_buffer = vec![0.0; initial_assignments.len()];
        let assignment_buffer = vec![0.0; assignment_rules.len()];
        let species_buffer = vec![0.0; species.len()];
        let mut params_buffer = vec![0.0; parameters.len()];

        for (idx, param) in parameters.iter().enumerate() {
            params_buffer[idx] = *params.get(param).unwrap();
        }

        // Create an input buffer we can use for all evaluations
        let buffer_size = sorted_vars.len() + 1 + parameters.len() * species.len();
        let input_buffer = vec![0.0; buffer_size];

        // Calculate the ranges for the input buffer
        let species_range = (1, 1 + species.len());
        let parameters_range = (species_range.1, species_range.1 + parameters.len());
        let assignment_rules_range = (
            parameters_range.1,
            parameters_range.1 + assignment_rules.len(),
        );
        let initial_assignments_range = (
            assignment_rules_range.1,
            assignment_rules_range.1 + initial_assignments.len(),
        );

        let system = Self {
            ode_jit,
            assignment_jit,
            initial_assignment_jit,
            var_map,
            species_mapping,
            params_mapping,
            initial_assignments_mapping,
            assignment_rules_mapping,
            sorted_vars,
            grad_params,
            grad_species,
            initial_assignment_buffer: RefCell::new(initial_assignment_buffer),
            assignment_buffer: RefCell::new(assignment_buffer),
            species_buffer: RefCell::new(species_buffer),
            params_buffer: RefCell::new(params_buffer),
            mode: RefCell::new(mode.unwrap_or(Mode::Regular)),
            input_buffer: RefCell::new(input_buffer),
            assignments_result_buffer: RefCell::new(vec![0.0; assignments.len()]),
            species_range,
            parameters_range,
            initial_assignments_range,
            assignment_rules_range,
        };

        Ok(system)
    }

    /// Integrates the ODE system over a specified time period.
    ///
    /// This method performs numerical integration of the system using the Dormand-Prince (DOPRI5) method.
    /// It handles initial conditions, parameter updates, initial assignments, and can operate in different modes
    /// including sensitivity analysis.
    ///
    /// # Arguments
    ///
    /// * `setup` - Configuration parameters for the integration including time span and tolerances
    /// * `initial_conditions` - HashMap mapping species names to their initial concentrations
    /// * `parameters` - Optional slice of parameter values to use for integration. If None, uses current values
    /// * `evaluate` - Optional vector of specific time points at which to evaluate the solution
    /// * `mode` - Optional integration mode (Regular or Sensitivity). Defaults to Regular if None
    ///
    /// # Returns
    ///
    /// Returns a Result containing either:
    /// * `Ok(T::Output)` - The integration results in the specified output format
    /// * `Err(SimulationError)` - An error that occurred during integration
    ///
    /// # Type Parameters
    ///
    /// * `T: OutputFormat` - The desired output format for the integration results
    pub fn integrate<T: OutputFormat>(
        &self,
        setup: &SimulationSetup,
        initial_conditions: HashMap<String, f64>,
        parameters: Option<&[f64]>,
        evaluate: Option<&Vec<f64>>,
        solver: impl ODEIntegrator + Copy,
        mode: Option<Mode>,
    ) -> Result<T::Output, SimulationError> {
        let solver = BasicODESolver::new(solver);
        let mode = mode.unwrap_or(Mode::Regular);
        self.mode.replace(mode.clone());

        // Create the initial conditions vector
        let initial_conditions =
            self.arrange_y0_vector(&initial_conditions, matches!(mode, Mode::Sensitivity))?;

        if let Some(params) = parameters {
            self.set_params_buffer(params);
        }

        // Evaluate the initial assignments before the integration
        // This is done because the initial assignments are not part of the state vector
        // and need to be evaluated separately
        if self.has_initial_assignments() {
            let input_vec = self.arrange_input_vector(
                setup.t0,
                &initial_conditions[..self.num_equations()],
                self.get_params_buffer().as_slice(),
                self.get_assignment_buffer().as_slice(),
                self.get_initial_assignment_buffer().as_slice(),
            );

            self.initial_assignment_jit.fun()(
                &input_vec,
                self.initial_assignment_buffer.borrow_mut().as_mut_slice(),
            );
        }

        // Solve the ODE system
        let (x_out, y_out) =
            solver.solve(self, (setup.t0, setup.t1), setup.dt, &initial_conditions)?;

        // If we want to evaluate at specific times, we need to interpolate the output to match the evaluation times
        let (x_out, y_out) = if let Some(evaluate) = evaluate {
            let interpolated_output = interpolate(&y_out, &x_out, evaluate)?;
            (evaluate.to_vec(), interpolated_output)
        } else {
            (x_out.to_vec(), y_out.to_vec())
        };

        let assignment_out: Option<StepperOutput> = if self.has_assignments() {
            Some(self.recalculate_assignments(&x_out, &y_out)?)
        } else {
            None
        };

        let output = T::create_output(
            y_out,
            assignment_out,
            Array1::from_vec(x_out.to_vec()),
            self,
            matches!(mode, Mode::Sensitivity),
        );

        Ok(output)
    }

    /// Integrates the ODE system over a list of setups in parallel.
    ///
    /// This method performs numerical integration of the system using the Dormand-Prince (DOPRI5) method.
    /// It handles initial conditions, parameter updates, initial assignments, and can operate in different modes
    /// including sensitivity analysis.
    ///
    /// # Arguments
    ///
    /// * `setups` - A slice of SimulationSetup instances defining the time span and tolerances for each integration
    /// * `initial_conditions` - A slice of HashMap instances mapping species names to their initial concentrations
    /// * `parameters` - An optional slice of parameter values to use for integration. If None, uses current values
    /// * `evaluate` - An optional slice of Vec<f64> instances defining the time points at which to evaluate the solution
    /// * `mode` - An optional Mode instance defining the integration mode (Regular or Sensitivity). Defaults to Regular if None
    ///
    /// # Returns
    ///
    /// Returns a Result containing a Vec of the integration results in the specified output format
    ///
    /// # Type Parameters
    ///
    /// * `T: OutputFormat + Send` - The desired output format for the integration results
    pub fn bulk_integrate<T: OutputFormat + Send>(
        &self,
        setups: &[SimulationSetup],
        initial_conditions: &[HashMap<String, f64>],
        parameters: Option<&[f64]>,
        evaluate: Option<&[Vec<f64>]>,
        solver: impl ODEIntegrator + Copy,
        mode: Option<Mode>,
    ) -> Result<Vec<T::Output>, SimulationError>
    where
        T::Output: Send,
    {
        let evaluates = if let Some(eval) = evaluate {
            eval.iter().map(Some).collect::<Vec<_>>()
        } else {
            vec![None; setups.len()]
        };

        let systems = (0..setups.len()).map(|_| self.clone()).collect::<Vec<_>>();
        let zipped = setups
            .iter()
            .zip(initial_conditions.iter())
            .zip(evaluates.iter())
            .collect::<Vec<_>>();

        let results = zipped
            .into_iter()
            .zip(systems)
            .map(|(((setup, initial_conditions), evaluate), system)| {
                system.integrate::<T>(
                    setup,
                    initial_conditions.clone(),
                    parameters,
                    *evaluate,
                    solver,
                    mode.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Creates a mapping between variable names and their indices in the state vector.
    ///
    /// This helper function takes a master variable mapping and a subset of variables,
    /// and creates a new mapping containing only the specified variables while preserving
    /// their original indices. This is used to maintain consistent indexing across different
    /// parts of the ODE system (species, parameters, assignments etc).
    ///
    /// # Arguments
    ///
    /// * `var_map` - A reference to the master HashMap mapping all variable names to indices.
    /// * `variables` - A slice of variable names to extract mappings for.
    ///
    /// # Returns
    ///
    /// Returns a Result containing:
    /// - Ok(HashMap<String, u32>): A new HashMap containing only the specified variables
    ///   mapped to their original indices from var_map
    /// - Err(SimulationError): If any required mapping is missing
    fn derive_mapping(
        var_map: &HashMap<String, u32>,
        variables: &[String],
    ) -> Result<HashMap<String, u32>, SimulationError> {
        let mut mapping = HashMap::new();
        for (key, index) in var_map.iter() {
            if variables.contains(key) {
                mapping.insert(key.clone(), *index);
            }
        }
        Ok(mapping)
    }

    /// Parses a collection of equations into an EquationSystem for JIT compilation.
    ///
    /// This function takes raw equation strings and converts them into a form that
    /// can be efficiently evaluated during simulation.
    ///
    /// # Arguments
    ///
    /// * `equations` - A reference to a HashMap mapping variable names to their equation strings
    /// * `var_map` - A reference to a HashMap mapping variable names to their indices
    ///
    /// # Returns
    ///
    /// Returns a Result containing either:
    /// - Ok(EquationSystem): A compiled system ready for evaluation
    /// - Err(EquationError): An error describing what went wrong during parsing/compilation
    fn parse_equations_to_systems(
        equations: &HashMap<String, String>,
        var_map: &HashMap<String, u32>,
    ) -> Result<EquationSystem, SimulationError> {
        let mut ordered_equations = equations
            .iter()
            .map(|(key, eq)| (key.clone(), eq.clone()))
            .collect::<Vec<(String, String)>>();

        ordered_equations.sort_by(|a, b| a.0.cmp(&b.0));

        let ordered_equations = ordered_equations
            .into_iter()
            .map(|(_, eq)| eq)
            .collect::<Vec<String>>();

        EquationSystem::from_var_map(ordered_equations, var_map)
            .map_err(SimulationError::EquationError)
    }

    /// Creates an input vector for equation evaluation in the correct order.
    ///
    /// This method arranges all system variables (time, species, parameters, etc.)
    /// into a single vector in the order expected by the equation evaluators.
    ///
    /// # Arguments
    ///
    /// * `t` - Current time value
    /// * `species` - Current species concentrations
    /// * `params` - Current parameter values
    /// * `assignments` - Current assignment rule values
    /// * `initial_assignments` - Current initial assignment values
    ///
    /// # Returns
    ///
    /// Returns a Vec<f64> containing all variables arranged in the correct order
    pub fn arrange_input_vector(
        &self,
        t: f64,
        species: &[f64],
        params: &[f64],
        assignments: &[f64],
        initial_assignments: &[f64],
    ) -> Vec<f64> {
        let mut input_vec = Vec::with_capacity(self.sorted_vars.len() + 1);
        input_vec.push(t);
        input_vec.extend_from_slice(species);
        input_vec.extend_from_slice(params);
        input_vec.extend_from_slice(assignments);
        input_vec.extend_from_slice(initial_assignments);
        input_vec
    }

    /// Internal method to set the input vector buffer.
    ///
    /// This method is used to set the input vector for the ODE system.
    /// It is internal to the ODESystem struct and not exposed to the public API.
    ///
    /// # Arguments
    ///
    /// * `t` - The current time
    /// * `y` - The current state
    /// * `params` - The current parameters
    /// * `assignments` - The current assignments
    /// * `initial_assignments` - The current initial assignments
    ///
    pub(crate) fn set_input_vector(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        assignments: &[f64],
        initial_assignments: &[f64],
    ) {
        let mut input_vec = self.input_buffer.borrow_mut();
        input_vec.clear();
        input_vec.push(t);
        input_vec.extend_from_slice(&y[..self.num_equations()]);
        input_vec.extend_from_slice(params);
        input_vec.extend_from_slice(assignments);
        input_vec.extend_from_slice(initial_assignments);
    }

    fn recalculate_assignments(
        &self,
        x_out: &[f64],
        y_out: &[Vec<f64>],
    ) -> Result<StepperOutput, SimulationError> {
        // Create input vectors
        let mut input_vecs = Vec::with_capacity(x_out.len());

        for (t, y) in x_out.iter().zip(y_out.iter()) {
            let input_vec = self.arrange_input_vector(
                *t,
                &y.as_slice()[..self.num_equations()],
                self.get_params_buffer().as_slice(),
                &vec![0.0; self.num_assignments()],
                self.get_initial_assignment_buffer().as_slice(),
            );
            input_vecs.push(input_vec);
        }

        // Compute the assignments in parallel
        let assignments = self.assignment_jit.eval_parallel(&input_vecs)?;

        Ok(assignments)
    }

    /// Arranges initial conditions into a vector matching the species ordering.
    ///
    /// This method takes a HashMap of initial conditions (species names mapped to their
    /// initial concentrations) and creates a vector where each species' concentration
    /// is placed at the index corresponding to its position in the system's species mapping.
    ///
    /// # Arguments
    ///
    /// * `y0` - A HashMap mapping species names to their initial concentrations
    ///
    /// # Returns
    ///
    /// Returns a Vec<f64> containing the initial concentrations arranged in the order
    /// defined by species_mapping
    ///
    /// # Panics
    ///
    /// Will panic if a species name in y0 is not found in the species_mapping
    pub fn arrange_y0_vector(
        &self,
        y0: &HashMap<String, f64>,
        use_sensitivity: bool,
    ) -> Result<Vec<f64>, SimulationError> {
        let sensitivity_dims = self.get_sensitivity_dims();

        let mut y0_vec = if use_sensitivity {
            vec![0.0; self.species_mapping.len() + sensitivity_dims]
        } else {
            vec![0.0; self.species_mapping.len()]
        };

        let mut all_species = self
            .get_species_mapping()
            .keys()
            .cloned()
            .collect::<Vec<String>>();

        all_species.sort();

        for (i, species) in all_species.iter().enumerate() {
            if let Some(value) = y0.get(species) {
                y0_vec[i] = *value;
            } else {
                return Err(SimulationError::Other(format!(
                    "Species {} not found in y0, but is in species_mapping",
                    species
                )));
            }
        }
        Ok(y0_vec)
    }

    /// Updates the parameter buffer with new values.
    ///
    /// This method allows dynamic updating of parameters during simulation.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice containing the new parameter values in the same order as sorted_vars
    pub fn set_params_buffer(&self, params: &[f64]) {
        self.params_buffer.replace(params.to_vec());
    }

    /// Returns a reference to the current parameter buffer.
    ///
    /// The parameter buffer contains the values of all parameters in the system,
    /// ordered according to sorted_vars. These parameters represent constants
    /// that can affect the behavior of the ODEs and other equations.
    ///
    /// # Returns
    ///
    /// A slice containing the current parameter values
    pub fn get_params_buffer(&self) -> Ref<Vec<f64>> {
        self.params_buffer.borrow()
    }

    /// Updates the species buffer with new values.
    ///
    /// This method allows dynamic updating of species concentrations during simulation.
    /// The species buffer stores the current concentrations of all chemical species in the system.
    ///
    /// # Arguments
    ///
    /// * `species` - A slice containing the new species values in the same order as sorted_vars
    pub fn set_species_buffer(&self, species: &[f64]) {
        self.species_buffer.replace(species.to_vec());
    }

    /// Returns a reference to the current species buffer.
    ///
    /// The species buffer contains the concentrations of all chemical species in the system,
    /// ordered according to sorted_vars.
    ///
    /// # Returns
    ///
    /// A slice containing the current species concentrations
    pub fn get_species_buffer(&self) -> Ref<Vec<f64>> {
        self.species_buffer.borrow()
    }

    /// Returns a reference to the current assignment rules buffer.
    ///
    /// The assignment buffer contains the values of all assignment rules that must be
    /// satisfied at all times during simulation. These are equations that define relationships
    /// between variables that must hold throughout the integration.
    ///
    /// # Returns
    ///
    /// A slice containing the current assignment rule values
    pub fn get_assignment_buffer(&self) -> Ref<Vec<f64>> {
        self.assignment_buffer.borrow()
    }

    /// Updates the assignment rules buffer with new values.
    ///
    /// This method allows updating the values of assignment rules during simulation.
    /// Assignment rules are equations that must be satisfied at all times.
    ///
    /// # Arguments
    ///
    /// * `assignments` - A slice containing the new assignment rule values in the same order as sorted_vars
    pub fn set_assignment_buffer(&self, assignments: &[f64]) {
        self.assignment_buffer.replace(assignments.to_vec());
    }

    /// Returns a reference to the variable mapping.
    ///
    /// The variable mapping contains the mapping between all variable names (including species,
    /// parameters, and other variables) and their indices in the system state vector. This mapping
    /// is used to track where each variable's value is stored during simulation.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping variable names to their indices
    pub fn get_var_mapping(&self) -> &HashMap<String, u32> {
        &self.var_map
    }

    /// Returns a reference to the initial assignments buffer.
    ///
    /// The initial assignments buffer contains values of equations that are only evaluated
    /// at the start of simulation (t=0). These assignments help set up the initial state
    /// of the system.
    ///
    /// # Returns
    ///
    /// A slice containing the initial assignment values
    pub fn get_initial_assignment_buffer(&self) -> Ref<Vec<f64>> {
        self.initial_assignment_buffer.borrow()
    }

    /// Updates the initial assignments buffer with new values.
    ///
    /// This method allows updating the values of initial assignments.
    /// Initial assignments are equations that are only evaluated at t=0 to help
    /// set up the initial state of the system.
    ///
    /// # Arguments
    ///
    /// * `initial_assignments` - A slice containing the new initial assignment values in the same order as sorted_vars
    pub fn set_initial_assignment_buffer(&self, initial_assignments: &[f64]) {
        self.initial_assignment_buffer
            .replace(initial_assignments.to_vec());
    }

    /// Returns a reference to the species mapping.
    ///
    /// The species mapping contains the mapping between species names and their indices
    /// in the state vector used during simulation. This mapping is used to track where
    /// each species' concentration is stored in the system state.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping species names to their indices
    pub fn get_species_mapping(&self) -> &HashMap<String, u32> {
        &self.species_mapping
    }

    /// Returns a sorted vector of species names.
    ///
    /// This method retrieves all species names from the species mapping and returns them
    /// in alphabetically sorted order. This can be useful for consistent iteration over
    /// species or displaying species in a deterministic order.
    ///
    /// # Returns
    ///
    /// A Vec<String> containing all species names in alphabetical order
    pub fn get_sorted_species(&self) -> Vec<String> {
        let mut sorted_species = self
            .species_mapping
            .keys()
            .cloned()
            .collect::<Vec<String>>();
        sorted_species.sort();
        sorted_species
    }

    /// Returns a sorted vector of assignment rule names.
    ///
    /// This method retrieves all assignment rule names from the assignment rules mapping and returns them
    /// in alphabetically sorted order. This can be useful for consistent iteration over
    /// assignment rules or displaying assignment rules in a deterministic order.
    ///
    /// # Returns
    ///
    /// A Vec<String> containing all assignment rule names in alphabetical order    
    pub fn get_sorted_assignments(&self) -> Vec<String> {
        let mut sorted_assignments = self
            .assignment_rules_mapping
            .keys()
            .cloned()
            .collect::<Vec<String>>();
        sorted_assignments.sort();
        sorted_assignments
    }

    /// Returns a reference to the parameters mapping.
    ///
    /// The parameters mapping contains the mapping between parameter names and their indices
    /// in the parameter vector. This mapping is used to track where each parameter value
    /// is stored during simulation and parameter optimization.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping parameter names to their indices
    pub fn get_params_mapping(&self) -> &HashMap<String, u32> {
        &self.params_mapping
    }

    /// Returns a sorted vector of parameter names.
    ///
    /// This method retrieves all parameter names from the parameters mapping and returns them
    /// in numerically sorted order. This can be useful for consistent iteration over
    /// parameters or displaying parameters in a deterministic order.
    ///
    /// # Returns
    ///
    /// A Vec<String> containing all parameter names in numerically sorted order
    pub fn get_sorted_params(&self) -> Vec<String> {
        let mut param_pairs: Vec<_> = self.params_mapping.iter().collect();
        param_pairs.sort_by_key(|(_, &idx)| idx);
        param_pairs
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Returns a reference to the initial assignments mapping.
    ///
    /// The initial assignments mapping contains the mapping between variable names and their
    /// indices for equations that are only evaluated at t=0. This mapping helps track
    /// where each initial assignment value is stored when setting up the initial system state.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping initial assignment variable names to their indices
    pub fn get_initial_assignments_mapping(&self) -> &HashMap<String, u32> {
        &self.initial_assignments_mapping
    }

    /// Returns a reference to the assignment rules mapping.
    ///
    /// The assignment rules mapping contains the mapping between variable names and their
    /// indices for equations that must be satisfied at all times during simulation.
    /// This mapping helps track where each assignment rule value is stored during integration.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping assignment rule variable names to their indices
    pub fn get_assignment_rules_mapping(&self) -> &HashMap<String, u32> {
        &self.assignment_rules_mapping
    }

    /// Returns the total number of sensitivity equations.
    ///
    /// This is calculated as the number of species equations multiplied by
    /// the number of parameters, since we need one sensitivity equation
    /// per species-parameter pair.
    ///
    /// # Returns
    ///
    /// The total number of sensitivity equations
    pub fn get_sensitivity_dims(&self) -> usize {
        // Equations * Parameters
        self.species_buffer.borrow().len() * self.params_buffer.borrow().len()
    }

    /// Returns the number of ODE equations in the system.
    ///
    /// This corresponds to the number of species whose concentrations
    /// are governed by differential equations.
    ///
    /// # Returns
    ///
    /// The number of ODE equations
    pub fn num_equations(&self) -> usize {
        self.species_buffer.borrow().len()
    }

    /// Returns the number of parameters in the system.
    ///
    /// # Returns
    ///
    /// The total number of model parameters
    pub fn num_parameters(&self) -> usize {
        self.params_buffer.borrow().len()
    }

    /// Returns the number of assignment rules in the system.
    ///
    /// Assignment rules are equations that must be satisfied at all times
    /// during simulation.
    ///
    /// # Returns
    ///
    /// The number of assignment rules
    pub fn num_assignments(&self) -> usize {
        self.assignment_rules_mapping.len()
    }

    /// Returns the number of initial assignments in the system.
    ///
    /// Initial assignments are equations that are only evaluated at t=0
    /// to set up the initial state.
    ///
    /// # Returns
    ///
    /// The number of initial assignments
    pub fn num_initial_assignments(&self) -> usize {
        self.initial_assignments_mapping.len()
    }

    /// Checks if the system has any assignment rules.
    ///
    /// # Returns
    ///
    /// true if the system contains assignment rules, false otherwise
    pub fn has_assignments(&self) -> bool {
        !self.assignment_rules_mapping.is_empty()
    }

    /// Checks if the system has any initial assignments.
    ///
    /// # Returns
    ///
    /// true if the system contains initial assignments, false otherwise
    pub fn has_initial_assignments(&self) -> bool {
        !self.initial_assignments_mapping.is_empty()
    }

    /// Returns the index range for assignment rules in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for assignment rules
    pub fn get_assignment_ranges(&self) -> &(usize, usize) {
        &self.assignment_rules_range
    }

    /// Returns the index range for species in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for species values
    pub fn get_species_range(&self) -> &(usize, usize) {
        &self.species_range
    }

    /// Returns the index range for parameters in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for parameter values
    pub fn get_params_range(&self) -> &(usize, usize) {
        &self.parameters_range
    }

    /// Returns the index range for initial assignments in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for initial assignment values
    pub fn get_initial_assignments_range(&self) -> &(usize, usize) {
        &self.initial_assignments_range
    }
}

impl ODEProblem for ODESystem {
    /// Defines the ODE system by computing derivatives at the current state.
    ///
    /// This method implements the system of ODEs by:
    /// 1. Building an input vector with current values
    /// 2. Evaluating assignment rules
    /// 3. Computing derivatives for each species
    ///
    /// The method is called by the ODE solver during integration.
    ///
    /// # Arguments
    ///
    /// * `t` - The current time point
    /// * `y` - The current state vector containing species concentrations
    /// * `dy` - Mutable reference to store computed derivatives
    ///
    /// # Panics
    ///
    /// May panic if equation evaluation fails (which shouldn't happen with valid equations)
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), argmin_math::Error> {
        self.set_input_vector(
            t,
            y,
            self.get_params_buffer().as_slice(),
            self.get_assignment_buffer().as_slice(),
            self.get_initial_assignment_buffer().as_slice(),
        );

        let mut input_vec = self.input_buffer.borrow_mut();
        let species_len = self.num_equations();

        if self.has_assignments() {
            let (assignments_start, assignments_end) = self.get_assignment_ranges();
            let mut assignments_result = self.assignments_result_buffer.borrow_mut();

            // Evaluate Assignment rules
            self.assignment_jit
                .eval_into(&input_vec.to_vec(), &mut *assignments_result)
                .unwrap();

            // Copy assignments to input vector
            input_vec[*assignments_start..*assignments_end].copy_from_slice(&assignments_result);

            // Copy assignments to assignment buffer
            self.assignment_buffer
                .borrow_mut()
                .copy_from_slice(&assignments_result);
        }

        // This is where we decide if we are doing a regular simulation or a sensitivity analysis
        // - In case of a sensitivity analysis, we need to calculate the right hand side of the ODEs and the sensitivities wrt parameters
        // - In case of a regular simulation, we only need to calculate the right hand side of the ODEs
        match *self.mode.borrow() {
            Mode::Sensitivity => {
                let params_len = self.params_buffer.borrow().len();
                let (species_slice, sensitivity_slice) = dy.split_at_mut(species_len);

                // Calculate species derivatives
                self.ode_jit.fun()(&input_vec, species_slice);

                // Calculate sensitivity
                let s = DMatrixView::from_slice(&y[species_len..], species_len, params_len);
                let mut ds = DMatrixViewMut::from_slice(sensitivity_slice, species_len, params_len);
                calculate_sensitivity(
                    &input_vec,
                    &self.grad_species,
                    &self.grad_params,
                    s,
                    &mut ds,
                );
            }
            Mode::Regular => {
                self.ode_jit.fun()(&input_vec, dy);
            }
        }
        Ok(())
    }
}

/// Calculates sensitivity matrix for parameter optimization.
///
/// This function computes the sensitivity matrix used in gradient-based parameter optimization.
/// It evaluates how changes in model parameters affect the system state variables.
///
/// The sensitivity equations are solved using the formula:
/// dS/dt = (∂f/∂x)S + ∂f/∂p
/// where:
/// - S is the sensitivity matrix
/// - ∂f/∂x is the Jacobian matrix of the system with respect to state variables
/// - ∂f/∂p is the Jacobian matrix with respect to parameters
///
/// # Arguments
/// * `input_vec` - Vector containing current values of state variables and parameters
/// * `dfdx` - Jacobian of the system with respect to state variables
/// * `dfdp` - Jacobian of the system with respect to parameters  
/// * `s` - Current sensitivity matrix
/// * `ds` - Output matrix for sensitivity derivatives
fn calculate_sensitivity(
    input_vec: &[f64],
    dfdx: &EquationSystem,
    dfdp: &EquationSystem,
    s: DMatrixView<f64>,
    ds: &mut DMatrixViewMut<f64>,
) {
    let input_vec = input_vec.to_vec();
    let dfdx: DMatrix<f64> = dfdx.eval_matrix(&input_vec).unwrap();
    let dfdp: DMatrix<f64> = dfdp.eval_matrix(&input_vec).unwrap();

    ds.copy_from(&(dfdx * s + dfdp));
}

impl TryFrom<EnzymeMLDocument> for ODESystem {
    type Error = SimulationError;

    fn try_from(doc: EnzymeMLDocument) -> Result<Self, Self::Error> {
        // Extract equations by type
        let odes = extract_equation_by_type(&doc.equations, EquationType::ODE);
        let assignments = extract_equation_by_type(&doc.equations, EquationType::ASSIGNMENT);
        let initial_assignments =
            extract_equation_by_type(&doc.equations, EquationType::INITIAL_ASSIGNMENT);

        // Extract parameters
        let params = doc
            .parameters
            .iter()
            .map(|p| (p.id.clone(), p.value.unwrap_or(0.0)))
            .collect::<HashMap<String, f64>>();

        Self::new(odes, initial_assignments, assignments, params, None)
    }
}

impl TryFrom<&EnzymeMLDocument> for ODESystem {
    type Error = SimulationError;

    fn try_from(doc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        // Extract equations by type
        let odes = extract_equation_by_type(&doc.equations, EquationType::ODE);
        let assignments = extract_equation_by_type(&doc.equations, EquationType::ASSIGNMENT);
        let initial_assignments =
            extract_equation_by_type(&doc.equations, EquationType::INITIAL_ASSIGNMENT);

        // Extract parameters
        let params = doc
            .parameters
            .iter()
            .map(|p| (p.id.clone(), p.value.unwrap_or(0.0)))
            .collect::<HashMap<String, f64>>();

        Self::new(odes, initial_assignments, assignments, params, None)
    }
}

impl std::fmt::Debug for ODESystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sorted vars: {:#?}", self.sorted_vars)?;
        writeln!(f, "Species mapping: {:#?}", self.species_mapping)?;
        writeln!(f, "Params mapping: {:#?}", self.params_mapping)?;
        writeln!(
            f,
            "Initial assignments mapping: {:#?}",
            self.initial_assignments_mapping
        )?;
        writeln!(
            f,
            "Assignment rules mapping: {:#?}",
            self.assignment_rules_mapping
        )?;
        Ok(())
    }
}

/// Extracts equations of a specific type from a collection of equations.
///
/// This function filters equations by type, sorts them by species ID for consistent ordering,
/// and returns just the equation strings.
///
/// # Arguments
///
/// * `equations` - A slice of Equation structs to filter
/// * `equation_type` - The EquationType to extract (e.g. Ode, AssignmentRule, etc)
///
/// # Returns
///
/// Returns a Vec<String> containing the equation strings for all equations matching the
/// specified type, sorted by species ID.
fn extract_equation_by_type(
    equations: &[Equation],
    equation_type: EquationType,
) -> HashMap<String, String> {
    equations
        .iter()
        .filter(|eq| eq.equation_type == equation_type)
        .map(|eq| (eq.species_id.clone(), eq.equation.clone()))
        .collect()
}

/// Represents the operating mode of the ODE system.
///
/// The mode determines how the system evaluates derivatives and gradients:
///
/// # Variants
///
/// * `Normal` - Standard ODE evaluation mode for regular simulation
/// * `ParameterGradient` - Calculates gradients with respect to model parameters
/// * `SpeciesGradient` - Calculates gradients with respect to species concentrations
#[derive(Clone, Debug)]
pub enum Mode {
    /// Standard ODE evaluation mode for regular simulation
    Regular,
    /// Sensitivity analysis mode
    Sensitivity,
}

#[cfg(test)]
mod tests {
    use output::MatrixResult;
    use peroxide::fuga::DP45;
    use setup::SimulationSetupBuilder;

    use super::*;
    use crate::{
        prelude::*,
        simulation::{output, setup},
    };

    fn create_enzmldoc() -> Result<EnzymeMLDocument, EnzymeMLDocumentBuilderError> {
        EnzymeMLDocumentBuilder::default()
            .name("test")
            .to_equations(
                EquationBuilder::default()
                    .species_id("substrate".to_string())
                    .equation("-v_max * substrate / (K_M + substrate)".to_string())
                    .equation_type(EquationType::ODE)
                    .build()
                    .unwrap(),
            )
            .to_parameters(
                ParameterBuilder::default()
                    .id("v_max".to_string())
                    .symbol("v_max".to_string())
                    .name("v_max".to_string())
                    .value(Some(10.0))
                    .build()
                    .unwrap(),
            )
            .to_parameters(
                ParameterBuilder::default()
                    .id("K_M".to_string())
                    .symbol("K_M".to_string())
                    .name("K_M".to_string())
                    .value(100.0)
                    .build()
                    .unwrap(),
            )
            .build()
    }

    #[test]
    fn test_simulation_with_sensitivity_to_matrix() {
        // Arrange
        let doc = create_enzmldoc().unwrap();
        let system: ODESystem = doc.try_into().unwrap();
        let dt = 0.1;
        let t0 = 0.0;
        let t1 = 5.0;

        let solver = DP45::default();

        let setup = SimulationSetupBuilder::default()
            .dt(dt)
            .t0(t0)
            .t1(t1)
            .build()
            .unwrap();

        let initial_conditions = HashMap::from([
            ("substrate".to_string(), 100.0),
            ("product".to_string(), 0.0),
        ]);

        // Act
        let result = system.integrate::<MatrixResult>(
            &setup,
            initial_conditions,
            None,
            Some(&vec![2.0, 3.0, 4.0, 5.0]),
            solver,
            Some(Mode::Sensitivity),
        );

        // Assert
        assert!(result.is_ok());

        if let Ok(MatrixResult {
            times,
            species,
            parameter_sensitivities,
            assignments: _,
        }) = result
        {
            assert_eq!(times.shape()[0], 4, "Times shape is incorrect");
            assert_eq!(species.shape()[0], 4, "Species shape is incorrect");
            assert_eq!(species.shape()[1], 1, "Species shape is incorrect");

            if let Some(parameter_sensitivities) = parameter_sensitivities {
                assert_eq!(
                    parameter_sensitivities.shape()[0],
                    4,
                    "Parameter sensitivities shape is incorrect"
                );
                assert_eq!(
                    parameter_sensitivities.shape()[1],
                    1,
                    "Parameter sensitivities shape is incorrect"
                );
                assert_eq!(
                    parameter_sensitivities.shape()[2],
                    2,
                    "Parameter sensitivities shape is incorrect"
                );
            } else {
                panic!("Parameter sensitivities are None");
            }
        } else {
            panic!("Result is not a Matrix");
        }
    }

    #[test]
    fn test_simulation_without_sensitivity_to_matrix() {
        // Arrange
        let doc = create_enzmldoc().unwrap();
        let system: ODESystem = doc.try_into().unwrap();
        let dt = 0.1;
        let t0 = 0.0;
        let t1 = 5.0;

        let solver = DP45::default();

        let setup = SimulationSetupBuilder::default()
            .dt(dt)
            .t0(t0)
            .t1(t1)
            .build()
            .unwrap();

        let initial_conditions = HashMap::from([
            ("substrate".to_string(), 100.0),
            ("product".to_string(), 0.0),
        ]);

        let result = system.integrate::<MatrixResult>(
            &setup,
            initial_conditions,
            None,
            Some(&vec![2.0, 3.0, 4.0, 5.0]),
            solver,
            None,
        );

        // Assert
        assert!(result.is_ok());

        if let Ok(MatrixResult {
            species,
            parameter_sensitivities,
            assignments: _,
            times,
        }) = result
        {
            assert_eq!(times.shape()[0], 4, "Times shape is incorrect");
            assert_eq!(species.shape()[0], 4, "Species shape is incorrect");
            assert_eq!(species.shape()[1], 1, "Species shape is incorrect");
            assert!(
                parameter_sensitivities.is_none(),
                "Parameter sensitivities should be None"
            );
        }
    }

    #[test]
    fn test_wrong_initial_conditions() {
        // Arrange
        let doc = create_enzmldoc().unwrap();
        let system: ODESystem = doc.try_into().unwrap();
        let solver = DP45::default();

        let setup = SimulationSetupBuilder::default()
            .dt(0.1)
            .t0(0.0)
            .t1(5.0)
            .build()
            .unwrap();

        let initial_conditions =
            HashMap::from([("no_sub".to_string(), 100.0), ("product".to_string(), 0.0)]);

        let result =
            system.integrate::<MatrixResult>(&setup, initial_conditions, None, None, solver, None);

        // Assert
        assert!(result.is_err(), "Result should be an error");
    }

    #[test]
    fn test_missing_species_in_initial_conditions() {
        // Arrange
        let doc = create_enzmldoc().unwrap();
        let system: ODESystem = doc.try_into().unwrap();
        let solver = DP45::default();
        let setup = SimulationSetupBuilder::default()
            .dt(0.1)
            .t0(0.0)
            .t1(5.0)
            .build()
            .unwrap();

        let initial_conditions = HashMap::from([("product".to_string(), 0.0)]);

        // Act
        let result =
            system.integrate::<MatrixResult>(&setup, initial_conditions, None, None, solver, None);

        // Assert
        assert!(result.is_err(), "Result should be an error");
    }
}
