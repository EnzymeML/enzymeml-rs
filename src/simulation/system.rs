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
    cell::UnsafeCell,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use evalexpr_jit::prelude::*;
use nalgebra::{DMatrixView, DMatrixViewMut};
use ndarray::Array1;
use peroxide::fuga::{BasicODESolver, ODEIntegrator, ODEProblem, ODESolver};
use rayon::prelude::*;

use crate::{
    create_stoich_eq_jit,
    prelude::{EnzymeMLDocument, Equation, EquationType, Reaction},
};

use super::{
    error::SimulationError, interpolation::interpolate, output::OutputFormat,
    stoich::derive_stoichiometry_matrix, SimulationSetup,
};

/// Type alias for a function that evaluates an equation system.
pub type EvalFunction = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync + 'static>;

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
pub struct ODESystem {
    // Variable mapping
    var_map: HashMap<String, u32>,
    species_mapping: HashMap<String, u32>,
    params_mapping: HashMap<String, u32>,
    initial_assignments_mapping: HashMap<String, u32>,
    assignment_rules_mapping: HashMap<String, u32>,
    constants_mapping: HashMap<String, u32>,
    sorted_vars: Vec<String>,
    stoich: bool,

    // Equation systems for all three types of equations
    ode_jit: EvalFunction,
    assignment_jit: EvalFunction,
    initial_assignment_jit: EvalFunction,

    // Gradient functions
    grad_params: EquationSystem,
    grad_species: EquationSystem,

    // Default values for parameters - these will be used to initialize contexts
    default_params: Vec<f64>,

    /// Starts and ends of the parts of the input vector
    species_range: (usize, usize),
    parameters_range: (usize, usize),
    initial_assignments_range: (usize, usize),
    assignment_rules_range: (usize, usize),
    constants_range: (usize, usize),
}

/// Context object that contains mutable state for an ODESystem.
///
/// This allows thread-safe execution by holding mutable state that can be passed
/// explicitly to simulation methods.
#[derive(Clone)]
pub struct ODESystemContext {
    // Buffers used for intermediate calculations
    pub(crate) initial_assignment_buffer: Vec<f64>,
    pub(crate) assignment_buffer: Vec<f64>,
    pub(crate) params_buffer: Vec<f64>,
    pub(crate) constants_buffer: Vec<f64>,

    // Mode
    pub(crate) mode: Mode,

    // Buffer for system evaluation input vector
    pub(crate) input_buffer: Vec<f64>,
    // Buffer for assignment rule evaluations
    pub(crate) assignments_result_buffer: Vec<f64>,

    // Pre-computed ranges for performance (avoid repeated calculations)
    pub(crate) params_range_cache: (usize, usize),
    pub(crate) assignments_range_cache: (usize, usize),
    pub(crate) initial_assignments_range_cache: (usize, usize),
    pub(crate) constants_range_cache: (usize, usize),
}

/// Wrapper struct that implements ODEProblem for an ODESystem with its context
struct ODEProblemWithContext<'a> {
    system: &'a ODESystem,
    context: UnsafeCell<&'a mut ODESystemContext>,
}

impl<'a> ODEProblemWithContext<'a> {
    fn new(system: &'a ODESystem, context: &'a mut ODESystemContext) -> Self {
        Self {
            system,
            context: UnsafeCell::new(context),
        }
    }
}

impl ODEProblem for ODEProblemWithContext<'_> {
    #[inline(always)]
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), argmin_math::Error> {
        // This is safe because we know ODEProblemWithContext has exclusive access to the context
        let context = unsafe { &mut *self.context.get() };

        let species_len = self.system.num_equations();

        self.system
            .populate_input_buffer_optimized(context, t, &y[..species_len]);

        if self.system.has_assignments() {
            // Evaluate Assignment rules
            (self.system.assignment_jit)(
                &context.input_buffer,
                &mut context.assignments_result_buffer,
            );

            // Use cached ranges for better performance
            let assignments_start = context.assignments_range_cache.0;
            let assignments_end = assignments_start + context.assignments_result_buffer.len();

            unsafe {
                // Use unsafe copy for maximum performance in this hot path
                context
                    .input_buffer
                    .get_unchecked_mut(assignments_start..assignments_end)
                    .copy_from_slice(&context.assignments_result_buffer);
                context
                    .assignment_buffer
                    .copy_from_slice(&context.assignments_result_buffer);
            }
        }

        // Eliminate branch by using function-specific logic
        // This is much faster than matching on every call
        match context.mode {
            Mode::Sensitivity => {
                self.rhs_sensitivity_mode(context, &y[species_len..], dy, species_len)
            }
            Mode::Regular => {
                // Simple case - just evaluate ODEs
                (self.system.ode_jit)(&context.input_buffer, dy);
            }
        }

        Ok(())
    }
}

/// Additional methods for ODEProblemWithContext
impl ODEProblemWithContext<'_> {
    /// Optimized sensitivity mode computation - separated to reduce branching overhead
    #[inline(always)]
    fn rhs_sensitivity_mode(
        &self,
        context: &mut ODESystemContext,
        y_sensitivity: &[f64],
        dy: &mut [f64],
        species_len: usize,
    ) {
        let params_len = context.params_buffer.len();
        let (species_slice, sensitivity_slice) = dy.split_at_mut(species_len);

        // Calculate species derivatives
        (self.system.ode_jit)(&context.input_buffer, species_slice);

        // Calculate sensitivity
        let s = DMatrixView::from_slice(y_sensitivity, species_len, params_len);
        let mut ds = DMatrixViewMut::from_slice(sensitivity_slice, species_len, params_len);

        calculate_sensitivity(
            &context.input_buffer,
            &self.system.grad_species,
            &self.system.grad_params,
            s,
            &mut ds,
            self.system.stoich,
        );
    }
}

/// Intermediate struct to hold all components of the system state during initialization.
///
/// This struct contains all the mappings, vectors, and ranges needed to create an ODE system,
/// making the code more maintainable and self-documenting than using large tuples.
struct SystemState {
    species: Vec<String>,
    parameters: Vec<String>,
    sorted_vars: Vec<String>,
    var_map: HashMap<String, u32>,
    species_mapping: HashMap<String, u32>,
    params_mapping: HashMap<String, u32>,
    initial_assignments_mapping: HashMap<String, u32>,
    assignment_rules_mapping: HashMap<String, u32>,
    constants_mapping: HashMap<String, u32>,
    default_params: Vec<f64>,
    species_range: (usize, usize),
    parameters_range: (usize, usize),
    assignment_rules_range: (usize, usize),
    initial_assignments_range: (usize, usize),
    constants_range: (usize, usize),
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
    pub(crate) fn from_odes(
        odes: HashMap<String, String>,
        initial_assignments: HashMap<String, String>,
        assignments: HashMap<String, String>,
        params: HashMap<String, f64>,
        constants: Vec<String>,
        _mode: Option<Mode>,
    ) -> Result<Self, SimulationError> {
        // First, set up the common system state
        let state = Self::prepare_system_state(
            odes.keys().map(|x| x.to_string()).collect(),
            params,
            initial_assignments.keys().map(|x| x.to_string()).collect(),
            assignments.keys().map(|x| x.to_string()).collect(),
            constants,
        )?;

        // Parse equations into EquationSystem instances
        let ode_jit = Self::parse_equations_to_systems(&odes, &state.var_map)?;
        let assignment_jit = Self::parse_equations_to_systems(&assignments, &state.var_map)?;
        let initial_assignment_jit =
            Self::parse_equations_to_systems(&initial_assignments, &state.var_map)?;

        // Get the gradient wrt to parameters and species
        let grad_params = ode_jit
            .jacobian_wrt(
                &state
                    .parameters
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>(),
            )
            .map_err(SimulationError::EquationError)?;
        let grad_species = ode_jit
            .jacobian_wrt(
                &state
                    .species
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>(),
            )
            .map_err(SimulationError::EquationError)?;

        let system = Self {
            ode_jit: Arc::clone(ode_jit.fun()),
            assignment_jit: Arc::clone(assignment_jit.fun()),
            initial_assignment_jit: Arc::clone(initial_assignment_jit.fun()),
            var_map: state.var_map,
            species_mapping: state.species_mapping,
            params_mapping: state.params_mapping,
            constants_mapping: state.constants_mapping,
            initial_assignments_mapping: state.initial_assignments_mapping,
            assignment_rules_mapping: state.assignment_rules_mapping,
            sorted_vars: state.sorted_vars,
            grad_params,
            grad_species,
            default_params: state.default_params,
            species_range: state.species_range,
            parameters_range: state.parameters_range,
            initial_assignments_range: state.initial_assignments_range,
            assignment_rules_range: state.assignment_rules_range,
            constants_range: state.constants_range,
            stoich: false,
        };

        Ok(system)
    }

    pub fn from_reactions(
        reactions: &[Reaction],
        initial_assignments: HashMap<String, String>,
        assignments: HashMap<String, String>,
        params: HashMap<String, f64>,
        constants: Vec<String>,
    ) -> Result<Self, SimulationError> {
        // Derive the stoichiometry matrix
        let (stoich, species) = derive_stoichiometry_matrix(reactions)?;

        // Sort the reactions by id
        let mut kinetic_laws = HashMap::new();
        for reaction in reactions {
            let id = reaction.id.clone();
            if let Some(kinetic_law) = reaction.kinetic_law.clone() {
                kinetic_laws.insert(id, kinetic_law.equation);
            }
        }

        // Prepare the system state
        let state = Self::prepare_system_state(
            species,
            params,
            initial_assignments.keys().map(|x| x.to_string()).collect(),
            assignments.keys().map(|x| x.to_string()).collect(),
            constants,
        )?;

        // Create the ODESystem
        let laws_jit = Self::parse_equations_to_systems(&kinetic_laws, &state.var_map)?;
        let assignment_jit = Self::parse_equations_to_systems(&assignments, &state.var_map)?;
        let initial_assignment_jit =
            Self::parse_equations_to_systems(&initial_assignments, &state.var_map)?;

        // Get the gradient wrt to parameters and species
        let grad_params = laws_jit
            .jacobian_wrt(
                &state
                    .parameters
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>(),
            )
            .map_err(SimulationError::EquationError)?;
        let grad_species = laws_jit
            .jacobian_wrt(
                &state
                    .species
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>(),
            )
            .map_err(SimulationError::EquationError)?;

        // Get the non-zero indices of the stoichiometry matrix
        let non_zero_indices = {
            let mut indices = Vec::new();
            for i in 0..stoich.nrows() {
                for j in 0..stoich.ncols() {
                    if stoich[(i, j)] != 0.0 {
                        indices.push((i, j));
                    }
                }
            }

            indices
        };

        // Create an anonymous function which takes the stoichiometry matrix and the input vector
        // and returns the derivatives
        let law_output_size = stoich.ncols();
        let ode_jit: EvalFunction =
            create_stoich_eq_jit!(laws_jit, stoich, non_zero_indices, law_output_size);

        let system = Self {
            ode_jit,
            assignment_jit: Arc::clone(assignment_jit.fun()),
            initial_assignment_jit: Arc::clone(initial_assignment_jit.fun()),
            var_map: state.var_map,
            species_mapping: state.species_mapping,
            params_mapping: state.params_mapping,
            constants_mapping: state.constants_mapping,
            initial_assignments_mapping: state.initial_assignments_mapping,
            assignment_rules_mapping: state.assignment_rules_mapping,
            sorted_vars: state.sorted_vars,
            grad_params,
            grad_species,
            default_params: state.default_params,
            species_range: state.species_range,
            parameters_range: state.parameters_range,
            initial_assignments_range: state.initial_assignments_range,
            assignment_rules_range: state.assignment_rules_range,
            constants_range: state.constants_range,
            stoich: true,
        };

        Ok(system)
    }

    /// Prepares the common system state used by both `from_odes` and `from_reactions`.
    ///
    /// This function handles the housekeeping steps of setting up all the mappings and
    /// data structures needed for the ODE system.
    ///
    /// # Arguments
    ///
    /// * `species_ids` - A vector of species IDs
    /// * `params` - A hashmap of parameter IDs to values
    /// * `initial_assignment_ids` - A vector of initial assignment IDs
    /// * `assignment_ids` - A vector of assignment rule IDs
    /// * `constants` - A vector of constant IDs
    ///
    /// # Returns
    ///
    /// A Result containing a SystemState struct with all the prepared system state components
    fn prepare_system_state(
        species_ids: Vec<String>,
        params: HashMap<String, f64>,
        initial_assignment_ids: Vec<String>,
        assignment_ids: Vec<String>,
        mut constants: Vec<String>,
    ) -> Result<SystemState, SimulationError> {
        // Extract all variables and sort them
        let mut species: Vec<String> = species_ids;
        let mut parameters: Vec<String> = params.keys().map(|x| x.to_string()).collect();
        let mut initial_assignment_rules: Vec<String> = initial_assignment_ids;
        let mut assignment_rules: Vec<String> = assignment_ids;

        species.sort();
        parameters.sort();
        initial_assignment_rules.sort();
        assignment_rules.sort();
        constants.sort();

        // Create the sorted list of all variables
        let mut sorted_vars = vec!["t".to_string()];
        sorted_vars.extend(species.clone());
        sorted_vars.extend(parameters.clone());
        sorted_vars.extend(assignment_rules.clone());
        sorted_vars.extend(initial_assignment_rules.clone());
        sorted_vars.extend(constants.clone());

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
        let constants_mapping = Self::derive_mapping(&var_map, &constants)?;

        // Create default parameter values
        let mut default_params = vec![0.0; parameters.len()];
        for (idx, param) in parameters.iter().enumerate() {
            default_params[idx] = *params.get(param).unwrap();
        }

        // Calculate the ranges for the input buffer
        let species_range = (1, 1 + species.len());
        let parameters_range = (species_range.1, species_range.1 + parameters.len());
        let assignment_rules_range = (
            parameters_range.1,
            parameters_range.1 + assignment_rules.len(),
        );
        let initial_assignments_range = (
            assignment_rules_range.1,
            assignment_rules_range.1 + initial_assignment_rules.len(),
        );
        let constants_range = (
            initial_assignments_range.1,
            initial_assignments_range.1 + constants.len(),
        );

        Ok(SystemState {
            species,
            parameters,
            sorted_vars,
            var_map,
            species_mapping,
            params_mapping,
            initial_assignments_mapping,
            assignment_rules_mapping,
            constants_mapping,
            default_params,
            species_range,
            parameters_range,
            assignment_rules_range,
            initial_assignments_range,
            constants_range,
        })
    }

    /// Integrates the ODE system over a specified time period.
    ///
    /// This method creates a default context and then calls integrate_with_context.
    ///
    /// # Arguments
    ///
    /// * `setup` - Configuration parameters for the integration
    /// * `initial_conditions` - HashMap mapping species names to their initial concentrations
    /// * `parameters` - Optional slice of parameter values to use for integration
    /// * `evaluate` - Optional vector of specific time points at which to evaluate the solution
    /// * `solver` - The ODE solver to use for integration
    /// * `mode` - Optional integration mode (Regular or Sensitivity)
    ///
    /// # Returns
    ///
    /// Returns a Result containing the integration results in the specified output format
    #[inline]
    pub fn integrate<T: OutputFormat>(
        &self,
        setup: &SimulationSetup,
        initial_conditions: &HashMap<String, f64>,
        parameters: Option<&[f64]>,
        evaluate: Option<&Vec<f64>>,
        solver: impl ODEIntegrator + Copy,
        mode: Option<Mode>,
    ) -> Result<T::Output, SimulationError> {
        // Create a default context, possibly with the specified mode
        let mut context = if let Some(m) = mode.clone() {
            self.create_context_with_mode(m)
        } else {
            self.create_context()
        };

        // Use integrate_internal directly with explicit type parameter
        self.integrate_internal::<T>(
            &mut context,
            setup,
            initial_conditions,
            parameters,
            evaluate,
            solver,
            mode,
        )
    }

    /// Integrates the ODE system over a list of setups in parallel.
    ///
    /// This method creates a default context and then calls bulk_integrate_with_context.
    ///
    /// # Arguments
    ///
    /// * `setups` - A slice of SimulationSetup instances for each integration
    /// * `initial_conditions` - A slice of HashMap instances mapping species names to initial concentrations
    /// * `parameters` - Optional slice of parameter values to use for integration
    /// * `evaluate` - Optional slice of time points at which to evaluate solutions
    /// * `solver` - The ODE solver to use for integration
    /// * `mode` - Optional integration mode (Regular or Sensitivity)
    ///
    /// # Returns
    ///
    /// Returns a Result containing a Vec of integration results
    pub fn bulk_integrate<T: OutputFormat + Send>(
        &self,
        setups: &[SimulationSetup],
        initial_conditions: &[HashMap<String, f64>],
        parameters: Option<&[f64]>,
        evaluate: Option<&[Vec<f64>]>,
        solver: impl ODEIntegrator + Copy + Send + Sync,
        mode: Option<Mode>,
    ) -> Result<Vec<T::Output>, SimulationError>
    where
        T::Output: Send,
    {
        // Create a default context, possibly with the specified mode
        let context = if let Some(m) = mode.clone() {
            self.create_context_with_mode(m)
        } else {
            self.create_context()
        };

        // Call the context version
        self.bulk_integrate_with_context::<T>(
            &context,
            setups,
            initial_conditions,
            parameters,
            evaluate,
            solver,
            mode,
        )
    }

    // Internal implementation for integrate_with_context
    #[allow(clippy::too_many_arguments)]
    fn integrate_internal<T: OutputFormat>(
        &self,
        context: &mut ODESystemContext,
        setup: &SimulationSetup,
        initial_conditions: &HashMap<String, f64>,
        parameters: Option<&[f64]>,
        evaluate: Option<&Vec<f64>>,
        solver: impl ODEIntegrator + Copy,
        mode: Option<Mode>,
    ) -> Result<T::Output, SimulationError> {
        let solver = BasicODESolver::new(solver);

        // Update mode if provided
        if let Some(m) = mode {
            context.mode = m;
        }

        // Fill the constants buffer from the initial conditions
        let constants = self.arrange_constants_buffer(initial_conditions)?;
        context.constants_buffer = constants;

        // Create the initial conditions vector
        let initial_conditions = self.arrange_y0_vector(
            initial_conditions,
            matches!(context.mode, Mode::Sensitivity),
        )?;

        if let Some(params) = parameters {
            // Avoid allocation by copying directly into the existing buffer
            if context.params_buffer.len() == params.len() {
                context.params_buffer.copy_from_slice(params);
            } else {
                context.params_buffer = params.to_vec();
            }
        }

        // Evaluate the initial assignments before the integration
        if self.has_initial_assignments() {
            // OPTIMIZATION: Reuse the input buffer to avoid allocation
            if context.input_buffer.len() < self.sorted_vars.len() + 1 {
                context.input_buffer.resize(self.sorted_vars.len() + 1, 0.0);
            }

            context.input_buffer[0] = setup.t0;

            // Copy values directly into buffer for better performance
            // Species values
            let species_len = self.num_equations();
            context.input_buffer[1..(species_len + 1)]
                .copy_from_slice(&initial_conditions[..species_len]);

            // Parameters
            let params_start = self.species_range.1;
            let params_len = context.params_buffer.len();
            for i in 0..params_len {
                context.input_buffer[params_start + i] = context.params_buffer[i];
            }

            // Assignments
            let assignments_start = self.parameters_range.1;
            let assignments_len = context.assignment_buffer.len();
            for i in 0..assignments_len {
                context.input_buffer[assignments_start + i] = context.assignment_buffer[i];
            }

            // Initial assignments
            let initial_assignments_start = self.assignment_rules_range.1;
            let initial_assignments_len = context.initial_assignment_buffer.len();
            for i in 0..initial_assignments_len {
                context.input_buffer[initial_assignments_start + i] =
                    context.initial_assignment_buffer[i];
            }

            // Constants
            let constants_start = self.initial_assignments_range.1;
            let constants_len = context.constants_buffer.len();
            for i in 0..constants_len {
                context.input_buffer[constants_start + i] = context.constants_buffer[i];
            }

            // Ensure result buffer is sized correctly
            if context.initial_assignment_buffer.len() != self.num_initial_assignments() {
                context
                    .initial_assignment_buffer
                    .resize(self.num_initial_assignments(), 0.0);
            }

            // Compute results directly into the context buffer
            (self.initial_assignment_jit)(
                &context.input_buffer,
                &mut context.initial_assignment_buffer,
            );
        }

        // Create a wrapper that implements ODEProblem
        let ode_problem = ODEProblemWithContext::new(self, context);

        // Solve the ODE system
        let (x_out, y_out) = solver.solve(
            &ode_problem,
            (setup.t0, setup.t1),
            setup.dt,
            &initial_conditions,
        )?;

        // If we want to evaluate at specific times, interpolate the output
        let (x_out, y_out) = if let Some(evaluate) = evaluate {
            let interpolated_output = interpolate(&y_out, &x_out, evaluate)?;
            (evaluate.to_vec(), interpolated_output)
        } else {
            (x_out.to_vec(), y_out.to_vec())
        };

        let assignment_out: Option<StepperOutput> = if self.has_assignments() {
            Some(self.recalculate_assignments_with_context(context, &x_out, &y_out)?)
        } else {
            None
        };

        // OPTIMIZATION: Convert to Array1 without intermediate vec
        let times = Array1::from_vec(x_out);

        let output = T::create_output(
            y_out,
            assignment_out,
            times,
            self,
            matches!(context.mode, Mode::Sensitivity),
        );

        Ok(output)
    }

    // Internal implementation for bulk_integrate_with_context
    #[allow(clippy::too_many_arguments)]
    fn bulk_integrate_internal<T: OutputFormat + Send>(
        &self,
        context: &ODESystemContext,
        setups: &[SimulationSetup],
        initial_conditions: &[HashMap<String, f64>],
        parameters: Option<&[f64]>,
        evaluate: Option<&[Vec<f64>]>,
        solver: impl ODEIntegrator + Copy + Send + Sync,
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

        let zipped = setups
            .iter()
            .zip(initial_conditions.iter())
            .zip(evaluates.iter())
            .collect::<Vec<_>>();

        // Clone self to ensure thread safety
        let self_clone = self.clone();

        let results = zipped
            .into_par_iter()
            .map(|((setup, initial_conditions), evaluate)| {
                // Create a new context for each thread based on the template
                let mut thread_context = context.clone();

                // Use the cloned self to avoid sharing self across threads
                self_clone.integrate_internal::<T>(
                    &mut thread_context,
                    setup,
                    initial_conditions,
                    parameters,
                    *evaluate,
                    solver,
                    mode.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    // Version of recalculate_assignments that uses a context
    fn recalculate_assignments_with_context(
        &self,
        context: &ODESystemContext,
        x_out: &[f64],
        y_out: &[Vec<f64>],
    ) -> Result<StepperOutput, SimulationError> {
        let n_timepoints = x_out.len();
        let n_assignments = self.num_assignments();

        // Pre-allocate the output to avoid resizing
        let mut assignments = Vec::with_capacity(n_timepoints);

        // Reuse this buffer for each evaluation to avoid allocations
        let mut input_buffer = context.input_buffer.clone();

        for (t_idx, (t, y)) in x_out.iter().zip(y_out.iter()).enumerate() {
            // Resize only once on the first iteration
            if t_idx == 0 && input_buffer.len() < self.sorted_vars.len() + 1 {
                input_buffer.resize(self.sorted_vars.len() + 1, 0.0);
            }

            // Time
            input_buffer[0] = *t;

            // Species - use direct indexing for better cache performance
            let species_len = self.num_equations();
            input_buffer[1..(species_len + 1)].copy_from_slice(&y[..species_len]);

            // Copy other values from context buffers
            // Parameters
            let params_start = self.species_range.1;
            let params_len = context.params_buffer.len();
            input_buffer[params_start..(params_len + params_start)]
                .copy_from_slice(&context.params_buffer[..params_len]);

            // Assignments
            let assignments_start = self.parameters_range.1;
            let assignments_len = context.assignment_buffer.len();
            input_buffer[assignments_start..(assignments_len + assignments_start)]
                .copy_from_slice(&context.assignment_buffer[..assignments_len]);

            // Initial assignments
            let initial_assignments_start = self.assignment_rules_range.1;
            let initial_assignments_len = context.initial_assignment_buffer.len();
            input_buffer
                [initial_assignments_start..(initial_assignments_len + initial_assignments_start)]
                .copy_from_slice(&context.initial_assignment_buffer[..initial_assignments_len]);

            // Constants
            let constants_start = self.initial_assignments_range.1;
            let constants_len = context.constants_buffer.len();
            input_buffer[constants_start..(constants_len + constants_start)]
                .copy_from_slice(&context.constants_buffer[..constants_len]);

            // Evaluate and push results
            let mut result = vec![0.0; n_assignments];
            (self.assignment_jit)(&input_buffer, &mut result);
            assignments.push(result);
        }

        Ok(assignments)
    }

    /// Integrates the ODE system over a specified time period using a provided context.
    ///
    /// This method performs numerical integration of the system using the provided solver.
    /// It uses the state from the provided context for buffers and mode settings.
    ///
    /// # Arguments
    ///
    /// * `context` - The context containing mutable state for this integration
    /// * `setup` - Configuration parameters for the integration
    /// * `initial_conditions` - HashMap mapping species names to their initial concentrations
    /// * `parameters` - Optional slice of parameter values to use for integration
    /// * `evaluate` - Optional vector of specific time points at which to evaluate the solution
    /// * `solver` - The ODE solver to use for integration
    /// * `mode` - Optional integration mode. If provided, updates the context's mode
    ///
    /// # Returns
    ///
    /// Returns a Result containing the integration results in the specified output format
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_with_context<T: OutputFormat>(
        &self,
        context: &mut ODESystemContext,
        setup: &SimulationSetup,
        initial_conditions: &HashMap<String, f64>,
        parameters: Option<&[f64]>,
        evaluate: Option<&Vec<f64>>,
        solver: impl ODEIntegrator + Copy,
        mode: Option<Mode>,
    ) -> Result<T::Output, SimulationError> {
        // Use a wrapper to call the existing integrate method with the context
        self.integrate_internal::<T>(
            context,
            setup,
            initial_conditions,
            parameters,
            evaluate,
            solver,
            mode,
        )
    }

    /// Integrates the ODE system over a list of setups in parallel using a provided context.
    ///
    /// This method performs parallel numerical integration for multiple setups.
    /// It creates a clone of the context for each parallel task.
    ///
    /// # Arguments
    ///
    /// * `context` - The context containing mutable state to use as a template
    /// * `setups` - A slice of SimulationSetup instances for each integration
    /// * `initial_conditions` - A slice of HashMap instances mapping species names to initial concentrations
    /// * `parameters` - Optional slice of parameter values to use for integration
    /// * `evaluate` - Optional slice of time points at which to evaluate solutions
    /// * `solver` - The ODE solver to use for integration
    /// * `mode` - Optional integration mode. If provided, updates each context's mode
    ///
    /// # Returns
    ///
    /// Returns a Result containing a Vec of integration results
    #[allow(clippy::too_many_arguments)]
    pub fn bulk_integrate_with_context<T: OutputFormat + Send>(
        &self,
        context: &ODESystemContext,
        setups: &[SimulationSetup],
        initial_conditions: &[HashMap<String, f64>],
        parameters: Option<&[f64]>,
        evaluate: Option<&[Vec<f64>]>,
        solver: impl ODEIntegrator + Copy + Send + Sync,
        mode: Option<Mode>,
    ) -> Result<Vec<T::Output>, SimulationError>
    where
        T::Output: Send,
    {
        // Use a wrapper to call the existing bulk_integrate method with the context
        self.bulk_integrate_internal::<T>(
            context,
            setups,
            initial_conditions,
            parameters,
            evaluate,
            solver,
            mode,
        )
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
    #[inline(always)]
    pub fn arrange_input_vector(
        &self,
        t: f64,
        species: &[f64],
        params: &[f64],
        assignments: &[f64],
        initial_assignments: &[f64],
        constants: &[f64],
    ) -> Vec<f64> {
        let mut input_vec = Vec::with_capacity(self.sorted_vars.len() + 1);
        input_vec.push(t);
        input_vec.extend_from_slice(species);
        input_vec.extend_from_slice(params);
        input_vec.extend_from_slice(assignments);
        input_vec.extend_from_slice(initial_assignments);
        input_vec.extend_from_slice(constants);
        input_vec
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
                    "Species {species} not found in y0, but is in species_mapping"
                )));
            }
        }
        Ok(y0_vec)
    }

    /// Arranges constants into a vector matching the constants mapping.
    ///
    /// This method takes a HashMap of initial conditions (constant names mapped to their
    /// initial concentrations) and creates a vector where each constant's concentration
    /// is placed at the index corresponding to its position in the system's constants mapping.
    ///
    /// # Arguments
    ///
    /// * `initial_conditions` - A HashMap mapping constant names to their initial concentrations
    ///
    /// # Returns
    ///
    /// Returns a Vec<f64> containing the constants arranged in the order defined by constants_mapping
    ///
    /// # Panics
    ///
    /// Will panic if a constant name in initial_conditions is not found in the constants_mapping
    fn arrange_constants_buffer(
        &self,
        initial_conditions: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, SimulationError> {
        let mut constants = vec![0.0; self.get_constants_mapping().len()];
        let mut missing_constants = vec![];

        for (idx, constant) in self.get_sorted_constants().iter().enumerate() {
            if let Some(value) = initial_conditions.get(constant) {
                constants[idx] = *value;
            } else {
                missing_constants.push(constant.clone());
            }
        }

        if !missing_constants.is_empty() {
            return Err(SimulationError::MissingConstants(missing_constants));
        }

        Ok(constants)
    }

    /// Updates the parameter buffer with new values.
    ///
    /// This method allows dynamic updating of parameters during simulation.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice containing the new parameter values in the same order as sorted_vars
    /// * `context` - The context whose parameters should be updated
    pub fn set_params(&self, params: &[f64], context: &mut ODESystemContext) {
        context.params_buffer.copy_from_slice(params);
    }

    /// Returns the number of ODE equations in the system.
    ///
    /// This corresponds to the number of species whose concentrations
    /// are governed by differential equations.
    ///
    /// # Returns
    ///
    /// The number of ODE equations
    #[inline(always)]
    pub fn num_equations(&self) -> usize {
        self.species_mapping.len()
    }

    /// Returns the number of parameters in the system.
    ///
    /// # Returns
    ///
    /// The total number of model parameters
    #[inline(always)]
    pub fn num_parameters(&self) -> usize {
        self.params_mapping.len()
    }

    /// Returns the number of assignment rules in the system.
    ///
    /// Assignment rules are equations that must be satisfied at all times
    /// during simulation.
    ///
    /// # Returns
    ///
    /// The number of assignment rules
    #[inline(always)]
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
    #[inline(always)]
    pub fn num_initial_assignments(&self) -> usize {
        self.initial_assignments_mapping.len()
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
    #[inline(always)]
    pub fn get_sensitivity_dims(&self) -> usize {
        // Equations * Parameters
        self.species_mapping.len() * self.params_mapping.len()
    }

    /// Checks if the system has any assignment rules.
    ///
    /// # Returns
    ///
    /// true if the system contains assignment rules, false otherwise
    #[inline(always)]
    pub fn has_assignments(&self) -> bool {
        !self.assignment_rules_mapping.is_empty()
    }

    /// Checks if the system has any initial assignments.
    ///
    /// # Returns
    ///
    /// true if the system contains initial assignments, false otherwise
    #[inline(always)]
    pub fn has_initial_assignments(&self) -> bool {
        !self.initial_assignments_mapping.is_empty()
    }

    /// Returns the index range for assignment rules in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for assignment rules
    #[inline(always)]
    pub fn get_assignment_ranges(&self) -> &(usize, usize) {
        &self.assignment_rules_range
    }

    /// Returns the index range for species in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for species values
    #[inline(always)]
    pub fn get_species_range(&self) -> &(usize, usize) {
        &self.species_range
    }

    /// Returns the index range for parameters in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for parameter values
    #[inline(always)]
    pub fn get_params_range(&self) -> &(usize, usize) {
        &self.parameters_range
    }

    /// Returns the index range for constants in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for constant values
    #[inline(always)]
    pub fn get_constants_range(&self) -> &(usize, usize) {
        &self.constants_range
    }

    /// Returns the index range for initial assignments in the input vector.
    ///
    /// # Returns
    ///
    /// A tuple containing the start and end indices for initial assignment values
    #[inline(always)]
    pub fn get_initial_assignments_range(&self) -> &(usize, usize) {
        &self.initial_assignments_range
    }

    /// Creates a new context for this ODESystem.
    ///
    /// This method allocates all necessary buffers based on the current state
    /// of the system.
    ///
    /// # Returns
    ///
    /// A new ODESystemContext initialized with default values
    pub fn create_context(&self) -> ODESystemContext {
        let mut context = ODESystemContext {
            initial_assignment_buffer: Vec::with_capacity(self.num_initial_assignments()),
            assignment_buffer: Vec::with_capacity(self.num_assignments()),
            params_buffer: self.default_params.clone(),
            constants_buffer: Vec::with_capacity(self.constants_mapping.len()),
            mode: Mode::Regular,
            input_buffer: Vec::with_capacity(self.sorted_vars.len() + 1),
            assignments_result_buffer: Vec::with_capacity(self.num_assignments()),
            // Pre-compute ranges for performance
            params_range_cache: self.parameters_range,
            assignments_range_cache: self.assignment_rules_range,
            initial_assignments_range_cache: self.initial_assignments_range,
            constants_range_cache: self.constants_range,
        };

        // Pre-allocate all buffers for maximum performance
        self.pre_allocate_context_buffers(&mut context, None);

        context
    }

    /// Creates a context with the specified mode.
    ///
    /// # Arguments
    ///
    /// * `mode` - The mode to use for simulation
    ///
    /// # Returns
    ///
    /// A new ODESystemContext initialized with default values and the specified mode
    pub fn create_context_with_mode(&self, mode: Mode) -> ODESystemContext {
        let mut context = self.create_context();
        context.mode = mode;
        context
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

    /// Returns a vector of parameter names.
    ///
    /// # Returns
    ///
    /// A vector of parameter names
    pub fn get_params(&self) -> Vec<String> {
        self.params_mapping.keys().cloned().collect()
    }

    /// Returns a reference to the constants mapping.
    ///
    /// The constants mapping contains the mapping between constant names and their indices
    /// in the constant vector. This mapping is used to track where each constant value
    /// is stored during simulation and parameter optimization.
    ///
    /// # Returns
    ///
    /// A reference to the HashMap mapping constant names to their indices
    pub fn get_constants_mapping(&self) -> &HashMap<String, u32> {
        &self.constants_mapping
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

    /// Returns a sorted vector of constant names.
    ///
    /// This method retrieves all constant names from the constants mapping and returns them
    /// in numerically sorted order. This can be useful for consistent iteration over
    /// constants or displaying constants in a deterministic order.
    ///
    /// # Returns
    ///
    /// A Vec<String> containing all constant names in numerically sorted order
    pub fn get_sorted_constants(&self) -> Vec<String> {
        let mut constant_pairs: Vec<_> = self.constants_mapping.iter().collect();
        constant_pairs.sort_by_key(|(_, &idx)| idx);
        constant_pairs
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Pre-allocates all buffers in the context to avoid reallocations during simulation.
    ///
    /// This method can significantly improve performance by ensuring that all buffers
    /// are properly sized before running simulations, avoiding costly reallocations
    /// during the simulation process.
    ///
    /// # Arguments
    ///
    /// * `context` - The context to pre-allocate buffers for
    /// * `max_size` - Optional maximum size to allocate for temporary buffers
    pub fn pre_allocate_context_buffers(
        &self,
        context: &mut ODESystemContext,
        max_size: Option<usize>,
    ) {
        // Determine buffer sizes
        let params_len = self.num_parameters();
        let assignments_len = self.num_assignments();
        let initial_assignments_len = self.num_initial_assignments();
        let constants_len = self.constants_mapping.len();

        // Add extra space for temporary buffers to avoid reallocations
        let max_buffer_size = max_size.unwrap_or(20);

        // Resize input buffer to hold all variables plus some extra space
        let min_input_size = self.sorted_vars.len() + 1;
        let input_size = std::cmp::max(min_input_size, max_buffer_size);

        if context.input_buffer.len() < input_size {
            context.input_buffer.resize(input_size, 0.0);
        }

        // Ensure parameter buffer is sized correctly
        if context.params_buffer.len() != params_len {
            context.params_buffer.resize(params_len, 0.0);
            // Copy default parameter values
            context.params_buffer.copy_from_slice(&self.default_params);
        }

        // Ensure assignment buffer is sized correctly
        if context.assignment_buffer.len() != assignments_len {
            context.assignment_buffer.resize(assignments_len, 0.0);
        }

        // Ensure assignment result buffer is sized correctly
        if context.assignments_result_buffer.len() != assignments_len {
            context
                .assignments_result_buffer
                .resize(assignments_len, 0.0);
        }

        // Ensure initial assignment buffer is sized correctly
        if context.initial_assignment_buffer.len() != initial_assignments_len {
            context
                .initial_assignment_buffer
                .resize(initial_assignments_len, 0.0);
        }

        // Ensure constants buffer is sized correctly
        if context.constants_buffer.len() != constants_len {
            context.constants_buffer.resize(constants_len, 0.0);
        }
    }

    /// Optimized method to populate the input buffer with all system variables
    /// This method minimizes memory operations and improves cache locality
    #[inline(always)]
    fn populate_input_buffer_optimized(
        &self,
        context: &mut ODESystemContext,
        t: f64,
        species: &[f64],
    ) {
        let total_size = self.sorted_vars.len() + 1;
        if context.input_buffer.len() < total_size {
            context.input_buffer.resize(total_size, 0.0);
        }

        // Time
        unsafe {
            *context.input_buffer.get_unchecked_mut(0) = t;
        }

        // Species - use cached ranges for better performance
        let species_end = 1 + species.len();
        unsafe {
            context
                .input_buffer
                .get_unchecked_mut(1..species_end)
                .copy_from_slice(species);
        }

        // OPTIMIZATION: Use cached ranges and unsafe operations for maximum performance
        // This creates better memory access patterns and eliminates bounds checking

        // Parameters
        if !context.params_buffer.is_empty() {
            let params_start = context.params_range_cache.0;
            let params_end = params_start + context.params_buffer.len();
            unsafe {
                context
                    .input_buffer
                    .get_unchecked_mut(params_start..params_end)
                    .copy_from_slice(&context.params_buffer);
            }
        }

        // Assignments
        if !context.assignment_buffer.is_empty() {
            let assignments_start = context.assignments_range_cache.0;
            let assignments_end = assignments_start + context.assignment_buffer.len();
            unsafe {
                context
                    .input_buffer
                    .get_unchecked_mut(assignments_start..assignments_end)
                    .copy_from_slice(&context.assignment_buffer);
            }
        }

        // Initial assignments
        if !context.initial_assignment_buffer.is_empty() {
            let initial_assignments_start = context.initial_assignments_range_cache.0;
            let initial_assignments_end =
                initial_assignments_start + context.initial_assignment_buffer.len();
            unsafe {
                context
                    .input_buffer
                    .get_unchecked_mut(initial_assignments_start..initial_assignments_end)
                    .copy_from_slice(&context.initial_assignment_buffer);
            }
        }

        // Constants
        if !context.constants_buffer.is_empty() {
            let constants_start = context.constants_range_cache.0;
            let constants_end = constants_start + context.constants_buffer.len();
            unsafe {
                context
                    .input_buffer
                    .get_unchecked_mut(constants_start..constants_end)
                    .copy_from_slice(&context.constants_buffer);
            }
        }
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
#[inline(always)]
fn calculate_sensitivity(
    input_vec: &[f64],
    dfdx: &EquationSystem,
    dfdp: &EquationSystem,
    s: DMatrixView<f64>,
    ds: &mut DMatrixViewMut<f64>,
    stoich: bool,
) {
    if stoich {
        unimplemented!("Sensitivity analysis with stoichiometry matrix is not supported yet");
    }

    // OPTIMIZATION: Create a single copy and reuse it for both evaluations
    // This eliminates the need for two separate to_vec() calls
    let input_copy = input_vec.to_vec();

    // Evaluate both Jacobians with the same input copy
    let dfdx_result: nalgebra::DMatrix<f64> = dfdx.eval_matrix(&input_copy).unwrap();
    let dfdp_result: nalgebra::DMatrix<f64> = dfdp.eval_matrix(&input_copy).unwrap();

    // Compute directly into ds, avoiding temporary allocations
    dfdx_result.mul_to(&s, ds); // ds = dfdx * s
    *ds += dfdp_result; // ds += dfdp
}

impl TryFrom<EnzymeMLDocument> for ODESystem {
    type Error = SimulationError;

    fn try_from(doc: EnzymeMLDocument) -> Result<Self, Self::Error> {
        ODESystem::try_from(&doc)
    }
}

impl TryFrom<&EnzymeMLDocument> for ODESystem {
    type Error = SimulationError;

    fn try_from(doc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        let has_kinetic_laws = doc.reactions.iter().any(|r| r.kinetic_law.is_some());
        let has_odes = doc
            .equations
            .iter()
            .any(|e| e.equation_type == EquationType::Ode);

        if has_kinetic_laws && has_odes {
            return Err(SimulationError::InvalidInput(
                "Cannot have both kinetic laws and ODEs in the same model".to_string(),
            ));
        }

        let assignments = extract_equation_by_type(&doc.equations, EquationType::Assignment);
        let initial_assignments =
            extract_equation_by_type(&doc.equations, EquationType::InitialAssignment);

        let constants = collect_constants(doc);
        let params = doc
            .parameters
            .iter()
            .map(|p| (p.id.clone(), p.value.unwrap_or(0.0)))
            .collect::<HashMap<String, f64>>();

        if has_kinetic_laws {
            Self::from_reactions(
                &doc.reactions,
                initial_assignments,
                assignments,
                params,
                constants,
            )
        } else {
            Self::from_odes(
                extract_equation_by_type(&doc.equations, EquationType::Ode),
                initial_assignments,
                assignments,
                params,
                constants,
                None,
            )
        }
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
        writeln!(f, "Constants mapping: {:#?}", self.constants_mapping)?;
        Ok(())
    }
}

/// Collects species that appear in any equation.
///
/// This function iterates through all species in the document and checks if they appear in any equation.
/// It returns a Vec<String> containing the species that appear in any equation.
///
/// # Arguments
///
/// * `doc` - The EnzymeMLDocument to collect constants from
///
/// # Returns
///
/// A Vec<String> containing the species that appear in any equation
fn collect_constants(doc: &EnzymeMLDocument) -> Vec<String> {
    let all_species = doc
        .small_molecules
        .iter()
        .map(|m| m.id.clone())
        .chain(doc.proteins.iter().map(|p| p.id.clone()))
        .chain(doc.complexes.iter().map(|c| c.id.clone()))
        .collect::<HashSet<String>>();

    all_species
        .into_iter()
        .filter(|species| {
            doc.equations
                .iter()
                .filter(|equation| equation.species_id == *species)
                .count() == 0 && // if species id is same, do not proceed
                doc.equations
                    .iter()
                    .any(|equation| equation.equation.contains(species))
        })
        .collect()
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

impl Clone for ODESystem {
    /// Clones the ODESystem
    ///
    /// # Returns
    ///
    /// A new ODESystem with the same fields as the original
    fn clone(&self) -> Self {
        Self {
            // Clone all fields
            var_map: self.var_map.clone(),
            species_mapping: self.species_mapping.clone(),
            params_mapping: self.params_mapping.clone(),
            initial_assignments_mapping: self.initial_assignments_mapping.clone(),
            assignment_rules_mapping: self.assignment_rules_mapping.clone(),
            constants_mapping: self.constants_mapping.clone(),
            sorted_vars: self.sorted_vars.clone(),
            ode_jit: Arc::clone(&self.ode_jit),
            assignment_jit: Arc::clone(&self.assignment_jit),
            initial_assignment_jit: Arc::clone(&self.initial_assignment_jit),
            grad_params: self.grad_params.clone(),
            grad_species: self.grad_species.clone(),
            default_params: self.default_params.clone(),
            stoich: self.stoich,

            // Copy the range tuples
            species_range: self.species_range,
            parameters_range: self.parameters_range,
            initial_assignments_range: self.initial_assignments_range,
            assignment_rules_range: self.assignment_rules_range,
            constants_range: self.constants_range,
        }
    }
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
                    .equation_type(EquationType::Ode)
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
            &initial_conditions,
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
            &initial_conditions,
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
            system.integrate::<MatrixResult>(&setup, &initial_conditions, None, None, solver, None);

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
            system.integrate::<MatrixResult>(&setup, &initial_conditions, None, None, solver, None);

        // Assert
        assert!(result.is_err(), "Result should be an error");
    }

    #[test]
    fn test_from_reactions() {
        // Arrange
        let doc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let system: ODESystem = (&doc).try_into().unwrap();
        let setup = SimulationSetupBuilder::default()
            .dt(1.0)
            .t0(0.0)
            .t1(2.0)
            .build()
            .expect("Failed to build simulation setup");

        let first_measurement = doc.measurements.first().unwrap();
        let initial_conditions: InitialCondition = first_measurement.into();

        let result = system
            .integrate::<SimulationResult>(
                &setup,
                &initial_conditions,
                None,                // We could also dynamically set new parameters
                None,                // We could also provide specific time points to extract
                RK4,                 // We could also use a different solver
                Some(Mode::Regular), // We could also use a different mode (e.g. Sensitivity)
            )
            .expect("Simulation failed");

        // Assert
        assert_eq!(result.time.len(), 3, "Times length is incorrect");
        assert_eq!(result.species.len(), 4, "Species length is incorrect");
    }
}
