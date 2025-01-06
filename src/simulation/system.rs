use std::collections::HashMap;
use std::error::Error;

use evalexpr::{
    Context, ContextWithMutableFunctions, ContextWithMutableVariables, EvalexprError, Function,
    HashMapContext, Node, Value,
};
use ode_solvers::*;
use rayon::prelude::*;

use crate::simulation::runner::Equations;

use super::error::SimulationError;

/// Type alias for the state vector used in the ODE system
type State = DVector<f64>;
/// Type alias for time values used in the ODE system
type Time = f64;

/// Represents an ODE system for simulating reaction networks
///
/// This struct contains all the information needed to simulate a system of ODEs,
/// including parameters, initial conditions, equations, and mappings between
/// species names and their indices in the state vector.
#[derive(Clone, Debug)]
pub(crate) struct ODESystem {
    /// Parameter values indexed by parameter name
    params: HashMap<String, f64>,
    /// Initial conditions indexed by species name
    y0: HashMap<String, f64>,
    /// System equations including ODEs and assignment rules
    pub(crate) equations: Equations,
    /// Values calculated from initial assignments
    initial_assignments: HashMap<String, f64>,
    /// Maps species names to their indices in the state vector
    pub(crate) species_mapping: HashMap<String, usize>,
    /// Parsed ASTs for ODE equations
    ode_asts: HashMap<String, Node>,
    /// Parsed ASTs for assignment rules
    assignment_asts: HashMap<String, Node>,
    /// Parsed ASTs for initial assignments
    initial_assignment_asts: HashMap<String, Node>,
}

impl ODESystem {
    /// Creates a new ODESystem instance.
    ///
    /// # Arguments
    ///
    /// * params - A hashmap of parameter names and their values.
    /// * y0 - A hashmap of initial conditions for the species.
    /// * equations - A hashmap of species names and their corresponding equations.
    ///
    /// # Returns
    ///
    /// Returns a Result containing the new ODESystem instance or an error.
    pub(crate) fn new(
        mut params: HashMap<String, f64>,
        y0: HashMap<String, f64>,
        equations: Equations,
    ) -> Result<Self, SimulationError> {
        // Create a mapping of species to index by sorting the species
        // This is necessary to map the equations to the state vector
        let mut species: Vec<String> = equations.ode.keys().map(|x| x.to_string()).collect();
        species.sort();

        let species_mapping: HashMap<String, usize> = species
            .iter()
            .enumerate()
            .map(|(i, x)| (x.to_string(), i))
            .collect();

        // Exclude parameters that are used in an assignment rule
        for (key, _) in equations.assignments.iter() {
            params.remove(key);
        }

        // Parse equations into ASTs using the helper function
        let ode_asts = Self::parse_equations_to_asts(&equations.ode)?;
        let assignment_asts = Self::parse_equations_to_asts(&equations.assignments)?;
        let initial_assignment_asts =
            Self::parse_equations_to_asts(&equations.initial_assignments)?;

        let mut system = Self {
            params,
            y0,
            equations,
            species_mapping,
            initial_assignments: HashMap::new(),
            ode_asts,
            assignment_asts,
            initial_assignment_asts,
        };

        system
            .calculate_initial_assignments()
            .map_err(|e| SimulationError::CalculateInitialAssignmentsError(e.to_string()))?;

        Ok(system)
    }

    /// Parses a collection of equations into Abstract Syntax Trees (ASTs)
    ///
    /// # Arguments
    ///
    /// * equations - A reference to a HashMap containing equation strings indexed by their identifiers
    ///
    /// # Returns
    ///
    /// Returns a Result containing a HashMap of parsed AST nodes indexed by the same identifiers,
    /// or a SimulationError if parsing fails
    fn parse_equations_to_asts(
        equations: &HashMap<String, String>,
    ) -> Result<HashMap<String, Node>, SimulationError> {
        equations
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    evalexpr::build_operator_tree(v)
                        .map_err(SimulationError::EvalExpressionError)?,
                ))
            })
            .collect()
    }

    /// Calculates initial assignments for the system
    ///
    /// This method evaluates any initial assignment rules to determine starting values
    /// for certain variables in the system. The results are stored in the initial_assignments
    /// field.
    fn calculate_initial_assignments(&mut self) -> Result<(), SimulationError> {
        let mut context = HashMapContext::new();

        // Add the initial conditions to the context
        for (species, &value) in self.y0.iter() {
            if !self.species_mapping.contains_key(species) {
                context
                    .set_value(species.into(), value.into())
                    .map_err(SimulationError::EvalExpressionError)?;
            }
        }

        // Add the dynamic species to the context
        for (species, _) in self.species_mapping.iter() {
            context
                .set_value(species.into(), self.y0[species].into())
                .map_err(SimulationError::EvalExpressionError)?;
        }

        // Add the parameters to the context
        for (key, &value) in self.params.iter() {
            context
                .set_value(key.into(), value.into())
                .map_err(SimulationError::EvalExpressionError)?;
        }

        for (symbol, ast) in self.initial_assignment_asts.iter() {
            let value = ast
                .eval_with_context(&context)
                .map_err(SimulationError::EvalExpressionError)?;

            self.initial_assignments
                .insert(symbol.clone(), value.as_number().unwrap());
        }

        Ok(())
    }

    /// Calculates values for assignment rules at a given state and time
    ///
    /// # Arguments
    ///
    /// * y - The current state vector
    /// * t - The current time
    ///
    /// # Returns
    ///
    /// Returns a Result containing a HashMap of assignment rule values indexed by variable name,
    /// or an error if evaluation fails
    pub(crate) fn calculate_assignment_rules(
        &self,
        y: &State,
        t: &Time,
    ) -> Result<HashMap<String, f64>, Box<dyn Error>> {
        let mut context = HashMapContext::new();
        let mut assignments = HashMap::new();

        // Add the time to the context
        context
            .set_value("t".into(), (*t).into())
            .map_err(SimulationError::EvalExpressionError)?;

        // Enable built-in functions and add custom log function
        context.set_builtin_functions_disabled(false).unwrap();
        context
            .set_function(
                "ln".into(),
                Function::new(|argument| {
                    if let Ok(float) = argument.as_float() {
                        Ok(Value::Float(float.ln()))
                    } else {
                        Err(EvalexprError::expected_number(argument.clone()))
                    }
                }),
            )
            .map_err(SimulationError::EvalExpressionError)?;

        // Add current species to the context
        for (species, i) in self.species_mapping.iter() {
            context
                .set_value(species.into(), y[*i].into())
                .map_err(SimulationError::EvalExpressionError)?;
        }

        // Add the parameters to the context
        for (key, &value) in self.params.iter() {
            context
                .set_value(key.into(), value.into())
                .map_err(SimulationError::EvalExpressionError)?;
        }

        // Add the initial assignments to the context
        for (key, &value) in self.initial_assignments.iter() {
            context
                .set_value(key.into(), value.into())
                .map_err(SimulationError::EvalExpressionError)?;
        }

        // Evaluate the assignment rules
        for (symbol, ast) in self.assignment_asts.iter() {
            let value = ast
                .eval_with_context(&context)
                .map_err(SimulationError::EvalExpressionError)?;

            assignments.insert(symbol.clone(), value.as_number().unwrap());
        }

        Ok(assignments)
    }
}

impl System<f64, State> for ODESystem {
    /// Defines the ODE system.
    ///
    /// This method implements the system of ODEs by evaluating the rate equations
    /// for each species at the current state and time.
    ///
    /// # Arguments
    ///
    /// * t - The current time.
    /// * y - The current state vector.
    /// * dy - The derivative of the state vector (output).
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let mut context = HashMapContext::new();
        context.set_builtin_functions_disabled(false).unwrap();

        // Add the time to the context
        context.set_value("t".into(), t.into()).unwrap();

        // Calculate assignment rules
        let assignments = self
            .calculate_assignment_rules(y, &t)
            .expect("Could not calculate assignment rules");

        // Add all values to context
        for (key, &value) in assignments.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        for (species, i) in self.species_mapping.iter() {
            context.set_value(species.into(), y[*i].into()).unwrap();
        }

        for (key, &value) in self.params.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        for (key, &value) in self.initial_assignments.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Pre-allocate a vector to store results
        let results: Vec<(usize, f64)> = self
            .species_mapping
            .par_iter()
            .map(|(species, &i)| {
                let ast = self.ode_asts.get(species).expect("AST not found");
                let value = ast.eval_with_context(&context).unwrap_or_else(|_| {
                    panic!("Could not evaluate equation for species {}", species)
                });
                (i, value.as_number().unwrap())
            })
            .collect();

        // Update dy with collected results
        for (i, value) in results {
            dy[i] = value;
        }
    }
}

impl ODESystem {
    /// Returns the initial state vector.
    ///
    /// Constructs a state vector from the initial conditions, ensuring species
    /// are in the correct order according to the species mapping.
    ///
    /// # Returns
    ///
    /// Returns the initial state vector as a DVector.
    pub fn get_y0_vector(&self) -> State {
        let n_dynamic_species = self.equations.ode.len();
        let mut y0_vec: Vec<f64> = vec![0.0; n_dynamic_species];
        for (species, i) in self.species_mapping.iter() {
            y0_vec[*i] = *self.y0.get(species).unwrap();
        }

        DVector::from_vec(y0_vec)
    }
}
