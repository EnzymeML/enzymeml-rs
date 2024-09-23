use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};

use evalexpr::{eval_with_context, Context, ContextWithMutableVariables, HashMapContext};
use ode_solvers::*;
use rayon::prelude::*;

use crate::simulation::runner::Equations;

type State = DVector<f64>;
type Time = f64;

#[derive(Clone, Debug)]
pub(crate) struct ODESystem {
    params: HashMap<String, f64>,
    y0: HashMap<String, f64>,
    pub(crate) equations: Equations,
    initial_assignments: HashMap<String, f64>,
    pub(crate) species_mapping: HashMap<String, usize>,
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
    ) -> Result<Self, Box<dyn Error>> {
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

        let mut system = Self {
            params,
            y0,
            equations,
            species_mapping,
            initial_assignments: HashMap::new(),
        };

        system.calculate_initial_assignments();

        Ok(system)
    }

    fn calculate_initial_assignments(&mut self) {
        let mut context = HashMapContext::new();

        // Add the initial conditions to the context
        for (species, &value) in self.y0.iter() {
            if !self.species_mapping.contains_key(species) {
                context.set_value(species.into(), value.into()).unwrap();
            }
        }

        // Add the dynamic species to the context
        for (species, _) in self.species_mapping.iter() {
            context
                .set_value(species.into(), self.y0[species].into())
                .unwrap();
        }

        // Add the parameters to the context
        for (key, &value) in self.params.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        for (symbol, eq) in self.equations.initial_assignments.iter() {
            let value = eval_with_context(eq, &context.clone()).unwrap_or_else(|_| panic!("Could not evaluate equation for species {}",
                symbol));

            self.initial_assignments
                .insert(symbol.clone(), value.as_number().unwrap());
        }
    }

    pub(crate) fn calculate_assignment_rules(
        &self,
        y: &State,
        t: &Time,
    ) -> Result<HashMap<String, f64>, Box<dyn Error>> {
        let mut context = HashMapContext::new();
        let mut assignments = HashMap::new();

        // Add the time to the context
        context.set_value("t".into(), (*t).into()).unwrap();

        // Add current species to the context
        for (species, i) in self.species_mapping.iter() {
            context.set_value(species.into(), y[*i].into()).unwrap();
        }

        // Add the parameters to the context
        for (key, &value) in self.params.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Add the initial assignments to the context
        for (key, &value) in self.initial_assignments.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Evaluate the assignment rules
        for (symbol, eq) in self.equations.assignments.iter() {
            let value = eval_with_context(eq, &context.clone()).unwrap_or_else(|_| panic!("Could not evaluate equation for species {}",
                symbol));

            assignments.insert(symbol.clone(), value.as_number().unwrap());
        }

        Ok(assignments)
    }
}

impl System<f64, State> for ODESystem {
    /// Defines the ODE system.
    ///
    /// # Arguments
    ///
    /// * t - The current time.
    /// * y - The current state vector.
    /// * dy - The derivative of the state vector.
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        // Create an intermediate collection wrapped in an Arc<Mutex<_>>
        let mut context = HashMapContext::new();
        context.set_builtin_functions_disabled(false).unwrap();

        let y = Arc::new(y.clone());

        // Add the time to the context
        context.set_value("t".into(), t.into()).unwrap();

        // Calculate assignment rules
        let assignments = self
            .calculate_assignment_rules(&y, &t)
            .expect("Could not calculate assignment rules");

        // Add the assignment rules to the context
        for (key, &value) in assignments.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Add current species to the context
        for (species, i) in self.species_mapping.iter() {
            context.set_value(species.into(), y[*i].into()).unwrap();
        }

        // Add the parameters to the context
        for (key, &value) in self.params.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Add the initial assignments to the context
        for (key, &value) in self.initial_assignments.iter() {
            context.set_value(key.into(), value.into()).unwrap();
        }

        // Wrap `dy` in an Arc<Mutex<_>> to allow safe concurrent modification
        let dy = Arc::new(Mutex::new(dy));

        // Use a parallel iterator to evaluate equations concurrently
        self.species_mapping.par_iter().for_each(|(species, &i)| {
            let eq = self.equations.ode.get(species).expect("Equation not found");

            // Evaluate the equation
            let value = eval_with_context(eq, &context.clone()).unwrap_or_else(|_| panic!("Could not evaluate equation for species {}",
                species));

            // Lock the `dy` for safe mutation
            let mut dy = dy.lock().unwrap();
            dy[i] = value.as_number().unwrap();
        });
    }
}

impl ODESystem {
    /// Returns the initial state vector.
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
