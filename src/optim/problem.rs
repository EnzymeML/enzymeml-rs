use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use peroxide::fuga::ODEIntegrator;
use sha2::{Digest, Sha256};

use crate::prelude::{Measurement, ODESystem, ObjectiveFunction};
use crate::utils::measurement_not_empty;
use crate::{
    prelude::{EnzymeMLDocument, SimulationSetup},
    simulation::init_cond::InitialCondition,
};

use super::error::OptimizeError;
use super::transformation::Transformation;

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
/// * `transformations` - Optional parameter transformations to enforce constraints
/// * `objective` - Objective function to optimize
#[derive(Debug, Clone)]
pub struct Problem<S: ODEIntegrator + Copy, L: ObjectiveFunction> {
    /// EnzymeML document containing the model definition, parameters and experimental data
    doc: EnzymeMLDocument,
    /// Optional user-provided initial conditions, otherwise derived from the document
    initials: Vec<InitialCondition>,
    /// Configuration for numerical integration of the model
    simulation_setup: Vec<SimulationSetup>,
    /// Optional parameter transformations to enforce constraints
    transformations: Vec<Transformation>,
    /// Objective function to optimize
    objective: L,
    /// Jitted ODE system
    ode_system: ODESystem,
    /// Observable species indices
    observable_species: Vec<usize>,
    /// Solver
    solver: S,

    // Buffer for measurement data
    measurement_buffer: Array2<f64>,
    n_points: usize,

    // Buffers for evaluation times
    evaluation_times: Vec<Vec<f64>>,

    // Fixed parameters
    pub(crate) fixed_params: Array1<f64>,
}

impl<S: ODEIntegrator + Copy, L: ObjectiveFunction> Problem<S, L> {
    /// Creates a new optimization problem from an EnzymeML document
    ///
    /// # Arguments
    /// * `doc` - EnzymeML document containing the model definition, parameters and experimental data
    /// * `objective` - Objective function to optimize
    /// * `dt` - Time step for numerical integration
    pub fn new(
        enzmldoc: &EnzymeMLDocument,
        objective: L,
        solver: S,
        dt: Option<f64>,
        transformations: Option<Vec<Transformation>>,
        fixed_params: impl Into<Option<Vec<String>>>,
    ) -> Result<Self, OptimizeError> {
        // Cloning the enzymeml document to avoid modifying the original document
        // We might need to add transformations, which will be used for initial assignments
        // and will be applied to the parameter vector before optimization
        let mut doc = enzmldoc.clone();
        doc.measurements.retain(measurement_not_empty);

        // Extract initial conditions and simulation setups from the document
        let initials: Vec<InitialCondition> = (&doc).into();
        let mut simulation_setup: Vec<SimulationSetup> = (&doc).try_into()?;

        // Add transformations to the document
        let transformations = transformations.unwrap_or_default();
        for transformation in transformations.iter() {
            transformation.add_to_enzmldoc(&mut doc)?;
        }

        // Adjust the integration settings for all setups
        let dt = dt.unwrap_or(0.1f64);

        for setup in simulation_setup.iter_mut() {
            setup.dt = dt;
        }

        // JIT the ODE system
        let ode_system = ODESystem::try_from(&doc)?;

        // Collect evaluation times and measurement data
        let evaluation_times = Self::get_evaluation_times(&doc)?;
        let measurement_buffer = Array2::try_from(&doc)?;
        let n_points = measurement_buffer.shape()[0] * measurement_buffer.shape()[1];

        // Convert observable species to indices
        let species_order = ode_system.get_sorted_species();
        let observable_species: Vec<usize> = Self::get_observable_species(&doc)?
            .iter()
            .filter_map(|s| species_order.iter().position(|sp| sp == s))
            .collect();

        // Get fixed parameters
        let fixed_params = fixed_params.into();
        let mut fixed_params_array: Array1<f64> = Array1::ones(doc.parameters.len());

        if let Some(fixed_params) = fixed_params {
            for (i, param) in ode_system.get_sorted_params().iter().enumerate() {
                if fixed_params.contains(param) {
                    fixed_params_array[i] = 0.0;
                }
            }
        }

        Ok(Self {
            doc: enzmldoc.clone(),
            initials,
            solver,
            simulation_setup,
            transformations,
            objective,
            ode_system,
            measurement_buffer,
            evaluation_times,
            n_points,
            observable_species,
            fixed_params: fixed_params_array,
        })
    }

    /// Collects evaluation time points from all measurements in the document
    ///
    /// This function iterates through all measurements in the document and extracts their time points,
    /// storing them in the Problem's evaluation_times vector.
    ///
    /// # Arguments
    /// * `enzmldoc` - The EnzymeML document to collect time points from
    ///
    /// # Returns
    /// * `Result<(), OptimizeError>` - Ok(()) if time points were successfully collected
    ///
    /// # Errors
    /// Returns `OptimizeError::NonHomogenousTimes` if:
    /// * Any measurement has inconsistent time points across its species
    /// * Time points cannot be extracted from a measurement
    fn get_evaluation_times(enzmldoc: &EnzymeMLDocument) -> Result<Vec<Vec<f64>>, OptimizeError> {
        let mut times = vec![];
        for measurement in enzmldoc.measurements.iter() {
            if let Ok(time) = Self::get_measurement_time(measurement) {
                times.push(time.clone());
            } else {
                return Err(OptimizeError::NonHomogenousTimes);
            }
        }

        Ok(times)
    }

    /// Gets the time points from a measurement by finding the first species with time data
    ///
    /// # Arguments
    /// * `measurement` - The measurement to extract time points from
    ///
    /// # Returns
    /// * `Result<&Vec<f64>, OptimizeError>` - Reference to the vector of time points if found
    ///
    /// # Errors
    /// Returns `OptimizeError::NoTimePoints` if no species in the measurement has time data
    fn get_measurement_time(measurement: &Measurement) -> Result<&Vec<f64>, OptimizeError> {
        Self::homogenous_times(measurement)?;
        measurement
            .species_data
            .iter()
            .find_map(|s| (!s.time.is_empty()).then_some(&s.time))
            .ok_or(OptimizeError::NoTimePoints)
    }

    /// Checks if all time vectors in a measurement have identical values
    ///
    /// This function verifies that all species in a measurement have the same time points
    /// by hashing each time vector and checking that there is only one unique hash.
    ///
    /// # Arguments
    /// * `measurement` - The measurement to check for homogeneous time points
    ///
    /// # Returns
    /// * `Result<(), OptimizeError>` - Ok(()) if all time vectors are identical,
    ///   or an error if time vectors differ or are missing
    ///
    /// # Errors
    /// Returns:
    /// * `OptimizeError::NoTimePoints` if no time points are found
    /// * `OptimizeError::NonHomogenousTimes` if time vectors differ between species
    fn homogenous_times(measurement: &Measurement) -> Result<(), OptimizeError> {
        let times = Self::collect_times(measurement);
        if times.is_empty() {
            return Err(OptimizeError::NoTimePoints);
        }

        // Collect unique hashes into a HashSet
        let unique_hashes: HashSet<_> = times
            .iter()
            .map(|time_vec| {
                let mut hasher = Sha256::new();
                for t in time_vec.iter() {
                    hasher.update(t.to_le_bytes());
                }
                hasher.finalize()
            })
            .collect();

        // If all time vectors are identical, there will be only one unique hash
        if unique_hashes.len() != 1 {
            return Err(OptimizeError::NonHomogenousTimes);
        }

        Ok(())
    }

    /// Collects all time vectors from a measurement's species data
    ///
    /// Extracts time point vectors from each species in the measurement,
    /// filtering out any species that don't have time data.
    ///
    /// # Arguments
    /// * `measurement` - The measurement to collect time points from
    ///
    /// # Returns
    /// * `Vec<&Vec<f64>>` - Vector of references to time point vectors
    fn collect_times(measurement: &Measurement) -> Vec<&Vec<f64>> {
        measurement
            .species_data
            .iter()
            .filter_map(|s| (!s.time.is_empty()).then_some(&s.time))
            .collect()
    }

    /// Gets the observable species from the document
    ///
    /// # Arguments
    /// * `enzmldoc` - The EnzymeML document to extract observable species from
    ///
    /// # Returns
    /// * `Result<Vec<String>, OptimizeError>` - Vector of observable species
    fn get_observable_species(enzmldoc: &EnzymeMLDocument) -> Result<Vec<String>, OptimizeError> {
        // Extract sets of observable species from each measurement
        let observable_sets: Vec<HashSet<String>> = enzmldoc
            .measurements
            .iter()
            .map(|measurement| {
                measurement
                    .species_data
                    .iter()
                    .filter(|species| !species.data.is_empty())
                    .map(|species| species.species_id.clone())
                    .collect()
            })
            .collect();

        // Handle case of no measurements
        if observable_sets.is_empty() {
            return Err(OptimizeError::NonHomogenousObservations);
        }

        // Check if all measurements have the same observable species
        let first_set = &observable_sets[0];
        if !observable_sets.iter().all(|set| set == first_set) {
            return Err(OptimizeError::NonHomogenousObservations);
        }

        // Convert the set to a sorted vector for consistent ordering
        Ok(first_set.iter().cloned().collect())
    }

    /// Applies transformations to a parameter vector
    ///
    /// # Arguments
    /// * `param_vec` - The parameter vector to apply transformations to
    ///
    /// # Returns
    /// * `Result<Vec<f64>, OptimizeError>` - The transformed parameter vector
    ///
    /// # Errors
    /// Returns `OptimizeError::UnknownParameter` if an unknown parameter is encountered
    pub fn apply_transformations(&self, param_vec: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let mut transformed_params = param_vec.to_vec();
        let param_order = self.ode_system.get_sorted_params();

        // Create a map of parameter name to transformation
        let transform_map: HashMap<_, _> = self
            .transformations
            .iter()
            .map(|t| (t.symbol(), t))
            .collect();

        // Apply transformations in one pass
        for (i, param) in param_order.iter().enumerate() {
            if let Some(transformation) = transform_map.get(param) {
                transformed_params[i] = transformation.apply_back(transformed_params[i]);
            }
        }

        Ok(transformed_params)
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
        self.doc
            .measurements
            .iter()
            .flat_map(|m| &m.species_data)
            .filter(|s| !s.data.is_empty())
            .map(|s| s.data.len())
            .sum()
    }

    /// Returns a reference to the underlying EnzymeML document
    ///
    /// # Returns
    /// * `&EnzymeMLDocument` - Reference to the EnzymeML document containing the model definition
    pub fn enzmldoc(&self) -> &EnzymeMLDocument {
        &self.doc
    }

    /// Returns a reference to the underlying EnzymeML document
    ///
    /// # Returns
    /// * `&EnzymeMLDocument` - Reference to the EnzymeML document containing the model definition
    pub fn enzmldoc_mut(&mut self) -> &mut EnzymeMLDocument {
        &mut self.doc
    }

    /// Returns a reference to the ODE system representation of the model
    ///
    /// # Returns
    /// * `&ODESystem` - Reference to the ODE system used for numerical integration
    pub fn ode_system(&self) -> &ODESystem {
        &self.ode_system
    }

    /// Returns a clone of the solver
    ///
    /// # Returns
    /// * `S` - Clone of the solver used for numerical integration
    pub fn solver(&self) -> S {
        self.solver
    }

    /// Returns a reference to the vector of simulation setups
    ///
    /// # Returns
    /// * `&Vec<SimulationSetup>` - Reference to simulation configurations for each measurement
    pub fn simulation_setup(&self) -> &Vec<SimulationSetup> {
        &self.simulation_setup
    }

    /// Returns a reference to the vector of initial conditions
    ///
    /// # Returns
    /// * `&Vec<InitialCondition>` - Reference to initial concentrations for each simulation
    pub fn initials(&self) -> &Vec<InitialCondition> {
        &self.initials
    }

    /// Returns a reference to the vector of parameter transformations
    ///
    /// # Returns
    /// * `&Vec<Transformation>` - Reference to parameter transformations applied in the optimization
    pub fn transformations(&self) -> &Vec<Transformation> {
        &self.transformations
    }

    /// Returns a reference to the objective function used for optimization
    ///
    /// # Returns
    /// * `&LossFunction` - Reference to the loss function for computing optimization cost
    pub fn objective(&self) -> &L {
        &self.objective
    }

    /// Returns a reference to the observable species indices
    ///
    /// # Returns
    /// * `&Vec<usize>` - Reference to the observable species indices
    pub fn observable_species(&self) -> &Vec<usize> {
        &self.observable_species
    }

    /// Returns a reference to the measurement buffer
    ///
    /// # Returns
    /// * `&Array2<f64>` - Reference to the measurement buffer
    pub fn measurement_buffer(&self) -> &Array2<f64> {
        &self.measurement_buffer
    }

    /// Returns the total number of data points across all measurements
    ///
    /// Counts the number of data points by summing up the length of data arrays
    /// for each species in each measurement that has data.
    ///
    /// # Returns
    /// * `usize` - Total number of data points
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Returns a reference to the vector of evaluation times
    ///
    /// # Returns
    /// * `&Vec<Vec<f64>>` - Reference to the vector of evaluation times
    pub fn evaluation_times(&self) -> &Vec<Vec<f64>> {
        &self.evaluation_times
    }

    /// Fixes a parameter
    ///
    /// # Arguments
    /// * `param` - The parameter to fix
    ///
    /// # Returns
    /// * `Result<(), OptimizeError>` - Ok(()) if the parameter was fixed
    pub fn fix_param(&mut self, param: &str) -> Result<(), OptimizeError> {
        let index = self
            .ode_system
            .get_sorted_params()
            .iter()
            .position(|p| p == param)
            .ok_or(OptimizeError::UnknownParameter(param.to_string()))?;
        self.fixed_params[index] = 0.0;
        Ok(())
    }

    /// Unfixes a parameter
    ///
    /// # Arguments
    /// * `param` - The parameter to unfix
    ///
    /// # Returns
    pub fn unfix_param(&mut self, param: &str) -> Result<(), OptimizeError> {
        let index = self
            .ode_system
            .get_sorted_params()
            .iter()
            .position(|p| p == param)
            .ok_or(OptimizeError::UnknownParameter(param.to_string()))?;
        self.fixed_params[index] = 1.0;
        Ok(())
    }

    pub fn fixed_params(&self) -> &Array1<f64> {
        &self.fixed_params
    }
}

pub struct ProblemBuilder<S: ODEIntegrator + Copy, L: ObjectiveFunction> {
    doc: EnzymeMLDocument,
    objective: L,
    dt: f64,
    transformations: Vec<Transformation>,
    fixed_params: Vec<String>,
    solver: S,
}

impl<S: ODEIntegrator + Copy, L: ObjectiveFunction> ProblemBuilder<S, L> {
    /// Creates a new ProblemBuilder with default settings
    ///
    /// # Arguments
    /// * `enzmldoc` - EnzymeML document containing model definition and data
    ///
    /// # Returns
    /// A new ProblemBuilder instance with default settings:
    /// - dt = 0.1
    /// - no transformations
    pub fn new(enzmldoc: &EnzymeMLDocument, solver: S, objective: L) -> Self {
        Self {
            doc: enzmldoc.clone(),
            objective,
            dt: 0.1,
            transformations: vec![],
            fixed_params: vec![],
            solver,
        }
    }

    /// Sets the time step for numerical integration
    ///
    /// # Arguments
    /// * `dt` - Time step size
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Sets multiple parameter transformations
    ///
    /// # Arguments
    /// * `transformations` - Vector of transformations to apply
    pub fn transformations(mut self, transformations: Vec<Transformation>) -> Self {
        self.transformations = transformations;
        self
    }

    /// Adds a single parameter transformation
    ///
    /// # Arguments
    /// * `transformation` - Transformation to add
    pub fn transform(mut self, transformation: Transformation) -> Self {
        self.transformations.push(transformation);
        self
    }

    /// Adds a fixed parameter
    ///
    /// # Arguments
    /// * `param` - Parameter to add
    pub fn fixed_param(mut self, param: String) -> Self {
        self.fixed_params.push(param);
        self
    }

    /// Sets the fixed parameters
    ///
    /// # Arguments
    /// * `fixed_params` - Vector of fixed parameters
    pub fn fixed_params(mut self, fixed_params: Vec<String>) -> Self {
        self.fixed_params = fixed_params;
        self
    }

    /// Builds the Problem instance with the configured settings
    ///
    /// # Returns
    /// Result containing either the constructed Problem or an OptimizeError
    pub fn build(self) -> Result<Problem<S, L>, OptimizeError> {
        let transformations = if self.transformations.is_empty() {
            None
        } else {
            Some(self.transformations)
        };
        Problem::new(
            &self.doc,
            self.objective,
            self.solver,
            Some(self.dt),
            transformations,
            Some(self.fixed_params),
        )
    }
}
