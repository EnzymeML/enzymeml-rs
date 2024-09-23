use std::collections::HashMap;
use std::error::Error;

use derive_builder::Builder;
use ode_solvers::{DVector, Dopri5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::enzyme_ml::{EnzymeMLDocument, Parameter};
use crate::prelude::{EquationType, Measurement};
use crate::simulation::result::SimulationResult;
use crate::simulation::system::ODESystem;

pub type InitialConditionType = HashMap<String, f64>;

#[derive(Debug, Clone, Builder, Serialize, Deserialize)]
pub struct SimulationSetup {
    #[builder(default = "0.0")]
    t0: f64,
    t1: f64,
    #[builder(default = "1e-1")]
    dt: f64,
    #[builder(default = "1e-3")]
    rtol: f64,
    #[builder(default = "1e-6")]
    atol: f64,
}

#[derive(Debug)]
pub enum InitCondInput {
    Single(HashMap<String, f64>),
    Multiple(Vec<HashMap<String, f64>>),
}

impl From<Measurement> for InitCondInput {
    fn from(measurement: Measurement) -> Self {
        Self::Single(HashMap::from_iter(
            measurement
                .species_data
                .iter()
                .map(|species| (species.species_id.clone(), species.initial.into())),
        ))
    }
}

impl From<Vec<Measurement>> for InitCondInput {
    fn from(measurements: Vec<Measurement>) -> Self {
        let mut init_conds = Vec::new();

        for measurement in measurements.iter() {
            init_conds.push(HashMap::from_iter(
                measurement
                    .species_data
                    .iter()
                    .map(|species| (species.species_id.clone(), species.initial.into())),
            ));
        }

        Self::Multiple(init_conds)
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Equations {
    pub(crate) ode: HashMap<String, String>,
    pub(crate) initial_assignments: HashMap<String, String>,
    pub(crate) assignments: HashMap<String, String>,
}

impl From<InitialConditionType> for InitCondInput {
    fn from(ic: HashMap<String, f64>) -> Self {
        InitCondInput::Single(ic)
    }
}

impl From<Vec<HashMap<String, f64>>> for InitCondInput {
    fn from(ics: Vec<HashMap<String, f64>>) -> Self {
        InitCondInput::Multiple(ics)
    }
}

/// Simulates the given EnzymeMLDocument with the provided initial conditions and simulation parameters.
///
/// # Arguments
///
/// * enzmldoc - A reference to the EnzymeMLDocument.
/// * initial_conditions - The initial conditions for the simulation, which can be either single or multiple.
/// * t0 - The initial time of the simulation.
/// * t1 - The final time of the simulation.
/// * dt - The time step for the simulation.
/// * rtol - The relative tolerance for the ODE solver.
/// * atol - The absolute tolerance for the ODE solver.
///
/// # Returns
///
/// Returns a Result containing a vector of SimulationResult or an error.
pub fn simulate(
    enzmldoc: &EnzymeMLDocument,
    initial_conditions: InitCondInput,
    setup: SimulationSetup,
) -> Result<Vec<SimulationResult>, Box<dyn Error>> {
    match initial_conditions {
        InitCondInput::Single(ic) => integrate(enzmldoc, ic, setup).map(|res| vec![res]),
        InitCondInput::Multiple(ics) => integrate_multiple(enzmldoc, ics, setup),
    }
}

/// Integrates multiple initial conditions for the given EnzymeMLDocument.
///
/// # Arguments
///
/// * enzmldoc - A reference to the EnzymeMLDocument.
/// * initial_conditions - A vector of initial conditions as hashmaps.
/// * setup - The simulation setup parameters.
///
/// # Returns
///
/// Returns a Result containing a vector of SimulationResult or an error.
fn integrate_multiple(
    enzmldoc: &EnzymeMLDocument,
    initial_conditions: Vec<HashMap<String, f64>>,
    setup: SimulationSetup,
) -> Result<Vec<SimulationResult>, Box<dyn Error>> {
    let results: Vec<SimulationResult> = initial_conditions
        .par_iter()
        .map(|ic| {
            let result = integrate(&enzmldoc.clone(), ic.clone(), setup.clone());
            result.unwrap()
        })
        .collect::<Vec<SimulationResult>>();

    Ok(results)
}

/// Integrates a single set of initial conditions for the given EnzymeMLDocument.
///
/// # Arguments
///
/// * enzmldoc - A reference to the EnzymeMLDocument.
/// * initial_conditions - A hashmap of initial conditions.
/// * setup - The simulation setup parameters.
///
/// # Returns
///
/// Returns a Result containing a SimulationResult or an error.
fn integrate(
    enzmldoc: &EnzymeMLDocument,
    initial_conditions: HashMap<String, f64>,
    setup: SimulationSetup,
) -> Result<SimulationResult, Box<dyn Error>> {
    let parameters = extract_all_parameters(enzmldoc)?;
    let equations = extract_equations(enzmldoc);

    let system = ODESystem::new(parameters, initial_conditions, equations)?;

    let y0 = system.get_y0_vector();
    let mut stepper = Dopri5::new(
        system.clone(),
        setup.t0,
        setup.t1,
        setup.dt,
        y0,
        setup.rtol,
        setup.atol,
    );

    let res = stepper.integrate();

    match res {
        Ok(_) => {
            let df = collect_results(stepper.x_out(), stepper.y_out(), &system);
            match df {
                Ok(df) => Ok(df),
                Err(e) => Err(e),
            }
        }
        Err(err) => Err(err.into()),
    }
}

/// Collects the results of the ODE solver into a SimulationResult.
///
/// # Arguments
///
/// * t - A reference to a vector of time points.
/// * y - A reference to a vector of solution vectors.
/// * species_mapping - A reference to a hashmap mapping species names to indices.
///
/// # Returns
///
/// Returns a Result containing a SimulationResult or an error.
fn collect_results(
    t: &Vec<f64>,
    y: &[DVector<f64>],
    system: &ODESystem,
) -> Result<SimulationResult, Box<dyn Error>> {
    let mut result = SimulationResult::new(t.to_owned());

    // Prepare assignments
    for (assignment, _) in system.equations.assignments.iter() {
        result.add_assignment(assignment.clone(), Vec::new());
    }

    // Calculate assignments
    for (t, data) in t.iter().zip(y.iter()) {
        let values = system.calculate_assignment_rules(data, t)?;
        for (assignment, value) in values.iter() {
            result.assignments.get_mut(assignment).unwrap().push(*value);
        }
    }

    for (species, i) in system.species_mapping.iter() {
        result.add_species(species.clone(), y.iter().map(|x| x[*i]).collect());
    }

    Ok(result)
}

/// Extracts all parameters from the given EnzymeMLDocument.
///
/// # Arguments
///
/// * doc - A reference to the EnzymeMLDocument.
///
/// # Returns
///
/// Returns a Result containing a hashmap of parameters or an error.
fn extract_all_parameters(doc: &EnzymeMLDocument) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    let mut params = HashMap::new();

    for param in doc.parameters.iter() {
        params.insert(param.id.clone(), param.clone());
    }

    validate_parameters(params)
}

/// Validates the extracted parameters.
///
/// # Arguments
///
/// * params - A hashmap of parameters.
///
/// # Returns
///
/// Returns a Result containing a validated hashmap of parameters or an error.
fn validate_parameters(
    params: HashMap<String, Parameter>,
) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    let mut validated_params: HashMap<String, f64> = HashMap::new();
    let mut errors = Vec::new();

    for (key, param) in params.iter() {
        if param.value.is_none() {
            errors.push(format!("Parameter {} is missing a value", key));
        } else {
            validated_params.insert(key.clone(), param.value.unwrap() as f64);
        }
    }

    if !errors.is_empty() {
        let msg = format!("Parameters missing values:\n{}", errors.join("\n"));
        return Err(msg.into());
    }

    Ok(validated_params)
}

/// Extracts all equations from the given EnzymeMLDocument.
///
/// # Arguments
///
/// * doc - A reference to the EnzymeMLDocument.
///
/// # Returns
///
/// Returns a hashmap of equations.
fn extract_equations(doc: &EnzymeMLDocument) -> Equations {
    let mut equations = Equations::default();
    for equation in doc.equations.iter() {
        match equation.equation_type {
            EquationType::Ode => {
                equations.ode.insert(
                    equation.species_id.clone().unwrap(),
                    prepare_equation(&equation.equation.clone()),
                );
            }
            EquationType::InitialAssignment => {
                equations.initial_assignments.insert(
                    equation.species_id.clone().unwrap(),
                    prepare_equation(&equation.equation.clone()),
                );
            }
            EquationType::Assignment => {
                equations.assignments.insert(
                    equation.species_id.clone().unwrap(),
                    prepare_equation(&equation.equation.clone()),
                );
            }
            _ => {}
        }
    }

    equations
}

fn prepare_equation(eq: &str) -> String {
    let functions = ["exp", "log", "sin", "cos", "tan", "sqrt", "abs"];
    let mut eq = eq.to_string();

    for &func in &functions {
        let math_func = format!("math::{}", func);
        eq = eq.replace(func, &math_func);
    }

    eq
}
