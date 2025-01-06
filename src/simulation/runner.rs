use std::collections::HashMap;

use ode_solvers::{DVector, Dopri5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::enzyme_ml::{EnzymeMLDocument, Parameter};
use crate::prelude::{EquationType, Measurement};
use crate::simulation::result::SimulationResult;
use crate::simulation::system::ODESystem;

use super::error::SimulationError;
use super::SimulationSetup;

pub type InitialConditionType = HashMap<String, f64>;

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
    parameters: Option<HashMap<String, f64>>,
    evaluate: Option<&Vec<f64>>,
) -> Result<Vec<SimulationResult>, SimulationError> {
    match initial_conditions {
        InitCondInput::Single(ic) => {
            integrate(enzmldoc, ic, setup, parameters, evaluate).map(|res| vec![res])
        }
        InitCondInput::Multiple(ics) => {
            integrate_multiple(enzmldoc, ics, setup, parameters, evaluate)
        }
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
    parameters: Option<HashMap<String, f64>>,
    evaluate: Option<&Vec<f64>>,
) -> Result<Vec<SimulationResult>, SimulationError> {
    let results: Vec<SimulationResult> = initial_conditions
        .par_iter()
        .map(|ic| {
            let result = integrate(
                &enzmldoc.clone(),
                ic.clone(),
                setup.clone(),
                parameters.clone(),
                evaluate,
            )?;
            Ok(result)
        })
        .collect::<Result<Vec<SimulationResult>, SimulationError>>()?;

    Ok(results)
}

/// Integrates a single set of initial conditions for the given EnzymeMLDocument.
///
/// # Arguments
///
/// * enzmldoc - A reference to the EnzymeMLDocument.
/// * initial_conditions - A hashmap of initial conditions.
/// * setup - The simulation setup parameters.
/// * parameters - A hashmap of parameters, if using in an optimization.
///
/// # Returns
///
/// Returns a Result containing a SimulationResult or an error.
fn integrate(
    enzmldoc: &EnzymeMLDocument,
    initial_conditions: HashMap<String, f64>,
    setup: SimulationSetup,
    parameters: Option<HashMap<String, f64>>,
    evaluate: Option<&Vec<f64>>,
) -> Result<SimulationResult, SimulationError> {
    let parameters = match parameters {
        Some(params) => params,
        None => extract_all_parameters(enzmldoc)?,
    };

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

    if res.is_err() {
        return Err(SimulationError::IntegrationError(res.err().unwrap()));
    }

    match evaluate {
        Some(eval) => {
            let time_points = stepper.x_out();
            let y_out = stepper.y_out();

            // Parallelize finding closest indices
            let indices: Vec<usize> = eval
                .par_iter()
                .map(|&target| {
                    time_points
                        .par_iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            (*a - target)
                                .abs()
                                .partial_cmp(&(*b - target).abs())
                                .unwrap()
                        })
                        .map(|(index, _)| index)
                        .unwrap()
                })
                .collect();

            // Parallelize collecting selected times and values
            let selected_times: Vec<f64> = indices.par_iter().map(|&i| time_points[i]).collect();

            let selected_y: Vec<DVector<f64>> =
                indices.par_iter().map(|&i| y_out[i].clone()).collect();

            collect_results(&selected_times, &selected_y, &system)
        }
        None => collect_results(stepper.x_out(), stepper.y_out(), &system),
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
) -> Result<SimulationResult, SimulationError> {
    let mut result = SimulationResult::new(t.to_owned());

    // Prepare assignments
    for (assignment, _) in system.equations.assignments.iter() {
        result.add_assignment(assignment.clone(), Vec::new());
    }

    // Calculate assignments
    for (t, data) in t.iter().zip(y.iter()) {
        let values = system
            .calculate_assignment_rules(data, t)
            .map_err(|e| SimulationError::CollectResultsError(e.to_string()))?;
        for (assignment, value) in values.iter() {
            result
                .assignments
                .get_mut(assignment)
                .ok_or(SimulationError::CollectResultsError(format!(
                    "Assignment rule for {} not found",
                    assignment
                )))?
                .push(*value);
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
fn extract_all_parameters(doc: &EnzymeMLDocument) -> Result<HashMap<String, f64>, SimulationError> {
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
) -> Result<HashMap<String, f64>, SimulationError> {
    let mut validated_params: HashMap<String, f64> = HashMap::new();
    let mut errors = Vec::new();

    for (key, param) in params.iter() {
        if param.value.is_none() {
            errors.push(format!("Parameter {} is missing a value", key));
        } else {
            validated_params.insert(key.clone(), param.value.unwrap());
        }
    }

    if !errors.is_empty() {
        let msg = format!("Parameters missing values:\n{}", errors.join("\n"));
        return Err(SimulationError::ValidateParametersError(msg));
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

#[derive(Debug)]
pub enum InitCondInput {
    Single(HashMap<String, f64>),
    Multiple(Vec<HashMap<String, f64>>),
}

impl From<&Measurement> for InitCondInput {
    fn from(measurement: &Measurement) -> Self {
        Self::Single(HashMap::from_iter(
            measurement
                .species_data
                .iter()
                .map(|species| (species.species_id.clone(), species.initial)),
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
                    .map(|species| (species.species_id.clone(), species.initial)),
            ));
        }

        if init_conds.len() == 1 {
            Self::Single(init_conds[0].clone())
        } else {
            Self::Multiple(init_conds)
        }
    }
}

impl From<&EnzymeMLDocument> for InitCondInput {
    fn from(doc: &EnzymeMLDocument) -> Self {
        Self::from(doc.measurements.clone())
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
