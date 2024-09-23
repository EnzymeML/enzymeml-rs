use std::collections::HashSet;
use std::error::Error;

use meval::Expr;

use crate::enzyme_ml;
use crate::enzyme_ml::{Parameter, ParameterBuilder, Variable, VariableBuilder};
use crate::prelude::{EnzymeMLDocument, EnzymeMLDocumentBuilder};

/// Represents the state of an EnzymeML document, either as a builder or a document.
pub enum EnzymeMLDocState<'a> {
    Builder(&'a mut EnzymeMLDocumentBuilder),
    Document(&'a mut EnzymeMLDocument),
}

impl<'a> From<&'a mut EnzymeMLDocumentBuilder> for EnzymeMLDocState<'a> {
    /// Converts a mutable reference to an `EnzymeMLDocumentBuilder` into an `EnzymeMLDocState`.
    fn from(builder: &'a mut EnzymeMLDocumentBuilder) -> Self {
        EnzymeMLDocState::Builder(builder)
    }
}

impl<'a> From<&'a mut EnzymeMLDocument> for EnzymeMLDocState<'a> {
    /// Converts a mutable reference to an `EnzymeMLDocument` into an `EnzymeMLDocState`.
    fn from(doc: &'a mut EnzymeMLDocument) -> Self {
        EnzymeMLDocState::Document(doc)
    }
}

/// Creates an equation and adds it to the EnzymeML document.
///
/// # Arguments
///
/// * `eq` - The equation as a string.
/// * `variables` - A vector of variable names used in the equation.
/// * `eq_type` - The type of the equation.
/// * `enzmldoc` - The state of the EnzymeML document.
///
/// # Returns
///
/// Returns a `Result` containing the `EquationBuilder` or an error if the creation fails.
pub fn create_equation(
    eq: &str,
    variables: Vec<String>,
    eq_type: enzyme_ml::EquationType,
    enzmldoc: EnzymeMLDocState,
) -> Result<enzyme_ml::EquationBuilder, Box<dyn Error>> {
    // Parse equation
    let equation: Expr = match eq.parse() {
        Ok(e) => e,
        Err(_) => panic!("Could not parse equation."),
    };

    let (params, variables) = extract_params_and_vars(variables, &equation);
    let mut eq_builder = enzyme_ml::EquationBuilder::default();

    eq_builder
        .variables(variables)
        .equation(eq.to_string())
        .equation_type(eq_type);

    match enzmldoc {
        EnzymeMLDocState::Builder(builder) => {
            params.iter().for_each(|p| {
                builder.to_parameters(p.clone());
            });
        }
        EnzymeMLDocState::Document(doc) => {
            params.iter().for_each(|p| {
                doc.parameters.push(p.clone());
            });
        }
    }

    Ok(eq_builder)
}

/// Extracts parameters and variables from the given equation.
///
/// # Arguments
///
/// * `variables` - A vector of variable names.
/// * `equation` - The parsed equation.
///
/// # Returns
///
/// Returns a tuple containing a vector of `Parameter` and a vector of `Variable`.
fn extract_params_and_vars(
    variables: Vec<String>,
    equation: &Expr,
) -> (Vec<Parameter>, Vec<Variable>) {
    // Extract variables from equation and ids from the enzml doc
    let symbols = extract_symbols(&equation);
    check_variable_consistency(&variables, &symbols);

    // Sort variables and parameters
    let params = filter_params(&variables, &symbols);
    let variables = filter_variables(&variables, &symbols);
    (params, variables)
}

/// Filters and creates `Variable` instances from the given symbols.
///
/// # Arguments
///
/// * `variables` - A reference to a vector of variable names.
/// * `symbols` - A reference to a vector of symbols extracted from the equation.
///
/// # Returns
///
/// Returns a vector of `Variable`.
fn filter_variables(variables: &Vec<String>, symbols: &Vec<String>) -> Vec<Variable> {
    variables
        .iter()
        .filter(|v| symbols.contains(v))
        .map(|v| {
            VariableBuilder::default()
                .id(v)
                .name(v)
                .symbol(v)
                .build()
                .expect("Could not build EqVariable.")
        })
        .collect()
}

/// Filters and creates `Parameter` instances from the given symbols.
///
/// # Arguments
///
/// * `variables` - A reference to a vector of variable names.
/// * `symbols` - A reference to a vector of symbols extracted from the equation.
///
/// # Returns
///
/// Returns a vector of `Parameter`.
fn filter_params(variables: &Vec<String>, symbols: &Vec<String>) -> Vec<Parameter> {
    symbols
        .iter()
        .filter(|s| !variables.contains(s))
        .map(|s| {
            ParameterBuilder::default()
                .id(s)
                .name(s)
                .symbol(s)
                .build()
                .expect("Could not build EqParameter.")
        })
        .collect()
}

/// Extracts symbols (variables) from the given equation.
///
/// # Arguments
///
/// * `eq` - A reference to the parsed equation.
///
/// # Returns
///
/// Returns a vector of symbols as strings.
pub fn extract_symbols(eq: &Expr) -> Vec<String> {
    let mut vars: HashSet<String> = HashSet::new();

    for var in eq.iter() {
        if let meval::tokenizer::Token::Var(v) = var {
            vars.insert(v.into());
        }
    }

    vars.into_iter().collect()
}

/// Checks the consistency of the given variables with the symbols extracted from the equation.
///
/// # Arguments
///
/// * `variables` - A reference to a vector of variable names.
/// * `symbols` - A reference to a vector of symbols extracted from the equation.
///
/// # Panics
///
/// Panics if a variable is not found in the symbols.
fn check_variable_consistency(variables: &Vec<String>, symbols: &Vec<String>) {
    for var in variables {
        if !symbols.contains(var) {
            panic!("Variable '{}' not found in equation.", var);
        }
    }
}
