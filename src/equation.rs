use meval::Expr;
use std::collections::HashSet;
use std::error::Error;

use crate::prelude::{EnzymeMLDocument, EnzymeMLDocumentBuilder, EquationBuilder, EquationType};
use crate::prelude::{Parameter, ParameterBuilder, Variable, VariableBuilder};

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
    variables: &[String],
    eq_type: EquationType,
    enzmldoc: EnzymeMLDocState,
) -> Result<EquationBuilder, Box<dyn Error>> {
    // Parse equation
    let equation: Expr = match eq.parse() {
        Ok(e) => e,
        Err(_) => panic!("Could not parse equation."),
    };

    let (params, variables) = extract_params_and_vars(&variables, &equation);
    let mut eq_builder = EquationBuilder::default();

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
                if !doc.parameters.iter().any(|p| p.symbol == p.symbol) {
                    doc.parameters.push(p.clone());
                }
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
pub(crate) fn extract_params_and_vars(
    variables: &[String],
    equation: &Expr,
) -> (Vec<Parameter>, Vec<Variable>) {
    // Extract variables from equation and ids from the enzml doc
    let symbols = extract_symbols(equation);

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
fn filter_variables(variables: &[String], symbols: &[String]) -> Vec<Variable> {
    symbols
        .iter()
        .filter(|v| variables.contains(v))
        .map(|v| {
            VariableBuilder::default()
                .id(v.clone())
                .name(v.clone())
                .symbol(v.clone())
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
fn filter_params(variables: &[String], symbols: &[String]) -> Vec<Parameter> {
    symbols
        .iter()
        .filter(|s| !variables.contains(s))
        .map(|s| {
            ParameterBuilder::default()
                .id(s.clone())
                .name(s.clone())
                .symbol(s.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_symbols() {
        let expr: Expr = "2*x + y*z + k".parse().unwrap();
        let symbols = extract_symbols(&expr);
        let mut expected = vec!["x", "y", "z", "k"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        expected.sort();
        let mut symbols = symbols;
        symbols.sort();
        assert_eq!(symbols, expected);
    }

    #[test]
    fn test_filter_variables() {
        let variables = vec!["x".to_string(), "y".to_string()];
        let symbols = vec!["x".to_string(), "y".to_string(), "k".to_string()];
        let vars = filter_variables(&variables, &symbols);
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].id, "x");
        assert_eq!(vars[1].id, "y");
    }

    #[test]
    fn test_filter_params() {
        let variables = vec!["x".to_string(), "y".to_string()];
        let symbols = vec!["x".to_string(), "y".to_string(), "k".to_string()];
        let params = filter_params(&variables, &symbols);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].id, "k");
    }

    #[test]
    fn test_create_equation() {
        let mut doc_builder = EnzymeMLDocumentBuilder::default();
        let eq = "2*x + y";
        let variables = vec!["x".to_string(), "y".to_string()];
        let eq_type = EquationType::RATE_LAW;

        let result = create_equation(
            eq,
            &variables,
            eq_type,
            EnzymeMLDocState::from(&mut doc_builder),
        );

        assert!(result.is_ok());

        if let Ok(mut eq_builder) = result {
            let eq_builder = eq_builder
                .species_id("test".to_string())
                .build()
                .expect("Could not build EqBuilder.");

            assert_eq!(eq_builder.species_id, "test");
            assert_eq!(eq_builder.equation, "2*x + y");
            assert_eq!(eq_builder.equation_type, EquationType::RATE_LAW);
            assert_eq!(eq_builder.variables.len(), 2);
        }
    }
}
