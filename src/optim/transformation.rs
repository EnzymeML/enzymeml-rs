use crate::prelude::{EnzymeMLDocument, Equation, EquationBuilder, EquationType};

use super::error::OptimizeError;

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
    Abs(String),
    Log(String),
    NegExp(String),
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
            Transformation::Log(s) => s,
            Transformation::Abs(s) => s,
            Transformation::NegExp(s) => s,
        };

        let equation_string = self.equation_string();

        EquationBuilder::default()
            .species_id(format!("{variable}_transformed"))
            .equation(equation_string.clone())
            .equation_type(EquationType::INITIAL_ASSIGNMENT)
            .build()
            .map_err(|e| OptimizeError::TransformationError {
                variable: variable.to_string(),
                transformation: equation_string,
                message: e.to_string(),
            })
    }

    pub fn equation_string(&self) -> String {
        match self {
            Transformation::SoftPlus(s) => format!("ln(1 + exp({s}))", s = s),
            Transformation::MultScale(s, scale) => format!("{s} * {scale}", s = s, scale = scale),
            Transformation::Pow(s, power) => format!("{s}^{power}", s = s, power = power),
            Transformation::Log(s) => format!("log({s})", s = s),
            Transformation::Abs(s) => format!("abs({s})", s = s),
            Transformation::NegExp(s) => format!("exp(-{s})", s = s),
        }
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
            Transformation::SoftPlus(_) => (1_f64 + value.exp()).ln(),
            Transformation::MultScale(_, scale) => value * scale,
            Transformation::Pow(_, power) => value.powi(*power as i32),
            Transformation::Log(_) => value.ln(),
            Transformation::Abs(_) => value.abs(),
            Transformation::NegExp(_) => (-value).exp(),
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
            Transformation::Log(s) => s.clone(),
            Transformation::Abs(s) => s.clone(),
            Transformation::NegExp(s) => s.clone(),
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
            Transformation::Log(s) => format!("{s}_transformed"),
            Transformation::Abs(s) => format!("{s}_transformed"),
            Transformation::NegExp(s) => format!("{s}_transformed"),
        }
    }

    /// Adds the transformation equation to an EnzymeML document
    ///
    /// Creates an equation representing this parameter transformation and adds it
    /// to the document's equations list. This allows the transformation to be
    /// applied during model simulation.
    ///
    /// # Arguments
    /// * `doc` - EnzymeML document to add the transformation equation to
    ///
    /// # Returns
    /// * `Result<(), OptimizeError>` - Success or error if equation creation fails
    pub fn add_to_enzmldoc(&self, doc: &mut EnzymeMLDocument) -> Result<(), OptimizeError> {
        let equation = self.equation()?;
        doc.equations.push(equation);

        for equation in doc.equations.iter_mut() {
            if equation.species_id != self.transform_symbol() {
                equation.equation = equation
                    .equation
                    .replace(&self.symbol(), &self.transform_symbol());
            }
        }

        Ok(())
    }
}
