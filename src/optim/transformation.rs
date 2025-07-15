use std::str::FromStr;

use regex::Regex;

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
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Transformation {
    SoftPlus(String),
    MultScale(String, f64),
    Pow(String, f64),
    Abs(String),
    Log(String),
    NegExp(String),
    Log1P(String),
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
            Transformation::Log1P(s) => s,
        };

        let equation_string = self.equation_string();

        let equation = EquationBuilder::default()
            .species_id(format!("{variable}_transformed"))
            .equation(equation_string.clone())
            .equation_type(EquationType::InitialAssignment)
            .build()
            .map_err(|e| OptimizeError::TransformationError {
                variable: variable.to_string(),
                transformation: equation_string,
                message: e.to_string(),
            })?;

        Ok(equation)
    }

    // Inverse transformations
    pub fn equation_string(&self) -> String {
        match self {
            Transformation::SoftPlus(s) => format!("exp({s}) - 1"),
            Transformation::MultScale(s, scale) => format!("{s} / {scale}"),
            Transformation::Pow(s, power) => format!("{s}^(1/{power})"),
            Transformation::Log(s) => format!("exp({s})"),
            Transformation::Abs(s) => format!("abs({s})"), // Abs is its own inverse
            Transformation::NegExp(s) => format!("-log({s})"),
            Transformation::Log1P(s) => format!("exp({s}) - 1"),
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
    pub fn apply_forward(&self, value: f64) -> f64 {
        match self {
            Transformation::SoftPlus(_) => (1_f64 + value.exp()).ln(),
            Transformation::MultScale(_, scale) => value * scale,
            Transformation::Pow(_, power) => value.powf(*power),
            Transformation::Log(_) => (1e-12 + value).ln(),
            Transformation::Abs(_) => value.abs(),
            Transformation::NegExp(_) => (-value).exp(),
            Transformation::Log1P(_) => (1_f64 + value).ln(),
        }
    }

    /// Applies the inverse transformation to a single parameter value
    ///
    /// Computes the original value for a transformed parameter according to the
    /// transformation type. This is used to convert optimizer results
    /// back to their original scale/meaning.
    ///
    /// # Arguments
    /// * `value` - Transformed parameter value to transform back
    ///
    /// # Returns
    /// * `f64` - The original parameter value
    pub fn apply_back(&self, value: f64) -> f64 {
        match self {
            Transformation::SoftPlus(_) => (value.exp() - 1.0).ln(),
            Transformation::MultScale(_, scale) => value / scale,
            Transformation::Pow(_, power) => value.powf(1.0 / power),
            Transformation::Log(_) => value.exp() - 1e-12,
            Transformation::Abs(_) => value.abs(),
            Transformation::NegExp(_) => -(value.ln()),
            Transformation::Log1P(_) => value.exp() - 1.0,
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
            Transformation::Log1P(s) => s.clone(),
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
            Transformation::Log1P(s) => format!("{s}_transformed"),
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
                replace_in_equation(
                    &mut equation.equation,
                    &self.symbol(),
                    &self.transform_symbol(),
                );
            }
        }

        for reaction in doc.reactions.iter_mut() {
            if reaction.kinetic_law.is_some() {
                replace_in_equation(
                    &mut reaction.kinetic_law.as_mut().unwrap().equation,
                    &self.symbol(),
                    &self.transform_symbol(),
                );
            }
        }

        Ok(())
    }
}

fn replace_in_equation(equation: &mut String, symbol: &str, transform_symbol: &str) {
    use regex::Regex;

    // Create a regex pattern that matches the symbol as a whole word
    // \b ensures word boundaries, and we escape the symbol in case it contains special regex characters
    let pattern = format!(r"\b{}\b", regex::escape(symbol));
    let re = Regex::new(&pattern).unwrap();

    *equation = re.replace_all(equation, transform_symbol).to_string();
}

impl FromStr for Transformation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let pattern = r"^(\w+)=([a-zA-Z1\-]+)(?:\((.*)\))?$";
        let re = Regex::new(pattern).unwrap();
        let captures = re.captures(s);

        if let Some(captures) = captures {
            let key = captures
                .get(1)
                .ok_or("Missing parameter key")?
                .as_str()
                .to_string();
            let transformation = captures
                .get(2)
                .ok_or("Missing transformation type")?
                .as_str()
                .to_string();
            let args = captures
                .get(3)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            match transformation.as_str() {
                "log" => Ok(Transformation::Log(key)),
                "multscale" => {
                    if args.is_empty() {
                        Err("multscale transformation requires a scale value".to_string())
                    } else {
                        args.parse::<f64>()
                            .map(|scale| Transformation::MultScale(key, scale))
                            .map_err(|_| {
                                "Invalid scale value for multscale transformation".to_string()
                            })
                    }
                }
                "pow" => {
                    if args.is_empty() {
                        Err("pow transformation requires a power value".to_string())
                    } else {
                        args.parse::<f64>()
                            .map(|power| Transformation::Pow(key, power))
                            .map_err(|_| "Invalid power value for pow transformation".to_string())
                    }
                }
                "abs" => Ok(Transformation::Abs(key)),
                "neg-exp" => Ok(Transformation::NegExp(key)),
                "softplus" => Ok(Transformation::SoftPlus(key)),
                "log1p" => Ok(Transformation::Log1P(key)),
                _ => Err("Invalid transformation".to_string()),
            }
        } else {
            Err("Invalid format. Expected key=transformation".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformation_from_str() {
        let transformation = Transformation::from_str("k_cat=log")
            .expect("Failed to parse transformation for k_cat=log");
        assert_eq!(transformation, Transformation::Log("k_cat".to_string()));

        let transformation = Transformation::from_str("k_cat=multscale(10.0)")
            .expect("Failed to parse transformation for k_cat=multscale(10.0)");
        assert_eq!(
            transformation,
            Transformation::MultScale("k_cat".to_string(), 10.0)
        );

        let transformation = Transformation::from_str("k_cat=pow(2.0)")
            .expect("Failed to parse transformation for k_cat=pow(2.0)");
        assert_eq!(
            transformation,
            Transformation::Pow("k_cat".to_string(), 2.0)
        );

        let transformation = Transformation::from_str("k_cat=abs")
            .expect("Failed to parse transformation for k_cat=abs");
        assert_eq!(transformation, Transformation::Abs("k_cat".to_string()));

        let transformation = Transformation::from_str("k_cat=neg-exp")
            .expect("Failed to parse transformation for k_cat=neg-exp");
        assert_eq!(transformation, Transformation::NegExp("k_cat".to_string()));

        let transformation = Transformation::from_str("k_cat=softplus")
            .expect("Failed to parse transformation for k_cat=softplus");
        assert_eq!(
            transformation,
            Transformation::SoftPlus("k_cat".to_string())
        );

        let transformation = Transformation::from_str("k_cat=log1p")
            .expect("Failed to parse transformation for k_cat=log1p");
        assert_eq!(transformation, Transformation::Log1P("k_cat".to_string()));
    }

    #[test]
    fn test_replace_in_equation_surgical() {
        // Test that replacement only affects exact word matches, not partial matches
        let mut equation = "k * k_cat + k1 + substrate_k + k_m".to_string();
        replace_in_equation(&mut equation, "k", "k_transformed");

        // Only the standalone "k" should be replaced, not "k_cat", "k1", "substrate_k", or "k_m"
        assert_eq!(equation, "k_transformed * k_cat + k1 + substrate_k + k_m");
    }

    #[test]
    fn test_replace_in_equation_multiple_occurrences() {
        // Test that all standalone occurrences are replaced
        let mut equation = "k + 2*k - k/3 + k_cat".to_string();
        replace_in_equation(&mut equation, "k", "k_transformed");

        // All standalone "k" should be replaced, but not "k_cat"
        assert_eq!(
            equation,
            "k_transformed + 2*k_transformed - k_transformed/3 + k_cat"
        );
    }

    #[test]
    fn test_replace_in_equation_with_special_chars() {
        // Test replacement with parameters that might contain regex special characters
        let mut equation = "k_m * substrate / (k_m + substrate)".to_string();
        replace_in_equation(&mut equation, "k_m", "k_m_transformed");

        // Both occurrences of "k_m" should be replaced
        assert_eq!(
            equation,
            "k_m_transformed * substrate / (k_m_transformed + substrate)"
        );
    }

    #[test]
    fn test_replace_in_equation_parentheses_and_operators() {
        // Test replacement in complex expressions with various operators
        let mut equation = "(k + k_cat) * exp(-k * t) + log(k)".to_string();
        replace_in_equation(&mut equation, "k", "k_transformed");

        // Only standalone "k" should be replaced, not "k_cat"
        assert_eq!(
            equation,
            "(k_transformed + k_cat) * exp(-k_transformed * t) + log(k_transformed)"
        );
    }

    #[test]
    fn test_replace_in_equation_no_match() {
        // Test that nothing changes when the symbol is not found
        let mut equation = "k_cat * substrate + k_m".to_string();
        let original = equation.clone();
        replace_in_equation(&mut equation, "k", "k_transformed");

        // Equation should remain unchanged
        assert_eq!(equation, original);
    }

    #[test]
    fn test_replace_in_equation_underscore_parameters() {
        // Test with parameters that have underscores to ensure word boundaries work correctly
        let mut equation = "k_cat_max * substrate / (k_m + substrate) + k_cat".to_string();
        replace_in_equation(&mut equation, "k_cat", "k_cat_transformed");

        // Only "k_cat" should be replaced, not "k_cat_max"
        assert_eq!(
            equation,
            "k_cat_max * substrate / (k_m + substrate) + k_cat_transformed"
        );
    }
}
