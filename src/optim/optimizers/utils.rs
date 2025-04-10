use crate::optim::Transformation;

use super::InitialGuesses;

/// Transform the initial guesses based on the transformations
///
/// Given there are trasnformations, we need to transform the initial guesses
/// based on the transformations. This way, users can specify initial guesses
/// in the original scale of the parameters.
///
/// # Arguments
///
/// * `param_order` - The order of the parameters
/// * `initial_guesses` - The initial guesses
/// * `transformations` - The transformations to apply
pub(crate) fn transform_initial_guesses(
    param_order: &[String],
    initial_guesses: &mut InitialGuesses,
    transformations: &[Transformation],
) {
    for transformation in transformations {
        let index = param_order
            .iter()
            .position(|p| p == &transformation.symbol())
            .expect("Parameter not found");

        initial_guesses.set_value_at(
            index,
            transformation.apply_forward(initial_guesses.get_value_at(index)),
        );
    }
}
