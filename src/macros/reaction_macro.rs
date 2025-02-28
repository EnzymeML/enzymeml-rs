//! Reaction Macro Module
//!
//! This module provides a powerful macro `build_reaction` for flexible and concise reaction construction.
//!
//! # Key Features
//!
//! - Create reactions with a single macro invocation
//! - Specify reaction ID, name, and reversibility
//! - Define species and their stoichiometric coefficients
//!
//! # Macro Behavior
//!
//! The macro simplifies reaction creation by:
//! - Automatically using `ReactionBuilder`
//! - Constructing `ReactionElementBuilder` for each species
//! - Handling error propagation with `?` operator

#[macro_export]
macro_rules! build_reaction {
    (
        $reaction_id:expr,
        $reaction_name:expr,
        $reversible:expr,
        $(
            $species_id:expr => $stoichiometry:expr
        ),*
    ) => {
        enzymeml::prelude::ReactionBuilder::default()
            .id($reaction_id)
            .name($reaction_name)
            .reversible($reversible)
            $(
                .to_species(
                    enzymeml::prelude::ReactionElementBuilder::default()
                        .species_id($species_id)
                        .stoichiometry($stoichiometry)
                        .build()?,
                )
            )*
            .build()?;
    };
}
