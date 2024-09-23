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
