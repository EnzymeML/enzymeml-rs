#[macro_export]
macro_rules! unit {

    ( 1 / $([ $num_prefix:tt $num_unit:ident ])+ ) => {
        enzymeml::enzyme_ml::UnitDefinition {
            // Convert all the prefixes and units to a string
            id: None,
            additional_properties: None,
            name: format!(
                "{}",
                vec![
                    $(
                        stringify!($num_prefix),
                        stringify!($num_unit)
                    ),+
                ].join("")
            ).replace('_', "").into(),
            // Parse units into base units
            base_units: {
                let mut units = vec![];
                $(
                    units.push(enzymeml::enzyme_ml::BaseUnit {
                        scale: Some(
                            enzymeml::macros::unit_maps::PREFIX_MAPPING
                                .get(stringify!($num_prefix))
                                .expect(
                                    format!("Prefix {} not found", stringify!($num_prefix)).as_str()
                                )
                                .clone()
                        ),
                        kind: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().0,
                        exponent: -1,
                        multiplier: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().1,
                        additional_properties: None,
                    });
                )+
                units
            },
        }
    };

    ( $([ $num_prefix:tt $num_unit:ident ])+ / $([ $den_prefix:tt $den_unit:ident ])+ ) => {
        enzymeml::enzyme_ml::UnitDefinition {
            // Convert all the prefixes and units to a string
            id: None,
            additional_properties: None,
            name: format!(
                "{} / {}",
                vec![
                    $(
                        stringify!($num_prefix),
                        stringify!($num_unit)
                    ),+
                ].join(""),
                vec![
                    $(
                        stringify!($den_prefix),
                        stringify!($den_unit)
                    ),+
                ].join("")
            ).replace('_', "").into(),
            // Parse units into base units
            base_units: {
                let mut units = vec![];
                $(
                    units.push(enzymeml::enzyme_ml::BaseUnit {
                        scale: Some(
                            enzymeml::macros::unit_maps::PREFIX_MAPPING
                                .get(stringify!($num_prefix))
                                .expect(
                                    format!("Prefix {} not found", stringify!($num_prefix)).as_str()
                                )
                                .clone()
                        ),
                        kind: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().0,
                        exponent: 1,
                        multiplier: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().1,
                        additional_properties: None,
                    });
                )+
                $(
                    units.push(enzymeml::enzyme_ml::BaseUnit {
                        scale: Some(
                            enzymeml::macros::unit_maps::PREFIX_MAPPING
                                .get(stringify!($den_prefix))
                                .expect(
                                    format!("Prefix {} not found", stringify!($den_prefix)).as_str()
                                )
                                .clone()
                        ),
                        kind: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($den_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($den_unit)).as_str()
                                ).clone().0,
                        exponent: -1,
                        multiplier: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($den_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($den_unit)).as_str()
                                ).clone().1,
                        additional_properties: None,
                    });
                )+
                units
            },
        }
    };

    ( $([ $num_prefix:tt $num_unit:ident ])+ ) => {
        enzymeml::enzyme_ml::UnitDefinition {
            // Convert all the prefixes and units to a string
            id: None,
            additional_properties: None,
            name: format!(
                "{}",
                vec![
                    $(
                        stringify!($num_prefix),
                        stringify!($num_unit)
                    ),+
                ].join("")
            ).replace('_', "").into(),
            // Parse units into base units
            base_units: {
                let mut units = vec![];
                $(
                    units.push(enzymeml::enzyme_ml::BaseUnit {
                        scale: Some(
                            enzymeml::macros::unit_maps::PREFIX_MAPPING
                                .get(stringify!($num_prefix))
                                .expect(
                                    format!("Prefix {} not found", stringify!($num_prefix)).as_str()
                                )
                                .clone()
                        ),
                        kind: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().0,
                        exponent: 1,
                        multiplier: enzymeml::macros::unit_maps::KIND_MAPPINGS
                                .get(stringify!($num_unit))
                                .expect(
                                    format!("Unit {} not found", stringify!($num_unit)).as_str()
                                ).clone().1,
                        additional_properties: None,
                    });
                )+
                units
            },
        }
    };
}
