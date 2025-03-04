//! Unit Macro Module
//!
//! This module provides a powerful macro `unit` for flexible and concise unit definition creation.
//!
//! # Key Features
//!
//! - Create unit definitions with a single macro invocation
//! - Support for inverse unit definitions (1 / [prefix unit])
//! - Automatic parsing of prefixes and base units
//! - Flexible handling of unit scales and exponents
//!
//! # Macro Behavior
//!
//! The macro simplifies unit definition creation by:
//! - Automatically converting prefixes and units to a standardized format
//! - Generating base unit representations
//! - Handling unit scale and exponent calculations
//! - Providing error checking for unknown prefixes or units

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
