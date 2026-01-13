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

/// Helper macro to create a single BaseUnit from prefix and unit tokens
#[macro_export]
macro_rules! build_base_unit {
    ($prefix:tt, $unit:ident, $exponent:expr) => {
        $crate::prelude::BaseUnit {
            scale: Some(
                $crate::macros::unit_maps::PREFIX_MAPPING
                    .get(stringify!($prefix))
                    .expect(&format!("Prefix {} not found", stringify!($prefix)))
                    .clone(),
            ),
            kind: $crate::macros::unit_maps::KIND_MAPPINGS
                .get(stringify!($unit))
                .expect(&format!("Unit {} not found", stringify!($unit)))
                .clone()
                .0,
            exponent: $exponent,
            multiplier: $crate::macros::unit_maps::KIND_MAPPINGS
                .get(stringify!($unit))
                .expect(&format!("Unit {} not found", stringify!($unit)))
                .clone()
                .1,
        }
    };
}

/// Helper macro to build unit name from prefix/unit pairs
#[macro_export]
macro_rules! build_unit_name {
    ($([$prefix:tt $unit:ident])+) => {
        vec![
            $(
                stringify!($prefix),
                stringify!($unit)
            ),+
        ].join("").replace('_', "")
    };
}

/// Helper macro to collect base units with specified exponent
#[macro_export]
macro_rules! collect_base_units {
    ($exponent:expr; $([$prefix:tt $unit:ident])+) => {
        {
            let mut units = vec![];
            $(
                units.push($crate::build_base_unit!($prefix, $unit, $exponent));
            )+
            units
        }
    };
}

/// Macro for creating inverse units (1 / unit)
#[macro_export]
macro_rules! inverse_unit {
    ($([$prefix:tt $unit:ident])+) => {
        $crate::prelude::UnitDefinition {
            id: None,
            name: format!("1 / {}", $crate::build_unit_name!($([$prefix $unit])+)).into(),
            base_units: $crate::collect_base_units!(-1; $([$prefix $unit])+),
        }
    };
}

/// Macro for creating ratio units (numerator / denominator)
#[macro_export]
macro_rules! ratio_unit {
    ($([$num_prefix:tt $num_unit:ident])+; $([$den_prefix:tt $den_unit:ident])+) => {
        $crate::prelude::UnitDefinition {
            id: None,
            name: format!(
                "{} / {}",
                $crate::build_unit_name!($([$num_prefix $num_unit])+),
                $crate::build_unit_name!($([$den_prefix $den_unit])+)
            ).into(),
            base_units: {
                let mut units = $crate::collect_base_units!(1; $([$num_prefix $num_unit])+);
                units.extend($crate::collect_base_units!(-1; $([$den_prefix $den_unit])+));
                units
            },
        }
    };
}

/// Macro for creating simple units
#[macro_export]
macro_rules! simple_unit {
    ($([$prefix:tt $unit:ident])+) => {
        $crate::prelude::UnitDefinition {
            id: None,
            name: $crate::build_unit_name!($([$prefix $unit])+).into(),
            base_units: $crate::collect_base_units!(1; $([$prefix $unit])+),
        }
    };
}

/// Main unit macro that delegates to appropriate sub-macros
#[macro_export]
macro_rules! unit {
    // Inverse units: 1 / [prefix unit]+
    (1 / $([$num_prefix:tt $num_unit:ident])+) => {
        $crate::inverse_unit!($([$num_prefix $num_unit])+)
    };

    // Ratio units: [prefix unit]+ / [prefix unit]+
    ($([$num_prefix:tt $num_unit:ident])+ / $([$den_prefix:tt $den_unit:ident])+) => {
        $crate::ratio_unit!($([$num_prefix $num_unit])+; $([$den_prefix $den_unit])+)
    };

    // Simple units: [prefix unit]+
    ($([$num_prefix:tt $num_unit:ident])+) => {
        $crate::simple_unit!($([$num_prefix $num_unit])+)
    };
}

#[cfg(test)]
mod tests {
    use crate::prelude::UnitType;

    #[test]
    fn test_unit_ratio_macro() {
        let unit = unit!([_ mole] / [_ liter]);
        assert_eq!(unit.name, Some("mole / liter".to_string()));
        assert_eq!(unit.base_units.len(), 2);
        assert_eq!(unit.base_units[0].kind, UnitType::Mole);
        assert_eq!(unit.base_units[0].exponent, 1);
        assert_eq!(unit.base_units[0].multiplier, None);
        assert_eq!(unit.base_units[1].kind, UnitType::Litre);
        assert_eq!(unit.base_units[1].exponent, -1);
        assert_eq!(unit.base_units[1].multiplier, None);
    }

    #[test]
    fn test_unit_ratio_macro_with_prefix() {
        let unit = unit!([m mol] / [_ liter]);
        assert_eq!(unit.name, Some("mmol / liter".to_string()));
        assert_eq!(unit.base_units.len(), 2);
        assert_eq!(unit.base_units[0].kind, UnitType::Mole);
        assert_eq!(unit.base_units[0].scale, Some(-3.0));
        assert_eq!(unit.base_units[0].exponent, 1);
        assert_eq!(unit.base_units[0].multiplier, None);
        assert_eq!(unit.base_units[1].kind, UnitType::Litre);
        assert_eq!(unit.base_units[1].exponent, -1);
        assert_eq!(unit.base_units[1].multiplier, None);
    }

    #[test]
    fn test_unit_macro_with_prefix() {
        let unit = unit!([m mole]);
        assert_eq!(unit.name, Some("mmole".to_string()));
        assert_eq!(unit.base_units.len(), 1);
        assert_eq!(unit.base_units[0].kind, UnitType::Mole);
        assert_eq!(unit.base_units[0].scale, Some(-3.0));
        assert_eq!(unit.base_units[0].exponent, 1);
    }

    #[test]
    fn test_unit_macro_inverse() {
        let unit = unit!(1 / [_ liter]);
        assert_eq!(unit.name, Some("1 / liter".to_string()));
        assert_eq!(unit.base_units.len(), 1);
        assert_eq!(unit.base_units[0].kind, UnitType::Litre);
        assert_eq!(unit.base_units[0].scale, Some(1.0));
        assert_eq!(unit.base_units[0].exponent, -1);
        assert_eq!(unit.base_units[0].multiplier, None);
    }
}
