//! Unit Mapping Module
//!
//! This module provides static mappings for unit kinds and prefixes used in EnzymeML unit definitions.
//!
//! # Key Features
//!
//! - Comprehensive mapping of unit names to their corresponding `UnitType`
//! - Support for multiple unit representations (full names, abbreviations)
//! - Conversion factors for time-based units (minutes, hours, days)
//!
//! # Unit Mappings
//!
//! The module defines mappings for various units including:
//! - Mole
//! - Liter
//! - Time units (seconds, minutes, hours, days)
//! - Mass (grams)
//! - Dimensionless quantities
//! - Temperature (Kelvin)
//!
//! # Usage
//!
//! The mappings are used internally by unit conversion and definition macros
//! to standardize and validate unit specifications.

use std::collections::HashMap;

use crate::prelude::UnitType;

lazy_static::lazy_static! {
    pub static ref KIND_MAPPINGS: HashMap<&'static str, (UnitType, Option<f64>)> = {
        let mut m = HashMap::new();
        // Mole
        m.insert("mole", (UnitType::MOLE, None));
        m.insert("mol", (UnitType::MOLE, None));

        // Liter
        m.insert("liter", (UnitType::LITRE, None));
        m.insert("litre", (UnitType::LITRE, None));
        m.insert("l", (UnitType::LITRE, None));

        // Second
        m.insert("second", (UnitType::SECOND, None));
        m.insert("s", (UnitType::SECOND, None));

        // Minute
        m.insert("minute", (UnitType::SECOND, Some(60_f64)));
        m.insert("min", (UnitType::SECOND, Some(60_f64)));
        m.insert("mins", (UnitType::SECOND, Some(60_f64)));
        m.insert("minutes", (UnitType::SECOND, Some(60_f64)));

        // Hour
        m.insert("hour", (UnitType::SECOND, Some(60_f64*60_f64)));
        m.insert("hours", (UnitType::SECOND, Some(60_f64*60_f64)));
        m.insert("hr", (UnitType::SECOND, Some(60_f64*60_f64)));
        m.insert("h", (UnitType::SECOND, Some(60_f64*60_f64)));

        // Day
        m.insert("day", (UnitType::SECOND, Some(60_f64*60_f64*24_f64)));
        m.insert("days", (UnitType::SECOND, Some(60_f64*60_f64*24_f64)));
        m.insert("d", (UnitType::SECOND, Some(60_f64*60_f64*24_f64)));

        // Gram
        m.insert("gram", (UnitType::GRAM, None));
        m.insert("g", (UnitType::GRAM, None));

        // Dimensionless
        m.insert("dimensionless", (UnitType::DIMENSIONLESS, None));
        m.insert("_", (UnitType::DIMENSIONLESS, None));

        // Kelvin
        m.insert("kelvin", (UnitType::KELVIN, None));
        m.insert("k", (UnitType::KELVIN, None));
        m.insert("K", (UnitType::KELVIN, None));

        m
    };

    pub static ref PREFIX_MAPPING: HashMap<&'static str, f64> = {
        let mut m: HashMap<&str, f64> = HashMap::new();
        m.insert("kilo", 3.0);
        m.insert("k", 3.0);
        m.insert("milli", -3.0);
        m.insert("m", -3.0);
        m.insert("micro", -6.0);
        m.insert("mu", -6.0);
        m.insert("u", -6.0);
        m.insert("nano", -9.0);
        m.insert("n", -9.0);
        m.insert("pico", -12.0);
        m.insert("p", -12.0);
        m.insert("femto", -15.0);
        m.insert("f", -15.0);
        m.insert("atto", -18.0);
        m.insert("a", -18.0);
        m.insert("_", 1.0);
        m
    };
}
