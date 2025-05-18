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
        m.insert("mole", (UnitType::Mole, None));
        m.insert("mol", (UnitType::Mole, None));

        // Liter
        m.insert("liter", (UnitType::Litre, None));
        m.insert("litre", (UnitType::Litre, None));
        m.insert("l", (UnitType::Litre, None));

        // Second
        m.insert("second", (UnitType::Second, None));
        m.insert("s", (UnitType::Second, None));

        // Minute
        m.insert("minute", (UnitType::Second, Some(60_f64)));
        m.insert("min", (UnitType::Second, Some(60_f64)));
        m.insert("mins", (UnitType::Second, Some(60_f64)));
        m.insert("minutes", (UnitType::Second, Some(60_f64)));

        // Hour
        m.insert("hour", (UnitType::Second, Some(60_f64*60_f64)));
        m.insert("hours", (UnitType::Second, Some(60_f64*60_f64)));
        m.insert("hr", (UnitType::Second, Some(60_f64*60_f64)));
        m.insert("h", (UnitType::Second, Some(60_f64*60_f64)));

        // Day
        m.insert("day", (UnitType::Second, Some(60_f64*60_f64*24_f64)));
        m.insert("days", (UnitType::Second, Some(60_f64*60_f64*24_f64)));
        m.insert("d", (UnitType::Second, Some(60_f64*60_f64*24_f64)));

        // Gram
        m.insert("gram", (UnitType::Gram, None));
        m.insert("g", (UnitType::Gram, None));

        // Dimensionless
        m.insert("dimensionless", (UnitType::Dimensionless, None));
        m.insert("_", (UnitType::Dimensionless, None));

        // Kelvin
        m.insert("kelvin", (UnitType::Kelvin, None));
        m.insert("k", (UnitType::Kelvin, None));
        m.insert("K", (UnitType::Kelvin, None));

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
