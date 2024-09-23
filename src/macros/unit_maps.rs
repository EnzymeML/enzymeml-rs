use std::collections::HashMap;

use crate::enzyme_ml::UnitType;

lazy_static::lazy_static! {
    pub static ref KIND_MAPPINGS: HashMap<&'static str, (UnitType, Option<f32>)> = {
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
        m.insert("minute", (UnitType::Second, Some(60_f32)));
        m.insert("min", (UnitType::Second, Some(60_f32)));
        m.insert("mins", (UnitType::Second, Some(60_f32)));
        m.insert("minutes", (UnitType::Second, Some(60_f32)));

        // Hour
        m.insert("hour", (UnitType::Second, Some(60_f32*60_f32)));
        m.insert("hours", (UnitType::Second, Some(60_f32*60_f32)));
        m.insert("hr", (UnitType::Second, Some(60_f32*60_f32)));
        m.insert("h", (UnitType::Second, Some(60_f32*60_f32)));

        // Day
        m.insert("day", (UnitType::Second, Some(60_f32*60_f32*24_f32)));
        m.insert("days", (UnitType::Second, Some(60_f32*60_f32*24_f32)));
        m.insert("d", (UnitType::Second, Some(60_f32*60_f32*24_f32)));

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

    pub static ref PREFIX_MAPPING: HashMap<&'static str, f32> = {
        let mut m: HashMap<&str, f32> = HashMap::new();
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
