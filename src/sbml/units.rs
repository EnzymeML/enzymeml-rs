//! SBML Units Conversion Module
//!
//! This module provides conversion implementations between SBML (Systems Biology Markup Language)
//! unit representations and our internal EnzymeML unit representation system.
//!
//! It includes trait implementations for converting:
//! - SBML UnitDefinition to EnzymeML UnitDefinition
//! - SBML Unit to EnzymeML BaseUnit
//! - SBML UnitKind to EnzymeML UnitType
//!
//! These conversions ensure proper handling of units when parsing SBML models into EnzymeML format.

use sbml::prelude::{
    Unit as SBMLUnit, UnitDefinition as SBMLUnitDefinition, UnitKind as SBMLUnitKind,
};

use crate::{
    prelude::{BaseUnit, UnitDefinition, UnitType},
    sbml::error::SBMLError,
};

/// Converts an SBML UnitDefinition to our EnzymeML UnitDefinition
///
/// This implementation extracts the id, name, and all base units from an SBML UnitDefinition
/// and constructs an equivalent EnzymeML UnitDefinition.
///
/// # Errors
///
/// Returns an `SBMLError` if any of the base units cannot be converted.
impl TryFrom<&SBMLUnitDefinition<'_>> for UnitDefinition {
    type Error = SBMLError;

    fn try_from(unit_definition: &SBMLUnitDefinition<'_>) -> Result<Self, Self::Error> {
        let base_units = unit_definition
            .units()
            .iter()
            .map(|unit| unit.as_ref().try_into())
            .collect::<Result<Vec<BaseUnit>, SBMLError>>()?;

        Ok(UnitDefinition {
            id: Some(unit_definition.id()),
            name: unit_definition.name(),
            base_units,
        })
    }
}

/// Converts an SBML Unit to our EnzymeML BaseUnit
///
/// This implementation extracts the kind, exponent, multiplier, and scale from an SBML Unit
/// and constructs an equivalent EnzymeML BaseUnit.
///
/// # Errors
///
/// Returns an `SBMLError` if the unit kind cannot be converted to a valid UnitType.
impl TryFrom<&SBMLUnit<'_>> for BaseUnit {
    type Error = SBMLError;

    fn try_from(unit: &SBMLUnit<'_>) -> Result<Self, Self::Error> {
        Ok(BaseUnit {
            kind: unit.kind().try_into()?,
            exponent: unit.exponent().into(),
            multiplier: unit.multiplier().into(),
            scale: Some(unit.scale().into()),
        })
    }
}

/// Converts a reference to an SBML UnitKind to our EnzymeML UnitType
///
/// This is a convenience implementation that delegates to the implementation for owned UnitKind.
///
/// # Errors
///
/// Returns an `SBMLError` if the unit kind cannot be converted to a valid UnitType.
impl TryFrom<&SBMLUnitKind> for UnitType {
    type Error = SBMLError;

    fn try_from(unit_kind: &SBMLUnitKind) -> Result<Self, Self::Error> {
        unit_kind.try_into()
    }
}

/// Converts an SBML UnitKind to our EnzymeML UnitType
///
/// This implementation maps each SBML UnitKind variant to the appropriate EnzymeML UnitType.
/// Note that both "Liter" and "Litre" from SBML are mapped to "Litre" in EnzymeML,
/// and both "Meter" and "Metre" are mapped to "Metre".
///
/// # Errors
///
/// Returns an `SBMLError::InvalidUnitKind` if the SBML UnitKind doesn't have a corresponding
/// EnzymeML UnitType (e.g., for SBMLUnitKind::Invalid).
impl TryFrom<SBMLUnitKind> for UnitType {
    type Error = SBMLError;

    fn try_from(unit_kind: SBMLUnitKind) -> Result<Self, Self::Error> {
        let unit_type = match unit_kind {
            SBMLUnitKind::Dimensionless => UnitType::Dimensionless,
            SBMLUnitKind::Ampere => UnitType::Ampere,
            SBMLUnitKind::Avogadro => UnitType::Avogadro,
            SBMLUnitKind::Becquerel => UnitType::Becquerel,
            SBMLUnitKind::Candela => UnitType::Candela,
            SBMLUnitKind::Celsius => UnitType::Celsius,
            SBMLUnitKind::Coulomb => UnitType::Coulomb,
            SBMLUnitKind::Farad => UnitType::Farad,
            SBMLUnitKind::Gram => UnitType::Gram,
            SBMLUnitKind::Gray => UnitType::Gray,
            SBMLUnitKind::Henry => UnitType::Henry,
            SBMLUnitKind::Hertz => UnitType::Hertz,
            SBMLUnitKind::Item => UnitType::Item,
            SBMLUnitKind::Joule => UnitType::Joule,
            SBMLUnitKind::Katal => UnitType::Katal,
            SBMLUnitKind::Kelvin => UnitType::Kelvin,
            SBMLUnitKind::Kilogram => UnitType::Kilogram,
            SBMLUnitKind::Liter => UnitType::Litre,
            SBMLUnitKind::Litre => UnitType::Litre,
            SBMLUnitKind::Lumen => UnitType::Lumen,
            SBMLUnitKind::Lux => UnitType::Lux,
            SBMLUnitKind::Meter => UnitType::Metre,
            SBMLUnitKind::Metre => UnitType::Metre,
            SBMLUnitKind::Mole => UnitType::Mole,
            SBMLUnitKind::Newton => UnitType::Newton,
            SBMLUnitKind::Ohm => UnitType::Ohm,
            SBMLUnitKind::Pascal => UnitType::Pascal,
            SBMLUnitKind::Radian => UnitType::Radian,
            SBMLUnitKind::Second => UnitType::Second,
            SBMLUnitKind::Siemens => UnitType::Siemens,
            SBMLUnitKind::Sievert => UnitType::Sievert,
            SBMLUnitKind::Steradian => UnitType::Steradian,
            SBMLUnitKind::Tesla => UnitType::Tesla,
            SBMLUnitKind::Volt => UnitType::Volt,
            SBMLUnitKind::Watt => UnitType::Watt,
            SBMLUnitKind::Weber => UnitType::Weber,
            _ => return Err(SBMLError::InvalidUnitKind(unit_kind.to_string())),
        };

        Ok(unit_type)
    }
}

#[cfg(test)]
mod tests {
    use sbml::SBMLDocument;

    use super::*;

    /// Tests conversion from SBML UnitKind to EnzymeML UnitType
    /// Verifies that Meter in SBML is correctly converted to Metre in EnzymeML
    #[test]
    fn test_unit_type_from_sbml_unit_kind() {
        let unit_kind = SBMLUnitKind::Meter;
        let unit_type = UnitType::try_from(unit_kind).unwrap();

        assert_eq!(unit_type, UnitType::Metre);
    }

    /// Tests that conversion fails for invalid SBML UnitKind
    /// This test expects a panic when trying to convert an invalid unit kind
    #[test]
    #[should_panic]
    fn test_unit_type_from_sbml_unit_kind_invalid() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let unit_def = model
            .build_unit_definition("test", "test")
            .unit(
                SBMLUnitKind::Invalid,
                Some(1),
                Some(1),
                Some(1.0),
                Some(1.0),
            )
            .build();

        let _: UnitDefinition = unit_def.as_ref().try_into().unwrap();
    }

    /// Tests conversion from SBML UnitDefinition to EnzymeML UnitDefinition
    /// Creates a Molar unit (mole per liter) and verifies all properties
    /// are correctly transferred during conversion
    #[test]
    fn test_unit_definition_from_sbml_unit_definition() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let unit_def = model
            .build_unit_definition("M", "Molar")
            .unit(SBMLUnitKind::Mole, Some(1), None, Some(1.0), None)
            .unit(SBMLUnitKind::Litre, Some(-1), None, Some(1.0), None)
            .build();

        let enzml_unit: UnitDefinition = unit_def
            .as_ref()
            .try_into()
            .expect("Failed to convert SBML unit definition to EnzymeML unit definition");

        assert_eq!(enzml_unit.id, Some("M".to_string()));
        assert_eq!(enzml_unit.name, Some("Molar".to_string()));
        assert_eq!(enzml_unit.base_units.len(), 2);
        assert_eq!(enzml_unit.base_units[0].kind, UnitType::Mole);
        assert_eq!(enzml_unit.base_units[0].exponent, 1);
        assert_eq!(enzml_unit.base_units[0].multiplier, Some(1.0));
        assert_eq!(enzml_unit.base_units[0].scale, Some(1.0));
        assert_eq!(enzml_unit.base_units[1].kind, UnitType::Litre);
        assert_eq!(enzml_unit.base_units[1].exponent, -1);
        assert_eq!(enzml_unit.base_units[1].multiplier, Some(1.0));
        assert_eq!(enzml_unit.base_units[1].scale, Some(1.0));
    }
}
