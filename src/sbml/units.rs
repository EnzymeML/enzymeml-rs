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
//!
//! The module also provides utility functions for handling unit definitions in SBML models,
//! such as `map_unit_definition` for creating or referencing unit definitions in an SBML model,
//! and `replace_slashes` for converting unit identifiers between formats.

use regex::Regex;
use sbml::prelude::{
    Model as SBMLModel, Unit as SBMLUnit, UnitDefinition as SBMLUnitDefinition,
    UnitKind as SBMLUnitKind,
};

use crate::{
    prelude::{BaseUnit, UnitDefinition, UnitType},
    sbml::error::SBMLError,
};

/// Creates or references a unit definition in an SBML model
///
/// This function either creates a new unit definition in the SBML model based on the provided
/// EnzymeML UnitDefinition, or returns the ID of an existing unit definition if one with the
/// same ID already exists.
///
/// # Arguments
///
/// * `model` - The SBML model to create the unit definition in
/// * `unit_definition` - The EnzymeML UnitDefinition to convert
///
/// # Returns
///
/// The ID of the created or referenced unit definition
///
/// # Errors
///
/// Returns an `SBMLError::MissingUnitDefinitionId` if the UnitDefinition has no ID
pub(crate) fn map_unit_definition(
    model: &SBMLModel,
    unit_definition: &UnitDefinition,
) -> Result<String, SBMLError> {
    if unit_definition.id.is_none() {
        return Err(SBMLError::MissingUnitDefinitionId(
            serde_json::to_string(unit_definition).unwrap(),
        ));
    }

    let id = replace_slashes(unit_definition.id.as_ref().unwrap());
    let name = unit_definition.name.clone().unwrap_or(id.clone());

    if model.get_unit_definition(&id).is_some() {
        // Unit definition already exists, do nothing
        return Ok(id.clone());
    }

    let unit_def = model.create_unit_definition(&id, &name);

    for base_unit in unit_definition.base_units.iter() {
        unit_def
            .build_unit((&base_unit.kind).into())
            .exponent(base_unit.exponent as i32)
            .multiplier(base_unit.multiplier.unwrap_or(1.0))
            .scale(base_unit.scale.unwrap_or(1.0) as i32);
    }

    Ok(id.clone())
}

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

/// Replaces slashes in unit identifiers with double underscores
///
/// This function is used to transform unit identifiers that may contain slashes (e.g., "mol/L")
/// into valid SBML identifiers by replacing slashes with double underscores (e.g., "mol__L").
///
/// # Arguments
///
/// * `id` - The identifier string to transform
///
/// # Returns
///
/// A new string with slashes replaced by double underscores
pub(crate) fn replace_slashes(id: &str) -> String {
    let pattern = Regex::new(r"[ ]*\/[ ]*").unwrap();
    pattern.replace(id, "__").to_string()
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

/// Converts an EnzymeML UnitType to an SBML UnitKind
///
/// This implementation provides a reference-based conversion from EnzymeML UnitType to SBML UnitKind.
/// It delegates to the implementation for owned UnitType.
impl From<&UnitType> for SBMLUnitKind {
    fn from(unit_type: &UnitType) -> Self {
        unit_type.clone().into()
    }
}

/// Converts an EnzymeML UnitType to an SBML UnitKind
///
/// This implementation maps each EnzymeML UnitType variant to the appropriate SBML UnitKind.
/// Note that EnzymeML "Litre" is mapped to SBML "Litre", and EnzymeML "Metre" is mapped to SBML "Metre".
impl From<UnitType> for SBMLUnitKind {
    fn from(unit_type: UnitType) -> Self {
        match unit_type {
            UnitType::Hertz => SBMLUnitKind::Hertz,
            UnitType::Item => SBMLUnitKind::Item,
            UnitType::Joule => SBMLUnitKind::Joule,
            UnitType::Katal => SBMLUnitKind::Katal,
            UnitType::Kelvin => SBMLUnitKind::Kelvin,
            UnitType::Kilogram => SBMLUnitKind::Kilogram,
            UnitType::Litre => SBMLUnitKind::Litre,
            UnitType::Lumen => SBMLUnitKind::Lumen,
            UnitType::Lux => SBMLUnitKind::Lux,
            UnitType::Metre => SBMLUnitKind::Metre,
            UnitType::Mole => SBMLUnitKind::Mole,
            UnitType::Newton => SBMLUnitKind::Newton,
            UnitType::Ohm => SBMLUnitKind::Ohm,
            UnitType::Pascal => SBMLUnitKind::Pascal,
            UnitType::Radian => SBMLUnitKind::Radian,
            UnitType::Second => SBMLUnitKind::Second,
            UnitType::Siemens => SBMLUnitKind::Siemens,
            UnitType::Sievert => SBMLUnitKind::Sievert,
            UnitType::Steradian => SBMLUnitKind::Steradian,
            UnitType::Tesla => SBMLUnitKind::Tesla,
            UnitType::Volt => SBMLUnitKind::Volt,
            UnitType::Watt => SBMLUnitKind::Watt,
            UnitType::Weber => SBMLUnitKind::Weber,
            UnitType::Ampere => SBMLUnitKind::Ampere,
            UnitType::Avogadro => SBMLUnitKind::Avogadro,
            UnitType::Becquerel => SBMLUnitKind::Becquerel,
            UnitType::Candela => SBMLUnitKind::Candela,
            UnitType::Celsius => SBMLUnitKind::Celsius,
            UnitType::Coulomb => SBMLUnitKind::Coulomb,
            UnitType::Dimensionless => SBMLUnitKind::Dimensionless,
            UnitType::Farad => SBMLUnitKind::Farad,
            UnitType::Gram => SBMLUnitKind::Gram,
            UnitType::Gray => SBMLUnitKind::Gray,
            UnitType::Henry => SBMLUnitKind::Henry,
        }
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
