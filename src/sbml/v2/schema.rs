//! SBML v2 annotation structures for EnzymeML
//!
//! This module contains the classes for the EnzymeML version 2 format annotations
//! used to map the EnzymeML JSON schema to SBML.
//!
//! The classes in this module define the structure for annotating SBML models with
//! EnzymeML-specific information. These annotations allow for the representation of
//! enzymatic reactions, experimental data, and associated metadata in a standardized format.
//!
//! Each annotation class corresponds to a specific aspect of enzymatic data, such as
//! small molecules, proteins, complexes, experimental measurements, and parameters.

use serde::{Deserialize, Serialize};

use crate::prelude::DataTypes;

const ENZYMEML_V2_NS: &str = "https://www.enzymeml.org/v2";

fn default_xmlns() -> String {
    ENZYMEML_V2_NS.to_string()
}

/// Top-level annotation class for EnzymeML version 2.
///
/// This struct serves as a container for all other annotation types and
/// is attached to SBML elements to provide EnzymeML-specific information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "annotation")]
pub struct V2Annotation {
    /// Annotation for small molecules.
    #[serde(
        rename = "smallMolecule",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub small_molecule: Option<SmallMoleculeAnnot>,

    /// Annotation for proteins.
    #[serde(rename = "protein", default, skip_serializing_if = "Option::is_none")]
    pub protein: Option<ProteinAnnot>,

    /// Annotation for complexes.
    #[serde(rename = "complex", default, skip_serializing_if = "Option::is_none")]
    pub complex: Option<ComplexAnnot>,

    /// Annotation for experimental data.
    #[serde(rename = "data", default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataAnnot>,

    /// Annotation for parameters.
    #[serde(rename = "parameter", default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<ParameterAnnot>,

    /// Annotation for variables.
    #[serde(rename = "variables", default, skip_serializing_if = "Option::is_none")]
    pub variables: Option<VariablesAnnot>,
}

/// Represents the annotation for a small molecule in the EnzymeML format.
///
/// This struct contains chemical identifiers that help uniquely identify
/// small molecules involved in enzymatic reactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "smallMolecule")]
pub struct SmallMoleculeAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The InChIKey of the small molecule.
    #[serde(rename = "inchiKey", default, skip_serializing_if = "Option::is_none")]
    pub inchikey: Option<String>,

    /// The canonical SMILES representation of the small molecule.
    #[serde(rename = "smiles", default, skip_serializing_if = "Option::is_none")]
    pub canonical_smiles: Option<String>,
}

/// Represents the annotation for a protein in the EnzymeML format.
///
/// This struct contains biological information about proteins, including
/// their enzymatic classification, origin, and sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "protein")]
pub struct ProteinAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The EC number of the protein.
    #[serde(rename = "ecnumber", default, skip_serializing_if = "Option::is_none")]
    pub ecnumber: Option<String>,

    /// The organism from which the protein is derived.
    #[serde(rename = "organism", default, skip_serializing_if = "Option::is_none")]
    pub organism: Option<String>,

    /// The taxonomic ID of the organism.
    #[serde(
        rename = "organismTaxId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism_tax_id: Option<String>,

    /// The amino acid sequence of the protein.
    #[serde(rename = "sequence", default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<String>,
}

/// Represents the annotation for a complex in the EnzymeML format.
///
/// This struct describes molecular complexes formed by multiple components,
/// such as protein-protein or protein-substrate complexes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "complex")]
pub struct ComplexAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// A list of participants in the complex.
    #[serde(rename = "participants", default)]
    pub participants: Vec<String>,
}

/// Represents the annotation for a modifier in the EnzymeML format.
///
/// This struct describes the modifier of a reaction, including its role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "modifier")]
pub struct ModifierAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The role of the modifier.
    #[serde(rename = "@modifierRole")]
    pub modifier_role: String,
}

/// Represents the annotation for data in the EnzymeML format.
///
/// This struct links experimental data files to measurements and provides
/// methods to convert the data into Measurement objects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "data")]
pub struct DataAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The file associated with the data.
    #[serde(rename = "@file")]
    pub file: String,

    /// A list of measurements associated with the data.
    #[serde(rename = "measurement", default)]
    pub measurements: Vec<MeasurementAnnot>,
}

/// Represents the annotation for a measurement in the EnzymeML format.
///
/// This struct describes experimental measurements, including conditions,
/// time units, and species data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "measurement")]
pub struct MeasurementAnnot {
    /// The ID of the measurement.
    #[serde(rename = "@id")]
    pub id: String,

    /// The name of the measurement.
    #[serde(rename = "@name", default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The unit of time.
    #[serde(rename = "@timeUnit", default, skip_serializing_if = "Option::is_none")]
    pub time_unit: Option<String>,

    /// The conditions associated with the measurement.
    #[serde(
        rename = "conditions",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub conditions: Option<ConditionsAnnot>,

    /// A list of species data associated with the measurement.
    #[serde(rename = "speciesData", default)]
    pub species_data: Vec<SpeciesDataAnnot>,
}

/// Represents the annotation for species data in the EnzymeML format.
///
/// This struct describes data associated with a specific species, including
/// its initial value, data type, and unit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "speciesData")]
pub struct SpeciesDataAnnot {
    /// The ID of the species.
    #[serde(rename = "@species")]
    pub species_id: String,

    /// The value associated with the species.
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<f64>,

    /// The type of data (default is "CONCENTRATION").
    #[serde(rename = "@type", default = "default_data_type")]
    pub data_type: DataTypes,

    /// The unit of the value.
    #[serde(rename = "@unit")]
    pub unit: String,
}

fn default_data_type() -> DataTypes {
    DataTypes::Concentration
}

/// Represents the annotation for conditions in the EnzymeML format.
///
/// This struct describes experimental conditions such as pH and temperature
/// under which measurements were taken.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "conditions")]
pub struct ConditionsAnnot {
    /// The pH conditions.
    #[serde(rename = "ph", default, skip_serializing_if = "Option::is_none")]
    pub ph: Option<PHAnnot>,

    /// The temperature conditions.
    #[serde(
        rename = "temperature",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub temperature: Option<TemperatureAnnot>,
}

/// Represents the annotation for pH in the EnzymeML format.
///
/// This struct describes the pH value of an experimental condition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "ph")]
pub struct PHAnnot {
    /// The pH value.
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
}

/// Represents the annotation for temperature in the EnzymeML format.
///
/// This struct describes the temperature value and unit of an experimental condition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "temperature")]
pub struct TemperatureAnnot {
    /// The temperature value.
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,

    /// The unit of the temperature value.
    #[serde(rename = "@unit", default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Represents the annotation for a parameter in the EnzymeML format.
///
/// This struct describes statistical properties of parameters used in
/// kinetic models, such as bounds and standard error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "parameter")]
pub struct ParameterAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The lower bound of the parameter.
    #[serde(
        rename = "lowerBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub lower_bound: Option<f64>,

    /// The upper bound of the parameter.
    #[serde(
        rename = "upperBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub upper_bound: Option<f64>,

    /// The standard deviation of the parameter.
    #[serde(
        rename = "stdDeviation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub stderr: Option<f64>,
}

/// Represents the annotation for variables in the EnzymeML format.
///
/// This struct serves as a container for variable annotations used in
/// kinetic models and equations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "variables")]
pub struct VariablesAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// A list of variables.
    #[serde(rename = "variable", default)]
    pub variables: Vec<VariableAnnot>,
}

/// Represents a variable in the EnzymeML format.
///
/// This struct describes variables used in kinetic models and equations,
/// including their identifiers and symbols.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "variable")]
pub struct VariableAnnot {
    /// The ID of the variable.
    #[serde(rename = "@id", default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The name of the variable.
    #[serde(rename = "@name", default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The symbol of the variable.
    #[serde(rename = "@symbol", default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses an XML string containing SBML v2 annotations.
    ///
    /// # Arguments
    /// * `xml` - The XML string to parse
    ///
    /// # Returns
    /// * `Result<V2Annotation, quick_xml::DeError>` - The parsed annotation or an error
    fn parse_v2_annotation(xml: &str) -> Result<V2Annotation, quick_xml::DeError> {
        quick_xml::de::from_str(xml)
    }

    #[test]
    fn test_parse_protein_annotation() {
        let xml = r#"
        <annotation>
            <protein xmlns="https://www.enzymeml.org/v2">
                <sequence>MKLLVL</sequence>
                <ecnumber>1.1.1.1</ecnumber>
                <organism>E. coli</organism>
                <organismTaxId>511145</organismTaxId>
            </protein>
        </annotation>
        "#;

        let annotation = parse_v2_annotation(xml).unwrap();
        assert!(annotation.protein.is_some());

        let protein = annotation.protein.unwrap();
        assert_eq!(protein.sequence, Some("MKLLVL".to_string()));
        assert_eq!(protein.ecnumber, Some("1.1.1.1".to_string()));
        assert_eq!(protein.organism, Some("E. coli".to_string()));
        assert_eq!(protein.organism_tax_id, Some("511145".to_string()));
    }

    #[test]
    fn test_parse_small_molecule_annotation() {
        let xml = r#"
        <annotation>
            <smallMolecule xmlns="https://www.enzymeml.org/v2">
                <inchiKey>WQZGKKKJIJFFOK-GASJEMHNSA-N</inchiKey>
                <smiles>C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O</smiles>
            </smallMolecule>
        </annotation>
        "#;

        let annotation = parse_v2_annotation(xml).unwrap();
        assert!(annotation.small_molecule.is_some());

        let small_molecule = annotation.small_molecule.unwrap();
        assert_eq!(
            small_molecule.inchikey,
            Some("WQZGKKKJIJFFOK-GASJEMHNSA-N".to_string())
        );
        assert!(small_molecule.canonical_smiles.is_some());
    }

    #[test]
    fn test_parse_parameter_annotation() {
        let xml = r#"
        <annotation>
            <parameter xmlns="https://www.enzymeml.org/v2">
                <lowerBound>0.1</lowerBound>
                <upperBound>10.0</upperBound>
                <stdDeviation>0.5</stdDeviation>
            </parameter>
        </annotation>
        "#;

        let annotation = parse_v2_annotation(xml).unwrap();
        assert!(annotation.parameter.is_some());

        let parameter = annotation.parameter.unwrap();
        assert_eq!(parameter.lower_bound, Some(0.1));
        assert_eq!(parameter.upper_bound, Some(10.0));
        assert_eq!(parameter.stderr, Some(0.5));
    }

    #[test]
    fn test_parse_complex_annotation() {
        let xml = r#"
        <annotation>
            <complex xmlns="https://www.enzymeml.org/v2">
                <participants>p0</participants>
                <participants>s0</participants>
            </complex>
        </annotation>
        "#;

        let annotation = parse_v2_annotation(xml).unwrap();
        assert!(annotation.complex.is_some());

        let complex = annotation.complex.unwrap();
        assert_eq!(
            complex.participants,
            vec!["p0".to_string(), "s0".to_string()]
        );
    }
}
