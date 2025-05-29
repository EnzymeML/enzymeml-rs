//! SBML v1 annotation structures for EnzymeML
//!
//! This module provides Rust equivalents of the Python pydantic-xml structures
//! for parsing SBML v1 format EnzymeML annotations using quick-xml with serde.

use serde::{Deserialize, Deserializer, Serialize};

use crate::sbml::v2::schema::VariableAnnot;

const ENZYMEML_V1_NS: &str = "http://sbml.org/enzymeml/version1";

fn default_xmlns() -> String {
    ENZYMEML_V1_NS.to_string()
}

/// Represents the top-level annotation in the EnzymeML v1 format.
///
/// This struct contains all the different types of annotations that can be
/// attached to elements in an SBML model using EnzymeML v1 format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "annotation")]
pub struct V1Annotation {
    /// Annotation for small molecules.
    #[serde(
        rename = "smallMolecule",
        alias = "enzymeml:reactant",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub small_molecule: Option<ReactantAnnot>,

    /// Annotation for proteins.
    #[serde(
        rename = "protein",
        alias = "enzymeml:protein",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub protein: Option<ProteinAnnot>,

    /// Annotation for complexes.
    #[serde(
        rename = "complex",
        alias = "enzymeml:complex",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub complex: Option<ComplexAnnot>,

    /// Annotation for experimental data.
    #[serde(
        rename = "data",
        alias = "enzymeml:data",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub data: Option<DataAnnot>,

    /// Annotation for parameters.
    #[serde(
        rename = "parameter",
        alias = "enzymeml:parameter",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub parameter: Option<ParameterAnnot>,

    /// Annotation for variables.
    #[serde(
        rename = "variables",
        alias = "enzymeml:variables",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub variables: Option<VariableAnnot>,
}

/// Represents the annotation for a parameter in the EnzymeML format.
///
/// This struct contains information about parameter bounds and initial values
/// that can be used in parameter estimation or simulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "parameter")]
pub struct ParameterAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The initial value of the parameter.
    #[serde(
        rename = "initialValue",
        alias = "enzymeml:initialValue",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub initial: Option<f64>,

    /// The upper bound of the parameter.
    #[serde(
        rename = "upperBound",
        alias = "enzymeml:upperBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub upper: Option<f64>,

    /// The lower bound of the parameter.
    #[serde(
        rename = "lowerBound",
        alias = "enzymeml:lowerBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub lower: Option<f64>,
}

/// Represents the annotation for a complex in the EnzymeML format.
///
/// A complex is formed by multiple participants (species) that interact
/// with each other.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "complex")]
pub struct ComplexAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// A list of participants in the complex.
    #[serde(rename = "participant", alias = "enzymeml:participant", default)]
    pub participants: Vec<String>,
}

/// Represents the annotation for a reactant in the EnzymeML format.
///
/// This struct contains chemical identifiers for small molecules that
/// participate in reactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "reactant")]
pub struct ReactantAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The InChI of the reactant.
    #[serde(
        rename = "inchi",
        alias = "enzymeml:inchi",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub inchi: Option<String>,

    /// The SMILES representation of the reactant.
    #[serde(
        rename = "smiles",
        alias = "enzymeml:smiles",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub smiles: Option<String>,

    /// The ChEBI ID of the reactant.
    #[serde(
        rename = "chebiID",
        alias = "enzymeml:chebiID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub chebi_id: Option<String>,
}

/// Represents the annotation for a protein in the EnzymeML format.
///
/// This struct contains biological identifiers and properties of proteins,
/// particularly enzymes that catalyze reactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "protein")]
pub struct ProteinAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// The amino acid sequence of the protein.
    #[serde(
        rename = "sequence",
        alias = "enzymeml:sequence",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub sequence: Option<String>,

    /// The EC number of the protein.
    #[serde(
        rename = "ECnumber",
        alias = "enzymeml:ECnumber",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub ecnumber: Option<String>,

    /// The UniProt ID of the protein.
    #[serde(
        rename = "uniprotID",
        alias = "enzymeml:uniprotID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub uniprotid: Option<String>,

    /// The organism from which the protein is derived.
    #[serde(
        rename = "organism",
        alias = "enzymeml:organism",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism: Option<String>,

    /// The taxonomic ID of the organism.
    #[serde(
        rename = "organismTaxID",
        alias = "enzymeml:organismTaxID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism_tax_id: Option<String>,
}

/// Represents the annotation for experimental data in the EnzymeML format.
///
/// This struct contains information about experimental measurements, including
/// file formats, measurement metadata, and file references.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "data")]
pub struct DataAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// A list of format annotations.
    #[serde(rename = "formats", alias = "enzymeml:formats")]
    pub formats: FormatsWrapper,

    /// A list of measurement annotations.
    #[serde(rename = "listOfMeasurements", alias = "enzymeml:listOfMeasurements")]
    pub measurements: MeasurementsWrapper,

    /// A list of file annotations.
    #[serde(rename = "files", alias = "enzymeml:files")]
    pub files: FilesWrapper,
}

/// Wrapper for formats to handle the wrapped XML structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FormatsWrapper {
    #[serde(rename = "format", alias = "enzymeml:format", default)]
    pub format: Vec<FormatAnnot>,
}

/// Wrapper for measurements to handle the wrapped XML structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MeasurementsWrapper {
    #[serde(rename = "measurement", alias = "enzymeml:measurement", default)]
    pub measurement: Vec<MeasurementAnnot>,
}

/// Wrapper for files to handle the wrapped XML structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FilesWrapper {
    #[serde(rename = "file", alias = "enzymeml:file", default)]
    pub file: Vec<FileAnnot>,
}

/// Represents the format annotation in the EnzymeML format.
///
/// This struct defines the structure of data files, specifying how columns
/// in data files map to species, time, or other measurements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "format")]
pub struct FormatAnnot {
    /// The ID of the format.
    #[serde(rename = "@id")]
    pub id: String,

    /// A list of column annotations.
    #[serde(rename = "column", alias = "enzymeml:column", default)]
    pub columns: Vec<ColumnAnnot>,
}

/// Represents the column annotation in the EnzymeML format.
///
/// This struct defines how a specific column in a data file should be interpreted,
/// including what species it refers to, what type of data it contains, and its units.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "column")]
pub struct ColumnAnnot {
    /// The ID of the species.
    #[serde(rename = "@species", default, skip_serializing_if = "Option::is_none")]
    pub species_id: Option<String>,

    /// The type of the column.
    #[serde(rename = "@type")]
    pub column_type: String,

    /// The unit of the column.
    #[serde(rename = "@unit")]
    pub unit: String,

    /// The index of the column.
    #[serde(rename = "@index")]
    pub index: usize,

    /// The replica of the column.
    #[serde(rename = "@replica", default, skip_serializing_if = "Option::is_none")]
    pub replica: Option<String>,

    /// Whether the column is calculated.
    #[serde(
        rename = "@isCalculated",
        default,
        deserialize_with = "deserialize_python_bool"
    )]
    pub is_calculated: bool,
}

/// Represents the measurement annotation in the EnzymeML format.
///
/// This struct contains metadata about a specific measurement, including
/// its identifier, name, associated file, and initial concentrations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "measurement")]
pub struct MeasurementAnnot {
    /// The ID of the measurement.
    #[serde(rename = "@id")]
    pub id: String,

    /// The name of the measurement.
    #[serde(rename = "@name")]
    pub name: String,

    /// The file associated with the measurement.
    #[serde(rename = "@file")]
    pub file: String,

    /// A list of initial concentration annotations.
    #[serde(rename = "initConc", alias = "enzymeml:initConc", default)]
    pub init_concs: Vec<InitConcAnnot>,
}

/// Represents the initial concentration annotation in the EnzymeML format.
///
/// This struct defines the initial concentration of a species (either a protein
/// or reactant) in a specific measurement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "initConc")]
pub struct InitConcAnnot {
    /// The protein associated with the initial concentration.
    #[serde(rename = "@protein", default, skip_serializing_if = "Option::is_none")]
    pub protein: Option<String>,

    /// The reactant associated with the initial concentration.
    #[serde(rename = "@reactant", default, skip_serializing_if = "Option::is_none")]
    pub reactant: Option<String>,

    /// The value of the initial concentration.
    #[serde(rename = "@value", default)]
    pub value: f64,

    /// The unit of the initial concentration.
    #[serde(rename = "@unit")]
    pub unit: String,
}

/// Represents the file annotation in the EnzymeML format.
///
/// This struct contains metadata about a data file, including its identifier,
/// location, and format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "file")]
pub struct FileAnnot {
    /// The ID of the file.
    #[serde(rename = "@id")]
    pub id: String,

    /// The file path.
    #[serde(rename = "@file")]
    pub location: String,

    /// The format of the file.
    #[serde(rename = "@format")]
    pub format: String,
}

/// Custom deserializer for Python-style boolean strings.
///
/// This function handles the deserialization of boolean values that may come
/// as Python-style strings "True" and "False" (case-insensitive) or as regular
/// boolean values.
fn deserialize_python_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct BoolVisitor;

    impl<'de> Visitor<'de> for BoolVisitor {
        type Value = bool;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a boolean or a string representing a boolean")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match value.to_lowercase().as_str() {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(E::custom(format!("Invalid boolean string: {}", value))),
            }
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            self.visit_str(&value)
        }
    }

    deserializer.deserialize_any(BoolVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    use quick_xml::de::from_str;

    pub fn parse_v1_annotation(xml: &str) -> Result<V1Annotation, quick_xml::DeError> {
        from_str(xml)
    }

    #[test]
    fn test_parse_protein_annotation() {
        let xml = r#"
        <annotation>
            <protein xmlns="http://sbml.org/enzymeml/version2">
                <sequence>MKLLVL</sequence>
                <ECnumber>1.1.1.1</ECnumber>
            </protein>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.protein.is_some());

        let protein = annotation.protein.unwrap();
        assert_eq!(protein.sequence, Some("MKLLVL".to_string()));
        assert_eq!(protein.ecnumber, Some("1.1.1.1".to_string()));
    }

    #[test]
    fn test_parse_protein_annotation_with_enzymeml_prefix() {
        let xml = r#"
        <annotation>
            <enzymeml:protein xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:sequence>MRRLAACLLATAVAAATSAALAQTSPMTPDITGKPFVAGDAANDYVKREVMIPMRDGVKLHTVIVLPKGARNAPIVLTRTPYDASGRTERLASPHMKDLLSAGDDVFVEGGYIRVFQDVRGKYGSEGDYVMTRPLRGPLNPSEVDHATDAWDTIDWLVKNVKESNGKVGMIGSSYEGFTVVMALTNPHPALKVAAPESPMIDGWMGDDWFNYGAFRQVNFDYFTGQLSKRGKGAGIPRQGHDDYSNFLQAGSAGDFAKAAGLEQLPWWHKLTEHAAYDSFWQEQALDKVMARTPLKVPTMWLQGLWDQEDMWGAIHSYAAMEPRDKSNKLNYLVMGPWRHSQVNSDASSLGALNFDGDTARQFRRDVLRPFFDQYLVDGAPKAATPPVFIYNTGENHWDRLQAWPRSCDKGCAAKSKPLYLQAGGKLSFQAPTAAQPAFEEYVSDPAKPVPFVPRPVDFGDRSMWTTWLVHDQRFVDGRPDVLTFVTEPLTAPLQIAGAPDVHLQASTSGSDSDWVVKLIDVYPDEMAADPKMGGYELPVSMAIFRGRYRESFSTPAPLAANQPLAFQFGLPTANHTFQPGHRVMVQVQSSLFPLYDRNPQTYVPNVFFAKPGDYQKATQRVYVAPGQGSYISLPVR</enzymeml:sequence>
                <enzymeml:ECnumber>3.1.1.43</enzymeml:ECnumber>
                <enzymeml:uniprotID>B0RS62</enzymeml:uniprotID>
                <enzymeml:organism>Xanthomonas campestris pv. campestris</enzymeml:organism>
            </enzymeml:protein>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.protein.is_some());

        let protein = annotation.protein.unwrap();
        assert!(protein.sequence.is_some());
        assert_eq!(protein.ecnumber, Some("3.1.1.43".to_string()));
        assert_eq!(protein.uniprotid, Some("B0RS62".to_string()));
        assert_eq!(
            protein.organism,
            Some("Xanthomonas campestris pv. campestris".to_string())
        );
    }

    #[test]
    fn test_parse_reactant_annotation() {
        let xml = r#"
        <annotation>
            <smallMolecule xmlns="http://sbml.org/enzymeml/version2">
                <inchi>InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1</inchi>
                <smiles>C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O</smiles>
                <chebiID>CHEBI:4167</chebiID>
            </smallMolecule>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.small_molecule.is_some());

        let reactant = annotation.small_molecule.unwrap();
        assert!(reactant.inchi.is_some());
        assert!(reactant.smiles.is_some());
        assert_eq!(reactant.chebi_id, Some("CHEBI:4167".to_string()));
    }

    #[test]
    fn test_parse_reactant_annotation_with_enzymeml_prefix() {
        let xml = r#"
        <annotation>
            <enzymeml:smallMolecule xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:inchi>1S/C9H11NO2/c1-12-9(11)7-10-8-5-3-2-4-6-8/h2-6,10H,7H2,1H3</enzymeml:inchi>
                <enzymeml:smiles>O(C([C@@H](C1=CC=CC=C1)N([H])[H])=O)C</enzymeml:smiles>
            </enzymeml:smallMolecule>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.small_molecule.is_some());

        let reactant = annotation.small_molecule.unwrap();
        assert_eq!(
            reactant.inchi,
            Some("1S/C9H11NO2/c1-12-9(11)7-10-8-5-3-2-4-6-8/h2-6,10H,7H2,1H3".to_string())
        );
        assert_eq!(
            reactant.smiles,
            Some("O(C([C@@H](C1=CC=CC=C1)N([H])[H])=O)C".to_string())
        );
    }

    #[test]
    fn test_parse_complex_annotation_with_enzymeml_prefix() {
        let xml = r#"
        <annotation>
            <enzymeml:complex xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:participant>p0</enzymeml:participant>
                <enzymeml:participant>s0</enzymeml:participant>
            </enzymeml:complex>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.complex.is_some());

        let complex = annotation.complex.unwrap();
        assert_eq!(
            complex.participants,
            vec!["p0".to_string(), "s0".to_string()]
        );
    }

    #[test]
    fn test_parse_parameter_annotation() {
        let xml = r#"
        <annotation>
            <parameter xmlns="https://www.enzymeml.org/v2">
                <initialValue>1.5</initialValue>
                <upperBound>10.0</upperBound>
                <lowerBound>0.1</lowerBound>
            </parameter>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.parameter.is_some());

        let parameter = annotation.parameter.unwrap();
        assert_eq!(parameter.initial, Some(1.5));
        assert_eq!(parameter.upper, Some(10.0));
        assert_eq!(parameter.lower, Some(0.1));
    }

    #[test]
    fn test_parse_parameter_annotation_with_enzymeml_prefix() {
        let xml = r#"
        <annotation>
            <enzymeml:parameter xmlns:enzymeml="https://www.enzymeml.org/v2">
                <enzymeml:initialValue>2.5</enzymeml:initialValue>
                <enzymeml:upperBound>15.0</enzymeml:upperBound>
                <enzymeml:lowerBound>0.5</enzymeml:lowerBound>
            </enzymeml:parameter>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.parameter.is_some());

        let parameter = annotation.parameter.unwrap();
        assert_eq!(parameter.initial, Some(2.5));
        assert_eq!(parameter.upper, Some(15.0));
        assert_eq!(parameter.lower, Some(0.5));
    }

    #[test]
    fn test_parse_complex_annotation() {
        let xml = r#"
        <annotation>
            <complex xmlns="http://sbml.org/enzymeml/version2">
                <participant>p0</participant>
                <participant>s0</participant>
            </complex>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.complex.is_some());

        let complex = annotation.complex.unwrap();
        assert_eq!(
            complex.participants,
            vec!["p0".to_string(), "s0".to_string()]
        );
    }

    #[test]
    fn test_parse_data_annotation_with_enzymeml_prefix() {
        let xml = r#"
        <annotation>
            <enzymeml:data xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:formats>
                    <enzymeml:format id="format0">
                        <enzymeml:column type="time" unit="u4" index="0"/>
                        <enzymeml:column replica="sub1_repl1" species="s0" type="conc" unit="u1" index="1" isCalculated="false"/>
                    </enzymeml:format>
                </enzymeml:formats>
                <enzymeml:listOfMeasurements>
                    <enzymeml:measurement file="file0" id="m0" name="Test measurement">
                        <enzymeml:initConc protein="p0" value="0.0002" unit="u1"/>
                        <enzymeml:initConc reactant="s0" value="20.0" unit="u1"/>
                    </enzymeml:measurement>
                </enzymeml:listOfMeasurements>
                <enzymeml:files>
                    <enzymeml:file file="./data/m0.csv" format="format0" id="file0"/>
                </enzymeml:files>
            </enzymeml:data>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml).unwrap();
        assert!(annotation.data.is_some());

        let data = annotation.data.unwrap();
        assert_eq!(data.formats.format.len(), 1);
        assert_eq!(data.measurements.measurement.len(), 1);
        assert_eq!(data.files.file.len(), 1);

        let format = &data.formats.format[0];
        assert_eq!(format.id, "format0");
        assert_eq!(format.columns.len(), 2);

        let measurement = &data.measurements.measurement[0];
        assert_eq!(measurement.id, "m0");
        assert_eq!(measurement.name, "Test measurement");
        assert_eq!(measurement.init_concs.len(), 2);

        let file = &data.files.file[0];
        assert_eq!(file.id, "file0");
        assert_eq!(file.location, "./data/m0.csv");
        assert_eq!(file.format, "format0");
    }

    #[test]
    fn test_parse_column_with_python_boolean_strings() {
        // Test with "True" string
        let xml_true = r#"
        <annotation>
            <enzymeml:data xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:formats>
                    <enzymeml:format id="format0">
                        <enzymeml:column type="time" unit="u4" index="0" isCalculated="True"/>
                    </enzymeml:format>
                </enzymeml:formats>
                <enzymeml:listOfMeasurements>
                    <enzymeml:measurement file="file0" id="m0" name="Test measurement">
                    </enzymeml:measurement>
                </enzymeml:listOfMeasurements>
                <enzymeml:files>
                    <enzymeml:file file="./data/m0.csv" format="format0" id="file0"/>
                </enzymeml:files>
            </enzymeml:data>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml_true).unwrap();
        assert!(annotation.data.is_some());
        let data = annotation.data.unwrap();
        assert_eq!(data.formats.format[0].columns[0].is_calculated, true);

        // Test with "False" string
        let xml_false = r#"
        <annotation>
            <enzymeml:data xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:formats>
                    <enzymeml:format id="format0">
                        <enzymeml:column type="time" unit="u4" index="0" isCalculated="False"/>
                    </enzymeml:format>
                </enzymeml:formats>
                <enzymeml:listOfMeasurements>
                    <enzymeml:measurement file="file0" id="m0" name="Test measurement">
                    </enzymeml:measurement>
                </enzymeml:listOfMeasurements>
                <enzymeml:files>
                    <enzymeml:file file="./data/m0.csv" format="format0" id="file0"/>
                </enzymeml:files>
            </enzymeml:data>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml_false).unwrap();
        assert!(annotation.data.is_some());
        let data = annotation.data.unwrap();
        assert_eq!(data.formats.format[0].columns[0].is_calculated, false);

        // Test with regular boolean
        let xml_bool = r#"
        <annotation>
            <enzymeml:data xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:formats>
                    <enzymeml:format id="format0">
                        <enzymeml:column type="time" unit="u4" index="0" isCalculated="true"/>
                    </enzymeml:format>
                </enzymeml:formats>
                <enzymeml:listOfMeasurements>
                    <enzymeml:measurement file="file0" id="m0" name="Test measurement">
                    </enzymeml:measurement>
                </enzymeml:listOfMeasurements>
                <enzymeml:files>
                    <enzymeml:file file="./data/m0.csv" format="format0" id="file0"/>
                </enzymeml:files>
            </enzymeml:data>
        </annotation>
        "#;

        let annotation = parse_v1_annotation(xml_bool).unwrap();
        assert!(annotation.data.is_some());
        let data = annotation.data.unwrap();
        assert_eq!(data.formats.format[0].columns[0].is_calculated, true);
    }
}
