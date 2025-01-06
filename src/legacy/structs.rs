use crate::{unwrap_enum, unwrap_list};
use serde::{Deserialize, Serialize};

fn default_level() -> u8 {
    3
}

fn default_version() -> u8 {
    2
}

#[allow(non_snake_case, clippy::upper_case_acronyms)]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SBML {
    #[serde(rename = "@level", default = "default_level")]
    level: u8,
    #[serde(rename = "@version", default = "default_version")]
    version: u8,
    model: Model,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Model {
    #[serde(rename = "@id")]
    id: String,

    #[serde(rename = "@name")]
    name: String,

    #[serde(
        rename = "listOfUnitDefinitions",
        deserialize_with = "unwrap_list_of_unit_definitions",
        default
    )]
    unit_definitions: Vec<UnitDefinition>,

    #[serde(
        rename = "listOfCompartments",
        deserialize_with = "unwrap_list_of_compartments",
        default
    )]
    compartments: Vec<Compartment>,

    #[serde(
        rename = "listOfSpecies",
        deserialize_with = "unwrap_list_of_species",
        default
    )]
    species: Vec<Species>,

    #[serde(rename = "listOfReactions")]
    reactions: Option<ListOfReactions>,
}

// Model: Container parsers
unwrap_list!(
    listOfUnitDefinitions,
    unitDefinition,
    UnitDefinition,
    unwrap_list_of_unit_definitions
);

unwrap_list!(
    listOfCompartments,
    compartment,
    Compartment,
    unwrap_list_of_compartments
);

// Species: Container parsers
unwrap_list!(listOfSpecies, species, Species, unwrap_list_of_species);

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct UnitDefinition {
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "listOfUnits", deserialize_with = "unwrap_list_of_units")]
    units: Vec<Unit>,
}

// UnitDefintion: Container parsers
unwrap_list!(listOfUnits, unit, Unit, unwrap_list_of_units);

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Unit {
    #[serde(rename = "@kind")]
    kind: String,
    #[serde(rename = "@exponent")]
    exponent: i32,
    #[serde(rename = "@scale")]
    scale: i32,
    #[serde(rename = "@multiplier")]
    multiplier: i32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Compartment {
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@spatialDimensions")]
    spatial_dimensions: u8,
    #[serde(rename = "@size")]
    size: f64,
    #[serde(rename = "@units")]
    units: String,
    #[serde(rename = "@constant")]
    constant: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Species {
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@compartment")]
    compartment: String,
    #[serde(rename = "@initialConcentration")]
    initial_concentration: f64,
    #[serde(rename = "@substanceUnits")]
    substance_units: String,
    #[serde(rename = "@hasOnlySubstanceUnits")]
    has_only_substance_units: bool,
    #[serde(rename = "@boundaryCondition")]
    boundary_condition: bool,
    #[serde(rename = "@constant")]
    constant: bool,
    #[serde(
        rename = "annotation",
        deserialize_with = "unwrap_enum_species_annotation"
    )]
    annotation: SpeciesAnnotation,
}

// Species: Annotation parsers
unwrap_enum!(
    SpeciesAnnotationHolder,
    annotation,
    SpeciesAnnotation,
    unwrap_enum_species_annotation
);

#[derive(Debug, Serialize, Deserialize, PartialEq)]
enum SpeciesAnnotation {
    #[serde(rename = "protein")]
    Protein(EnzymeMLProtein),
    #[serde(rename = "reactant")]
    Reactant(EnzymeMLReactant),
    #[serde(rename = "complex")]
    Complex(EnzymeMLComplex),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLProtein {
    #[serde(rename = "sequence")]
    sequence: String,
    #[serde(rename = "ECnumber")]
    ec_number: String,
    #[serde(rename = "uniprotID")]
    uniprot_id: String,
    #[serde(rename = "organism")]
    organism: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLComplex {
    #[serde(rename = "participant")]
    participants: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLReactant {
    #[serde(rename = "inchi")]
    inchi: String,
    #[serde(rename = "smiles")]
    smiles: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Parameter {
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: Option<String>,
    #[serde(rename = "@value")]
    value: f64,
    #[serde(rename = "@units")]
    units: String,
    #[serde(rename = "@constant")]
    constant: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLData {
    #[serde(
        rename = "formats",
        deserialize_with = "unwrap_list_of_formats",
        default
    )]
    formats: Vec<EnzymeMLFormat>,
    #[serde(
        rename = "listOfMeasurements",
        deserialize_with = "unwrap_list_of_measurements",
        default
    )]
    measurements: Vec<EnzymeMLMeasurement>,
    #[serde(rename = "files", deserialize_with = "unwrap_list_of_files", default)]
    files: Vec<EnzymeMLFile>,
}

// EnzymeMLData: Format parsers
unwrap_list!(files, file, EnzymeMLFile, unwrap_list_of_files);
unwrap_list!(formats, format, EnzymeMLFormat, unwrap_list_of_formats);
unwrap_list!(
    listOfMeasurements,
    measurement,
    EnzymeMLMeasurement,
    unwrap_list_of_measurements
);

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLFormat {
    #[serde(rename = "@id")]
    format_id: String,
    #[serde(rename = "column")]
    columns: Vec<EnzymeMLColumn>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLColumn {
    #[serde(rename = "@replica")]
    replica: Option<String>,
    #[serde(rename = "@species")]
    species: Option<String>,
    #[serde(rename = "@type")]
    column_type: String,
    #[serde(rename = "@unit")]
    unit: String,
    #[serde(rename = "@index")]
    index: u32,
    #[serde(rename = "@isCalculated")]
    is_calculated: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLMeasurement {
    #[serde(rename = "@file")]
    file: String,
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "initConc")]
    init_concentrations: Vec<EnzymeMLInitConc>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ListOfReactions {
    #[serde(rename = "@reaction")]
    reactions: Vec<EnzymeMLReaction>,
    #[serde(rename = "@annotation")]
    annotation: Option<EnzymeMLData>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLInitConc {
    #[serde(rename = "@protein")]
    protein: String,
    #[serde(rename = "@value")]
    value: f64,
    #[serde(rename = "@unit")]
    unit: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLFile {
    #[serde(rename = "@file")]
    file: String,
    #[serde(rename = "@format")]
    format: String,
    #[serde(rename = "@id")]
    id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct EnzymeMLReaction {
    #[serde(rename = "@metaid")]
    metaid: String,
    #[serde(rename = "@sboTerm")]
    sbo_term: String,
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@reversible")]
    reversible: bool,
    #[serde(
        rename = "listOfReactants",
        deserialize_with = "unwrap_list_of_reactants"
    )]
    reactants: Vec<SpeciesReference>,
    #[serde(
        rename = "listOfProducts",
        deserialize_with = "unwrap_list_of_products"
    )]
    products: Vec<SpeciesReference>,
}

// EnzymeMLReaction: Container parsers
unwrap_list!(
    listOfReactants,
    speciesReference,
    SpeciesReference,
    unwrap_list_of_reactants
);
unwrap_list!(
    listOfProducts,
    speciesReference,
    SpeciesReference,
    unwrap_list_of_products
);

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SpeciesReference {
    #[serde(rename = "@sboTerm")]
    sbo_term: String,
    #[serde(rename = "@species")]
    species: String,
    #[serde(rename = "@stoichiometry")]
    stoichiometry: f64,
    #[serde(rename = "@constant")]
    constant: bool,
    annotation: Option<EnzymeMLData>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use quick_xml::de::from_str;

    #[test]
    fn test_empty_sbml() {
        let xml_input = r#"
        <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
            <model id="EnzymeML_Lagerman" name="EnzymeML_Lagerman"/>
        </sbml>
        "#;

        let actual: SBML = from_str(xml_input).unwrap();
        assert_eq!(
            SBML {
                level: 3,
                version: 2,
                model: Model {
                    id: "EnzymeML_Lagerman".to_string(),
                    name: "EnzymeML_Lagerman".to_string(),
                    unit_definitions: vec![],
                    compartments: vec![],
                    species: vec![],
                    reactions: None,
                }
            },
            actual
        );
    }

    #[test]
    fn test_deserialize_unit_definition() {
        let xml_input = r#"
        <unitDefinition metaid="METAID_U0" id="u0" name="ml">
            <listOfUnits>
                <unit kind="litre" exponent="1" scale="-3" multiplier="1"/>
            </listOfUnits>
        </unitDefinition>
        "#;

        let expected = UnitDefinition {
            id: "u0".to_string(),
            name: "ml".to_string(),
            units: vec![Unit {
                kind: "litre".to_string(),
                exponent: 1,
                scale: -3,
                multiplier: 1,
            }],
        };

        let actual: UnitDefinition = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_compartment() {
        let xml_input = r#"
        <compartment id="v0" name="Falcon Tube" spatialDimensions="3" size="5" units="u0" constant="true"/>
        "#;

        let expected = Compartment {
            id: "v0".to_string(),
            name: "Falcon Tube".to_string(),
            spatial_dimensions: 3,
            size: 5.0,
            units: "u0".to_string(),
            constant: true,
        };

        let actual: Compartment = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_species_protein() {
        let xml_input = r#"
        <species metaid="METAID_P0" sboTerm="SBO:0000252" id="p0" name="EA" compartment="v0" initialConcentration="0" substanceUnits="u1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
            <annotation>
                <enzymeml:protein xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                    <enzymeml:sequence>MRRL</enzymeml:sequence>
                    <enzymeml:ECnumber>3.1.1.43</enzymeml:ECnumber>
                    <enzymeml:uniprotID>B0RS62</enzymeml:uniprotID>
                    <enzymeml:organism>Xanthomonas campestris pv. campestris</enzymeml:organism>
                </enzymeml:protein>
            </annotation>
        </species>
        "#;

        let expected = Species {
            id: "p0".to_string(),
            name: "EA".to_string(),
            compartment: "v0".to_string(),
            initial_concentration: 0.0,
            substance_units: "u1".to_string(),
            has_only_substance_units: false,
            boundary_condition: false,
            constant: false,
            annotation: SpeciesAnnotation::Protein(EnzymeMLProtein {
                sequence: "MRRL".to_string(),
                ec_number: "3.1.1.43".to_string(),
                uniprot_id: "B0RS62".to_string(),
                organism: "Xanthomonas campestris pv. campestris".to_string(),
            }),
        };

        let actual: Species = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_species_reactant() {
        let xml_input = r#"
        <species metaid="METAID_P1" sboTerm="SBO:0000252" id="p1" name="EA" compartment="v0" initialConcentration="0" substanceUnits="u1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
            <annotation>
                <enzymeml:reactant xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                    <enzymeml:inchi>1S</enzymeml:inchi>
                    <enzymeml:smiles>C(C</enzymeml:smiles>
                </enzymeml:reactant>
            </annotation>
        </species>
        "#;

        let expected = Species {
            id: "p1".to_string(),
            name: "EA".to_string(),
            compartment: "v0".to_string(),
            initial_concentration: 0.0,
            substance_units: "u1".to_string(),
            has_only_substance_units: false,
            boundary_condition: false,
            constant: false,
            annotation: SpeciesAnnotation::Reactant(EnzymeMLReactant {
                inchi: "1S".to_string(),
                smiles: "C(C".to_string(),
            }),
        };

        let actual: Species = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_species_complex() {
        let xml_input = r#"
        <species metaid="METAID_P2" sboTerm="SBO:0000252" id="p2" name="ED" compartment="v0" initialConcentration="0" substanceUnits="u1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
            <annotation>
                <enzymeml:complex xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                    <enzymeml:participant>p0</enzymeml:participant>
                    <enzymeml:participant>p1</enzymeml:participant>
                </enzymeml:complex>
            </annotation>
        </species>
        "#;

        let expected = Species {
            id: "p2".to_string(),
            name: "ED".to_string(),
            compartment: "v0".to_string(),
            initial_concentration: 0.0,
            substance_units: "u1".to_string(),
            has_only_substance_units: false,
            boundary_condition: false,
            constant: false,
            annotation: SpeciesAnnotation::Complex(EnzymeMLComplex {
                participants: vec!["p0".to_string(), "p1".to_string()],
            }),
        };

        let actual: Species = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_parameter() {
        let xml_input = r#"
        <parameter id="v_r" name="v_r" value="0" units="u5" constant="true"/>
        "#;

        let expected = Parameter {
            id: "v_r".to_string(),
            name: Some("v_r".to_string()),
            value: 0.0,
            units: "u5".to_string(),
            constant: true,
        };

        let actual: Parameter = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_enzymeml_data() {
        let xml_input = r#"
        <enzymeml:data xmlns:enzymeml="http://sbml.org/enzymeml/version2">
          <enzymeml:formats>
            <enzymeml:format id="format0">
              <enzymeml:column type="time" unit="u4" index="0"/>
            </enzymeml:format>
            <enzymeml:format id="format1">
              <enzymeml:column replica="sub1_repl2" species="s0" type="conc" unit="u1" index="1" isCalculated="false"/>
            </enzymeml:format>
          </enzymeml:formats>
          <enzymeml:listOfMeasurements>
            <enzymeml:measurement file="file0" id="m0" name="Cephalexin synthesis 1">
              <enzymeml:initConc protein="p0" value="0.0002" unit="u1"/>
            </enzymeml:measurement>
          </enzymeml:listOfMeasurements>
          <enzymeml:files>
            <enzymeml:file file="./data/m0.csv" format="format0" id="file0"/>
          </enzymeml:files>
        </enzymeml:data>
        "#;

        let actual: EnzymeMLData = from_str(xml_input).unwrap();

        let expected = EnzymeMLData {
            formats: vec![
                EnzymeMLFormat {
                    format_id: "format0".to_string(),
                    columns: vec![EnzymeMLColumn {
                        replica: None,
                        species: None,
                        column_type: "time".to_string(),
                        unit: "u4".to_string(),
                        index: 0,
                        is_calculated: None,
                    }],
                },
                EnzymeMLFormat {
                    format_id: "format1".to_string(),
                    columns: vec![EnzymeMLColumn {
                        replica: Some("sub1_repl2".to_string()),
                        species: Some("s0".to_string()),
                        column_type: "conc".to_string(),
                        unit: "u1".to_string(),
                        index: 1,
                        is_calculated: Some(false),
                    }],
                },
            ],
            measurements: vec![EnzymeMLMeasurement {
                file: "file0".to_string(),
                id: "m0".to_string(),
                name: "Cephalexin synthesis 1".to_string(),
                init_concentrations: vec![EnzymeMLInitConc {
                    protein: "p0".to_string(),
                    value: 0.0002,
                    unit: "u1".to_string(),
                }],
            }],
            files: vec![EnzymeMLFile {
                file: "./data/m0.csv".to_string(),
                format: "format0".to_string(),
                id: "file0".to_string(),
            }],
        };

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_deserialize_enzymeml_reaction() {
        let xml_input = r#"
            <reaction metaid="METAID_R3" sboTerm="SBO:0000176" id="r3" name="reaction-4" reversible="true">
                <listOfReactants>
                    <speciesReference sboTerm="SBO:0000015" species="c3" stoichiometry="1" constant="false"/>
                </listOfReactants>
                <listOfProducts>
                    <speciesReference sboTerm="SBO:0000011" species="s0" stoichiometry="1" constant="false"/>
                </listOfProducts>
            </reaction>
        "#;

        let expected = EnzymeMLReaction {
            metaid: "METAID_R3".to_string(),
            sbo_term: "SBO:0000176".to_string(),
            id: "r3".to_string(),
            name: "reaction-4".to_string(),
            reversible: true,
            reactants: vec![SpeciesReference {
                sbo_term: "SBO:0000015".to_string(),
                species: "c3".to_string(),
                stoichiometry: 1.0,
                constant: false,
                annotation: None,
            }],
            products: vec![SpeciesReference {
                sbo_term: "SBO:0000011".to_string(),
                species: "s0".to_string(),
                stoichiometry: 1.0,
                constant: false,
                annotation: None,
            }],
        };

        let actual: EnzymeMLReaction = from_str(xml_input).unwrap();
        assert_eq!(expected, actual);
    }
}
