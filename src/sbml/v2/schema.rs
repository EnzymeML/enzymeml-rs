//! SBML v2 Annotation Schema for EnzymeML
//!
//! This module defines the complete annotation schema for EnzymeML version 2 format,
//! providing structured representations of enzymatic data within SBML models. The schema
//! enables comprehensive annotation of biochemical entities, experimental measurements,
//! and kinetic parameters following the EnzymeML v2 specification.
//!
//! ## Overview
//!
//! The annotation schema supports the following core components:
//! - **Molecular Entities**: Small molecules, proteins, and molecular complexes with
//!   chemical identifiers, sequences, and structural information
//! - **Experimental Data**: Time-series measurements with conditions, units, and
//!   species-specific data points for kinetic analysis
//! - **Model Parameters**: Kinetic parameters with statistical bounds, uncertainties,
//!   and optimization constraints for model fitting
//! - **Variables**: Mathematical variables used in kinetic equations and model definitions
//!
//! ## Schema Structure
//!
//! All annotations are structured hierarchically under the `V2Annotation` root element,
//! which serves as the container for specific annotation types. Each annotation type
//! corresponds to a particular aspect of enzymatic data and maintains consistency with
//! the EnzymeML JSON schema while providing SBML-compatible XML serialization.
//!
//! ## Usage
//!
//! These annotation structures are primarily used during SBML import/export operations
//! to preserve EnzymeML-specific metadata that extends beyond standard SBML capabilities.
//! The annotations enable round-trip conversion between EnzymeML and SBML formats while
//! maintaining data integrity and experimental context.

use serde::{Deserialize, Serialize};

use crate::{prelude::DataTypes, sbml::utils::IsEmpty};

pub const ENZYMEML_V2_NS: &str = "https://www.enzymeml.org/v2";

fn default_xmlns() -> String {
    ENZYMEML_V2_NS.to_string()
}

/// Root annotation container for EnzymeML version 2 format
///
/// This structure serves as the top-level container for all EnzymeML v2 annotations
/// within SBML models. It provides a unified interface for accessing different types
/// of experimental and modeling data while maintaining compatibility with SBML
/// annotation standards.
///
/// The container supports selective annotation of SBML elements, allowing different
/// model components to carry only relevant EnzymeML metadata. This approach optimizes
/// storage efficiency while preserving complete experimental context where needed.
///
/// ## Supported Annotation Types
///
/// - **Small Molecules**: Chemical identifiers and structural representations
/// - **Proteins**: Enzymatic information, sequences, and taxonomic data
/// - **Complexes**: Multi-component molecular assemblies and interactions
/// - **Experimental Data**: Time-series measurements with conditions and metadata
/// - **Parameters**: Kinetic parameters with statistical properties and constraints
/// - **Variables**: Mathematical variables for kinetic equations and model definitions
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "annotation")]
pub struct V2Annotation {
    /// Chemical annotation for small molecule entities
    ///
    /// Contains structural identifiers including InChI, InChIKey, and SMILES
    /// representations for unambiguous chemical identification and database linking
    #[serde(
        rename = "smallMolecule",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub small_molecule: Option<SmallMoleculeAnnot>,

    /// Biological annotation for protein entities
    ///
    /// Includes enzymatic classification, taxonomic information, and amino acid
    /// sequences for comprehensive protein characterization and database integration
    #[serde(rename = "protein", default, skip_serializing_if = "Option::is_none")]
    pub protein: Option<ProteinAnnot>,

    /// Structural annotation for molecular complex entities
    ///
    /// Defines multi-component assemblies and their constituent participants
    /// for modeling protein-protein and protein-substrate interactions
    #[serde(rename = "complex", default, skip_serializing_if = "Option::is_none")]
    pub complex: Option<ComplexAnnot>,

    /// Experimental data annotation linking measurements to data files
    ///
    /// Associates time-series measurements with external data sources while
    /// maintaining experimental conditions and measurement metadata
    #[serde(rename = "data", default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataAnnot>,

    /// Kinetic parameter annotation with statistical properties
    ///
    /// Provides parameter bounds, uncertainties, and optimization constraints
    /// for robust kinetic model fitting and parameter estimation
    #[serde(rename = "parameter", default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<ParameterAnnot>,

    /// Mathematical variable annotation for kinetic equations
    ///
    /// Defines variables used in rate equations and model expressions
    /// with proper symbol mapping and identification
    #[serde(rename = "variables", default, skip_serializing_if = "Option::is_none")]
    pub variables: Option<VariablesAnnot>,
}

/// Chemical annotation for small molecule entities in enzymatic reactions
///
/// This structure provides comprehensive chemical identification for small molecules
/// participating in enzymatic reactions. It supports multiple chemical identifier
/// formats to ensure compatibility with various chemical databases and enable
/// unambiguous molecular identification across different systems.
///
/// The annotation includes both structural (SMILES, InChI) and hash-based (InChIKey)
/// identifiers, providing flexibility for different use cases while maintaining
/// chemical accuracy and database interoperability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "smallMolecule")]
pub struct SmallMoleculeAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// International Chemical Identifier Key for database linking
    ///
    /// Provides a fixed-length hash representation of the molecular structure
    /// that enables efficient database searches and cross-referencing with
    /// chemical databases like PubChem, ChEBI, and others
    #[serde(rename = "inchiKey", default, skip_serializing_if = "Option::is_none")]
    pub inchikey: Option<String>,

    /// International Chemical Identifier for structural representation
    ///
    /// Contains the complete structural description of the molecule in
    /// standardized InChI format, enabling precise chemical specification
    /// and structural comparison across different chemical databases
    pub inchi: Option<String>,

    /// Canonical SMILES notation for chemical structure
    ///
    /// Provides a standardized linear notation for representing molecular
    /// structure that is both human-readable and machine-processable,
    /// supporting structural analysis and chemical informatics applications
    #[serde(rename = "smiles", default, skip_serializing_if = "Option::is_none")]
    pub canonical_smiles: Option<String>,

    /// Synonymous names for the small molecule
    ///
    /// Provides alternative names for the small molecule that are used to
    /// identify the molecule in different contexts, such as in different
    /// databases or in different languages
    #[serde(rename = "synonyms", default, skip_serializing_if = "Option::is_none")]
    pub synonyms: Option<Vec<String>>,
}

impl IsEmpty for SmallMoleculeAnnot {
    fn is_empty(&self) -> bool {
        self.inchikey.is_none() && self.canonical_smiles.is_none()
    }
}

/// Biological annotation for protein entities with enzymatic properties
///
/// This structure captures essential biological information about proteins,
/// particularly focusing on enzymatic properties and taxonomic classification.
/// It supports integration with biological databases and enables comprehensive
/// protein characterization for enzymatic studies.
///
/// The annotation includes enzymatic classification through EC numbers,
/// taxonomic information for organism identification, and sequence data
/// for molecular-level protein analysis and database cross-referencing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "protein")]
pub struct ProteinAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Enzyme Commission number for enzymatic classification
    ///
    /// Provides standardized enzymatic classification according to the
    /// International Union of Biochemistry and Molecular Biology (IUBMB)
    /// nomenclature system, enabling functional annotation and database linking
    #[serde(rename = "ecnumber", default, skip_serializing_if = "Option::is_none")]
    pub ecnumber: Option<String>,

    /// Source organism for taxonomic identification
    ///
    /// Specifies the biological origin of the protein using standardized
    /// organism names, supporting taxonomic classification and enabling
    /// organism-specific analysis and database integration
    #[serde(rename = "organism", default, skip_serializing_if = "Option::is_none")]
    pub organism: Option<String>,

    /// NCBI Taxonomy identifier for precise organism classification
    ///
    /// Provides unambiguous taxonomic identification through the NCBI
    /// Taxonomy database identifier system, enabling precise organism
    /// specification and cross-database compatibility
    #[serde(
        rename = "organismTaxId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism_tax_id: Option<String>,

    /// Amino acid sequence in single-letter notation
    ///
    /// Contains the complete protein sequence using standard single-letter
    /// amino acid codes, enabling sequence-based analysis, homology searches,
    /// and molecular-level protein characterization
    #[serde(rename = "sequence", default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<String>,
}

impl IsEmpty for ProteinAnnot {
    fn is_empty(&self) -> bool {
        self.ecnumber.is_none()
            && self.organism.is_none()
            && self.organism_tax_id.is_none()
            && self.sequence.is_none()
    }
}

/// Structural annotation for molecular complexes and assemblies
///
/// This structure represents multi-component molecular assemblies formed by
/// interactions between proteins, substrates, cofactors, or other molecular
/// entities. It enables modeling of complex biochemical interactions and
/// multi-molecular catalytic mechanisms.
///
/// The complex annotation maintains references to all participating entities,
/// allowing for comprehensive representation of molecular interactions while
/// preserving individual component identities and enabling complex-specific
/// analysis and modeling approaches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "complex")]
pub struct ComplexAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// List of molecular entity identifiers participating in the complex
    ///
    /// Contains references to all molecular components that form the complex,
    /// including proteins, substrates, cofactors, and other entities.
    /// Each participant is identified by its corresponding SBML species identifier
    #[serde(rename = "participants", default)]
    pub participants: Vec<String>,
}

impl IsEmpty for ComplexAnnot {
    fn is_empty(&self) -> bool {
        self.participants.is_empty()
    }
}

/// Functional annotation for reaction modifiers and regulatory elements
///
/// This structure describes entities that modify or regulate enzymatic reactions
/// without being consumed or produced. It captures the specific role of modifiers
/// such as inhibitors, activators, cofactors, or allosteric regulators in
/// enzymatic processes.
///
/// The modifier annotation enables precise specification of regulatory mechanisms
/// and supports comprehensive modeling of complex enzymatic regulation patterns
/// while maintaining compatibility with SBML modifier specifications.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "modifier")]
pub struct ModifierAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Functional role of the modifier in the enzymatic reaction
    ///
    /// Specifies the type of modification or regulation provided by this entity,
    /// such as "inhibitor", "activator", "cofactor", or "allosteric_regulator".
    /// This enables precise modeling of regulatory mechanisms and their effects
    #[serde(rename = "@modifierRole")]
    pub modifier_role: String,
}

/// Experimental data annotation linking measurements to external data files
///
/// This structure establishes the connection between SBML model elements and
/// external experimental data files, enabling comprehensive representation of
/// time-series measurements and experimental conditions. It serves as the bridge
/// between theoretical models and empirical data.
///
/// The data annotation supports multiple measurements within a single data file,
/// enabling efficient organization of experimental datasets while maintaining
/// clear associations between measurements and their corresponding model elements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "data")]
pub struct DataAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Path or identifier of the external data file
    ///
    /// Specifies the location or identifier of the file containing experimental
    /// time-series data. This enables separation of model structure from
    /// experimental data while maintaining clear data provenance and accessibility
    #[serde(rename = "@file")]
    pub file: String,

    /// Collection of measurement definitions within the data file
    ///
    /// Contains detailed specifications for each measurement included in the
    /// associated data file, including experimental conditions, time units,
    /// and species-specific measurement parameters for comprehensive data description
    #[serde(rename = "measurement", default)]
    pub measurements: Vec<MeasurementAnnot>,
}

impl IsEmpty for DataAnnot {
    fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }
}

/// Experimental measurement annotation with conditions and metadata
///
/// This structure provides comprehensive description of individual experimental
/// measurements, including experimental conditions, temporal information, and
/// species-specific data specifications. It enables detailed characterization
/// of experimental setups and measurement protocols.
///
/// The measurement annotation supports flexible experimental design representation
/// while maintaining standardized metadata structure for consistent data interpretation
/// and analysis across different experimental contexts and measurement types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "measurement")]
pub struct MeasurementAnnot {
    /// Unique identifier for the measurement within the dataset
    ///
    /// Provides unambiguous identification of individual measurements,
    /// enabling data association and reference resolution within
    /// experimental datasets and model validation workflows
    #[serde(rename = "@id")]
    pub id: String,

    /// Human-readable name for the measurement
    ///
    /// Offers descriptive identification of the measurement for
    /// documentation and user interface purposes, complementing
    /// the unique identifier with meaningful context
    #[serde(rename = "@name", default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Temporal unit specification for time-series data
    ///
    /// Defines the unit of measurement for time values in the
    /// associated time-series data, ensuring proper temporal
    /// scaling and unit consistency across measurements
    #[serde(rename = "@timeUnit", default, skip_serializing_if = "Option::is_none")]
    pub time_unit: Option<String>,

    /// Experimental conditions during measurement acquisition
    ///
    /// Specifies environmental and experimental parameters such as
    /// pH, temperature, and other conditions that may affect
    /// enzymatic activity and measurement interpretation
    #[serde(
        rename = "conditions",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub conditions: Option<ConditionsAnnot>,

    /// Species-specific measurement data specifications
    ///
    /// Contains detailed information about measured species including
    /// initial concentrations, data types, units, and measurement
    /// parameters for comprehensive species characterization
    #[serde(rename = "speciesData", default)]
    pub species_data: Vec<SpeciesDataAnnot>,
}

/// Species-specific measurement data annotation with quantitative parameters
///
/// This structure describes experimental data associated with individual molecular
/// species, including initial conditions, measurement types, and unit specifications.
/// It enables precise characterization of species behavior during experimental
/// measurements and supports quantitative analysis workflows.
///
/// The species data annotation maintains direct association with SBML species
/// while providing EnzymeML-specific measurement context and quantitative parameters
/// necessary for kinetic analysis and model validation procedures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "speciesData")]
pub struct SpeciesDataAnnot {
    /// Reference to the corresponding SBML species identifier
    ///
    /// Establishes direct association with the molecular species
    /// defined in the SBML model, ensuring consistent entity
    /// identification across model structure and measurement data
    #[serde(rename = "@species")]
    pub species_id: String,

    /// Initial concentration or activity value for the species
    ///
    /// Specifies the starting value for the species at the
    /// beginning of the measurement period, providing essential
    /// initial condition information for kinetic analysis
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<f64>,

    /// Type of measurement data for the species
    ///
    /// Defines whether the measurement represents concentration,
    /// activity, or other quantitative properties, enabling
    /// appropriate data interpretation and analysis methods
    #[serde(rename = "@type", default = "default_data_type")]
    pub data_type: DataTypes,

    /// Unit specification for the measurement values
    ///
    /// Provides the measurement unit for proper quantitative
    /// interpretation and unit consistency validation across
    /// the experimental dataset and modeling framework
    #[serde(rename = "@unit")]
    pub unit: String,
}

fn default_data_type() -> DataTypes {
    DataTypes::Concentration
}

/// Experimental conditions annotation for measurement context
///
/// This structure captures environmental and experimental parameters
/// that influence enzymatic activity and measurement outcomes. It provides
/// essential context for proper data interpretation and enables condition-specific
/// analysis and modeling approaches.
///
/// The conditions annotation supports extensible experimental parameter
/// specification while maintaining standardized representation for common
/// experimental variables such as pH and temperature that significantly
/// affect enzymatic behavior and measurement reliability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "conditions")]
pub struct ConditionsAnnot {
    /// pH conditions during experimental measurement
    ///
    /// Specifies the hydrogen ion concentration conditions
    /// that significantly affect enzymatic activity and
    /// measurement interpretation in biochemical systems
    #[serde(rename = "ph", default, skip_serializing_if = "Option::is_none")]
    pub ph: Option<PHAnnot>,

    /// Temperature conditions during experimental measurement
    ///
    /// Defines thermal conditions that influence enzymatic
    /// kinetics and measurement accuracy, supporting
    /// temperature-dependent analysis and modeling
    #[serde(
        rename = "temperature",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub temperature: Option<TemperatureAnnot>,
}

/// pH value annotation for experimental conditions
///
/// This structure represents hydrogen ion concentration conditions during
/// experimental measurements. pH significantly affects enzymatic activity,
/// protein stability, and substrate binding affinity, making it a critical
/// parameter for enzymatic studies and model validation.
///
/// The pH annotation provides standardized representation of acidity conditions
/// while supporting integration with experimental protocols and enabling
/// pH-dependent analysis of enzymatic behavior and measurement data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "ph")]
pub struct PHAnnot {
    /// Numerical pH value on the standard pH scale
    ///
    /// Represents the negative logarithm of hydrogen ion
    /// concentration, typically ranging from 0 to 14,
    /// providing standardized acidity measurement for
    /// experimental condition specification
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
}

impl IsEmpty for PHAnnot {
    fn is_empty(&self) -> bool {
        self.value.is_none()
    }
}

/// Temperature annotation for experimental thermal conditions
///
/// This structure represents thermal conditions during experimental measurements,
/// including both temperature values and their associated units. Temperature
/// significantly affects enzymatic kinetics, protein stability, and reaction
/// rates, making it essential for accurate experimental characterization.
///
/// The temperature annotation supports various unit systems while providing
/// standardized representation for thermal conditions, enabling temperature-dependent
/// modeling and cross-study comparison of enzymatic behavior under different
/// thermal conditions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "temperature")]
pub struct TemperatureAnnot {
    /// Numerical temperature value in specified units
    ///
    /// Represents the thermal condition during experimental
    /// measurement, providing essential information for
    /// temperature-dependent enzymatic analysis and modeling
    #[serde(rename = "@value", default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,

    /// Unit specification for temperature measurement
    ///
    /// Defines the unit system used for temperature values,
    /// such as Celsius, Kelvin, or Fahrenheit, ensuring
    /// proper unit interpretation and conversion capabilities
    #[serde(rename = "@unit", default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl IsEmpty for TemperatureAnnot {
    fn is_empty(&self) -> bool {
        self.value.is_none() && self.unit.is_none()
    }
}

/// Kinetic parameter annotation with statistical properties and constraints
///
/// This structure provides comprehensive specification of kinetic parameters
/// used in enzymatic models, including statistical properties, optimization
/// bounds, and uncertainty information. It enables robust parameter estimation
/// and supports advanced modeling workflows.
///
/// The parameter annotation facilitates integration with parameter estimation
/// algorithms while maintaining statistical rigor and providing essential
/// information for model validation, uncertainty analysis, and optimization
/// procedures in enzymatic modeling contexts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "parameter")]
pub struct ParameterAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Initial value for parameter estimation procedures
    ///
    /// Provides starting point for optimization algorithms
    /// and parameter estimation workflows, influencing
    /// convergence behavior and estimation accuracy
    #[serde(
        rename = "initialValue",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub initial: Option<f64>,

    /// Lower boundary constraint for parameter optimization
    ///
    /// Establishes minimum allowable value for the parameter
    /// during estimation procedures, ensuring biologically
    /// meaningful parameter values and numerical stability
    #[serde(
        rename = "lowerBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub lower_bound: Option<f64>,

    /// Upper boundary constraint for parameter optimization
    ///
    /// Defines maximum allowable value for the parameter
    /// during estimation procedures, preventing unrealistic
    /// parameter values and maintaining model validity
    #[serde(
        rename = "upperBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub upper_bound: Option<f64>,

    /// Standard error or uncertainty estimate for the parameter
    ///
    /// Provides statistical uncertainty information from
    /// parameter estimation procedures, enabling confidence
    /// interval calculation and uncertainty propagation analysis
    #[serde(
        rename = "stdDeviation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub stderr: Option<f64>,

    /// Unit specification for parameter values
    ///
    /// Defines the measurement unit for parameter values,
    /// ensuring dimensional consistency and enabling proper
    /// unit conversion and validation across model components
    #[serde(rename = "@unit", default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl IsEmpty for ParameterAnnot {
    fn is_empty(&self) -> bool {
        self.lower_bound.is_none() && self.upper_bound.is_none() && self.stderr.is_none()
    }
}

/// Variable collection annotation for kinetic equation definitions
///
/// This structure serves as a container for mathematical variables used in
/// kinetic equations and model expressions. It enables comprehensive variable
/// management and supports complex mathematical modeling approaches in
/// enzymatic systems.
///
/// The variables annotation facilitates organization of mathematical expressions
/// while maintaining clear variable identification and enabling systematic
/// variable management across different model components and equation systems.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "variables")]
pub struct VariablesAnnot {
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Collection of mathematical variable definitions
    ///
    /// Contains individual variable specifications used in
    /// kinetic equations and mathematical expressions,
    /// supporting complex modeling workflows and equation systems
    #[serde(rename = "variable", default)]
    pub variables: Vec<VariableAnnot>,
}

impl IsEmpty for VariablesAnnot {
    fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }
}

/// Mathematical variable annotation for kinetic equations and expressions
///
/// This structure defines individual variables used in kinetic equations,
/// rate expressions, and mathematical models. It provides comprehensive
/// variable specification including identification, naming, and symbolic
/// representation for mathematical modeling workflows.
///
/// The variable annotation enables precise mathematical expression definition
/// while supporting variable resolution and symbolic manipulation in complex
/// kinetic modeling systems and equation-based analysis approaches.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename = "variable")]
pub struct VariableAnnot {
    /// Unique identifier for the variable within the model
    ///
    /// Provides unambiguous variable identification for
    /// reference resolution and mathematical expression
    /// parsing in kinetic modeling contexts
    #[serde(rename = "@id", default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Human-readable name for the variable
    ///
    /// Offers descriptive identification for documentation
    /// and user interface purposes, complementing the
    /// unique identifier with meaningful context
    #[serde(rename = "@name", default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Mathematical symbol representation for the variable
    ///
    /// Provides symbolic notation used in mathematical
    /// expressions and equations, enabling proper symbol
    /// resolution and expression rendering
    #[serde(rename = "@symbol", default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
}

impl IsEmpty for VariableAnnot {
    fn is_empty(&self) -> bool {
        self.id.is_none() && self.name.is_none() && self.symbol.is_none()
    }
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
