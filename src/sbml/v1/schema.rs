//! SBML v1 annotation structures for EnzymeML
//!
//! This module provides comprehensive Rust data structures that mirror the Python pydantic-xml
//! structures used for parsing and serializing SBML v1 format EnzymeML annotations. These
//! structures enable seamless interoperability between Rust and Python-based EnzymeML workflows
//! while maintaining type safety and efficient serialization/deserialization.
//!
//! # Overview
//!
//! The EnzymeML v1 annotation format extends SBML with specialized annotations that capture
//! biochemical reaction data, experimental measurements, and molecular information. This module
//! implements the complete v1 schema including:
//!
//! - **Entity Annotations**: Structures for small molecules, proteins, and complexes
//! - **Parameter Annotations**: Support for bounded parameters with units
//! - **Data Annotations**: Comprehensive experimental data organization
//! - **Measurement Structure**: Time-series data with column mappings and initial conditions
//!
//! # Namespace Handling
//!
//! All structures support both prefixed and non-prefixed XML elements through serde aliases:
//! - Standard format: `<parameter xmlns="http://sbml.org/enzymeml/version1">`
//! - Prefixed format: `<enzymeml:parameter xmlns:enzymeml="http://sbml.org/enzymeml/version1">`
//!
//! # Serialization Features
//!
//! - Automatic XML namespace handling with default values
//! - Python boolean compatibility ("True"/"False" strings)
//! - Optional field serialization with skip_serializing_if
//! - Type conversions between EnzymeML and v1 column types
//! - Empty structure detection for optimization

use serde::{Deserialize, Deserializer, Serialize};
use variantly::Variantly;

use crate::{
    prelude::DataTypes,
    sbml::{utils::IsEmpty, v2::schema::VariableAnnot},
};

/// The XML namespace identifier for EnzymeML v1 annotations
pub(crate) const ENZYMEML_V1_NS: &str = "http://sbml.org/enzymeml/version1";

/// Provides the default XML namespace for v1 annotations
fn default_xmlns() -> String {
    ENZYMEML_V1_NS.to_string()
}

/// Top-level annotation container for EnzymeML v1 format
///
/// This structure serves as the root element for all EnzymeML v1 annotations within SBML
/// documents. It can contain any combination of entity-specific annotations depending on
/// the SBML element being annotated.
///
/// # Usage Patterns
///
/// Different SBML elements typically contain specific annotation types:
/// - **Species elements**: small_molecule, protein, or complex annotations
/// - **Parameter elements**: parameter annotations with bounds and units
/// - **Reaction elements**: data annotations with experimental measurements
/// - **Assignment rules**: variables annotations for mathematical relationships
///
/// # XML Structure
///
/// ```xml
/// <annotation>
///   <enzymeml:protein xmlns:enzymeml="http://sbml.org/enzymeml/version1">
///     <enzymeml:sequence>MKLLVL...</enzymeml:sequence>
///     <enzymeml:ECnumber>1.1.1.1</enzymeml:ECnumber>
///   </enzymeml:protein>
/// </annotation>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "annotation")]
pub struct V1Annotation {
    /// Chemical annotation for small molecule species
    ///
    /// Contains structural identifiers like InChI, SMILES, and database references
    /// for small molecules participating in biochemical reactions.
    #[serde(
        rename = "smallMolecule",
        alias = "enzymeml:reactant",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub small_molecule: Option<ReactantAnnot>,

    /// Biological annotation for protein species
    ///
    /// Includes amino acid sequences, enzyme classification numbers, and
    /// organism information for proteins catalyzing reactions.
    #[serde(
        rename = "protein",
        alias = "enzymeml:protein",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub protein: Option<ProteinAnnot>,

    /// Structural annotation for molecular complexes
    ///
    /// Defines multi-component assemblies formed by interactions between
    /// multiple species (proteins, small molecules, or other complexes).
    #[serde(
        rename = "complex",
        alias = "enzymeml:complex",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub complex: Option<ComplexAnnot>,

    /// Experimental data organization and file references
    ///
    /// Contains comprehensive measurement data structure including file locations,
    /// column definitions, and measurement metadata for time-series experiments.
    #[serde(
        rename = "data",
        alias = "enzymeml:data",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub data: Option<DataAnnot>,

    /// Parameter estimation and bounds information
    ///
    /// Defines initial values, upper/lower bounds, and units for parameters
    /// used in kinetic modeling and parameter estimation workflows.
    #[serde(
        rename = "parameter",
        alias = "enzymeml:parameter",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub parameter: Option<ParameterAnnot>,

    /// Mathematical variable definitions and relationships
    ///
    /// Specifies derived quantities and mathematical expressions that depend
    /// on other model variables, imported from v2 schema for compatibility.
    #[serde(
        rename = "variables",
        alias = "enzymeml:variables",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub variables: Option<VariableAnnot>,
}

/// Parameter annotation with bounds and estimation metadata
///
/// This structure extends basic SBML parameters with additional information required
/// for parameter estimation, optimization, and uncertainty quantification. It supports
/// the common workflow of defining parameter search spaces and initial guesses.
///
/// # Parameter Estimation Workflow
///
/// 1. **Initial Value**: Starting point for optimization algorithms
/// 2. **Bounds**: Constraints defining feasible parameter space
/// 3. **Units**: Dimensional analysis and unit conversion support
///
/// # XML Representation
///
/// ```xml
/// <parameter xmlns="http://sbml.org/enzymeml/version1" unit="per_second">
///   <initialValue>1.5</initialValue>
///   <upperBound>10.0</upperBound>
///   <lowerBound>0.1</lowerBound>
/// </parameter>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "parameter")]
pub struct ParameterAnnot {
    /// XML namespace declaration for validation and compatibility
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Starting value for parameter estimation algorithms
    ///
    /// Provides the initial guess for optimization routines. Should be within
    /// the bounds defined by upper and lower limits when specified.
    #[serde(
        rename = "initialValue",
        alias = "enzymeml:initialValue",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub initial: Option<f64>,

    /// Maximum allowed value during parameter estimation
    ///
    /// Defines the upper constraint for optimization algorithms to prevent
    /// parameters from reaching unrealistic or unstable values.
    #[serde(
        rename = "upperBound",
        alias = "enzymeml:upperBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub upper: Option<f64>,

    /// Minimum allowed value during parameter estimation
    ///
    /// Establishes the lower constraint for optimization, often used to
    /// enforce physical constraints like non-negative concentrations.
    #[serde(
        rename = "lowerBound",
        alias = "enzymeml:lowerBound",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub lower: Option<f64>,

    /// Unit identifier referencing SBML unit definitions
    ///
    /// Links to unit definitions in the SBML model for dimensional analysis
    /// and unit conversion during parameter estimation and simulation.
    #[serde(rename = "@unit", default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Optimization trait implementation for empty parameter detection
impl IsEmpty for ParameterAnnot {
    fn is_empty(&self) -> bool {
        self.initial.is_none() && self.upper.is_none() && self.lower.is_none()
    }
}

/// Molecular complex annotation with participant information
///
/// Represents multi-component molecular assemblies formed through non-covalent
/// interactions. Complexes can include any combination of proteins, small molecules,
/// and other complexes, enabling hierarchical assembly modeling.
///
/// # Complex Formation Examples
///
/// - **Enzyme-Substrate Complex**: Protein + small molecule substrate
/// - **Allosteric Complex**: Enzyme + effector molecule
/// - **Multi-subunit Enzyme**: Multiple protein chains
/// - **Ternary Complex**: Enzyme + substrate + cofactor
///
/// # XML Structure
///
/// ```xml
/// <complex xmlns="http://sbml.org/enzymeml/version1">
///   <participant>p0</participant>  <!-- Protein ID -->
///   <participant>s0</participant>  <!-- Small molecule ID -->
/// </complex>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "complex")]
pub struct ComplexAnnot {
    /// XML namespace for schema validation
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// List of species identifiers forming the complex
    ///
    /// Each participant corresponds to an SBML species ID. The order may be
    /// significant for certain complex formation mechanisms or stoichiometry.
    #[serde(rename = "participant", alias = "enzymeml:participant", default)]
    pub participants: Vec<String>,
}

/// Empty complex detection for serialization optimization
impl IsEmpty for ComplexAnnot {
    fn is_empty(&self) -> bool {
        self.participants.is_empty()
    }
}

/// Small molecule annotation with chemical identifiers
///
/// Provides comprehensive chemical identification for small molecules using
/// standard cheminformatics formats. These identifiers enable cross-referencing
/// with chemical databases and support automated analysis workflows.
///
/// # Identifier Types
///
/// - **InChI**: International Chemical Identifier for unique structural representation
/// - **SMILES**: Simplified Molecular Input Line Entry System for structure encoding  
/// - **ChEBI ID**: Chemical Entities of Biological Interest database reference
///
/// # Database Integration
///
/// These identifiers facilitate integration with:
/// - PubChem for chemical property lookup
/// - ChEBI for biological role annotation
/// - KEGG for pathway analysis
/// - MetaCyc for metabolic network mapping
///
/// # XML Example
///
/// ```xml
/// <smallMolecule xmlns="http://sbml.org/enzymeml/version1">
///   <inchi>InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2</inchi>
///   <smiles>C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O</smiles>
///   <chebiID>CHEBI:4167</chebiID>
/// </smallMolecule>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "reactant")]
pub struct ReactantAnnot {
    /// International Chemical Identifier for structural uniqueness
    ///
    /// Provides a standardized way to represent molecular structure that is
    /// independent of naming conventions and ensures global uniqueness.
    #[serde(
        rename = "inchi",
        alias = "enzymeml:inchi",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub inchi: Option<String>,

    /// Simplified Molecular Input Line Entry System notation
    ///
    /// Compact string representation of molecular structure that is widely
    /// supported by cheminformatics tools and chemical databases.
    #[serde(
        rename = "smiles",
        alias = "enzymeml:smiles",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub smiles: Option<String>,

    /// Chemical Entities of Biological Interest database identifier
    ///
    /// Links to the ChEBI database for additional chemical and biological
    /// information including cellular roles and biochemical pathways.
    #[serde(
        rename = "chebiID",
        alias = "enzymeml:chebiID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub chebi_id: Option<String>,
}

/// Empty annotation detection for storage optimization
impl IsEmpty for ReactantAnnot {
    fn is_empty(&self) -> bool {
        self.inchi.is_none() && self.smiles.is_none() && self.chebi_id.is_none()
    }
}

/// Protein annotation with biological and structural information
///
/// Captures essential protein characteristics including sequence information,
/// functional classification, and taxonomic origin. This comprehensive annotation
/// supports enzyme kinetics modeling and cross-species comparisons.
///
/// # Information Categories
///
/// - **Structural**: Amino acid sequence for structure-function analysis
/// - **Functional**: EC number for enzymatic activity classification
/// - **Database**: UniProt ID for comprehensive protein information
/// - **Taxonomic**: Organism and tax ID for evolutionary context
///
/// # Enzyme Classification
///
/// EC numbers follow the hierarchical system:
/// - Class (1-7): Type of reaction catalyzed
/// - Subclass: Chemical bonds or groups involved  
/// - Sub-subclass: Specific chemical groups
/// - Serial number: Individual enzyme identifier
///
/// # XML Representation
///
/// ```xml
/// <protein xmlns="http://sbml.org/enzymeml/version1">
///   <sequence>MKLLVLCLLATVAVAATSAA...</sequence>
///   <ECnumber>3.1.1.43</ECnumber>
///   <uniprotID>B0RS62</uniprotID>
///   <organism>Xanthomonas campestris pv. campestris</organism>
///   <organismTaxID>340</organismTaxID>
/// </protein>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "protein")]
pub struct ProteinAnnot {
    /// XML namespace for document validation
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Primary amino acid sequence in single-letter code
    ///
    /// Complete protein sequence enabling structure prediction, homology
    /// analysis, and structure-function relationship studies.
    #[serde(
        rename = "sequence",
        alias = "enzymeml:sequence",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub sequence: Option<String>,

    /// Enzyme Commission number for functional classification
    ///
    /// Four-part numerical classification system defining the type of
    /// biochemical reaction catalyzed by the enzyme.
    #[serde(
        rename = "ECnumber",
        alias = "enzymeml:ECnumber",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub ecnumber: Option<String>,

    /// Universal Protein Resource database identifier
    ///
    /// Links to comprehensive protein information including structure,
    /// function, cellular location, and post-translational modifications.
    #[serde(
        rename = "uniprotID",
        alias = "enzymeml:uniprotID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub uniprotid: Option<String>,

    /// Source organism scientific name
    ///
    /// Taxonomic identification of the species from which the protein
    /// was isolated or cloned, important for enzyme properties.
    #[serde(
        rename = "organism",
        alias = "enzymeml:organism",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism: Option<String>,

    /// NCBI Taxonomy database identifier for the source organism
    ///
    /// Numerical identifier linking to detailed taxonomic information
    /// and enabling cross-species enzyme comparison studies.
    #[serde(
        rename = "organismTaxID",
        alias = "enzymeml:organismTaxID",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub organism_tax_id: Option<String>,

    /// Additional unknown or custom protein annotations
    ///
    /// Captures any additional fields present in the annotation that are not
    /// explicitly handled by the known fields above. This allows for forward
    /// compatibility and preservation of custom annotations.
    #[serde(flatten, default)]
    pub additional_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// Empty protein annotation detection for optimization
impl IsEmpty for ProteinAnnot {
    fn is_empty(&self) -> bool {
        self.sequence.is_none()
            && self.ecnumber.is_none()
            && self.uniprotid.is_none()
            && self.organism.is_none()
            && self.organism_tax_id.is_none()
    }
}

/// Comprehensive experimental data annotation structure
///
/// Organizes all aspects of experimental measurement data including file references,
/// data structure definitions, and measurement metadata. This hierarchical organization
/// enables complex experimental setups with multiple measurements and data files.
///
/// # Workflow Integration
///
/// 1. **File Definition**: CSV files are registered with unique identifiers
/// 2. **Format Specification**: Column structures define data interpretation
/// 3. **Measurement Mapping**: Experiments are linked to files and formats
/// 4. **Initial Conditions**: Starting concentrations are specified per measurement
///
/// # XML Structure Example
///
/// ```xml
/// <data xmlns="http://sbml.org/enzymeml/version1">
///   <formats>
///     <format id="format0">
///       <column type="time" unit="second" index="0"/>
///       <column species="s0" type="conc" unit="mM" index="1"/>
///     </format>
///   </formats>
///   <listOfMeasurements>
///     <measurement id="m0" name="Time course" file="file0">
///       <initConc reactant="s0" value="20.0" unit="mM"/>
///     </measurement>
///   </listOfMeasurements>
///   <files>
///     <file id="file0" file="data/timecourse.csv" format="format0"/>
///   </files>
/// </data>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "data")]
pub struct DataAnnot {
    /// XML namespace declaration
    #[serde(rename = "@xmlns", default = "default_xmlns")]
    pub xmlns: String,

    /// Data structure definitions for file interpretation
    ///
    /// Contains format specifications that define how columns in data files
    /// should be interpreted, including data types, units, and species mapping.
    #[serde(rename = "formats", alias = "enzymeml:formats")]
    pub formats: FormatsWrapper,

    /// Individual measurement definitions and metadata
    ///
    /// Links experimental measurements to data files and provides initial
    /// condition information required for kinetic modeling.
    #[serde(rename = "listOfMeasurements", alias = "enzymeml:listOfMeasurements")]
    pub measurements: MeasurementsWrapper,

    /// Physical file locations and format associations
    ///
    /// Defines the mapping between logical file identifiers and actual
    /// file paths within the COMBINE archive or file system.
    #[serde(rename = "files", alias = "enzymeml:files")]
    pub files: FilesWrapper,
}

/// Empty data annotation detection for serialization efficiency
impl IsEmpty for DataAnnot {
    fn is_empty(&self) -> bool {
        self.formats.format.is_empty()
            && self.measurements.measurement.is_empty()
            && self.files.file.is_empty()
    }
}

/// XML wrapper for format definitions to handle list serialization
///
/// Provides proper XML structure for multiple format definitions while
/// maintaining compatibility with both single and multiple format scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FormatsWrapper {
    /// Collection of data format specifications
    #[serde(rename = "format", alias = "enzymeml:format", default)]
    pub format: Vec<FormatAnnot>,
}

/// XML wrapper for measurement definitions to handle list serialization
///
/// Encapsulates multiple measurement annotations while preserving the
/// required XML structure for proper deserialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MeasurementsWrapper {
    /// Collection of individual measurement specifications
    #[serde(rename = "measurement", alias = "enzymeml:measurement", default)]
    pub measurement: Vec<MeasurementAnnot>,
}

/// XML wrapper for file definitions to handle list serialization
///
/// Maintains proper XML structure for file reference collections while
/// supporting both single and multiple file scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FilesWrapper {
    /// Collection of file location and format specifications
    #[serde(rename = "file", alias = "enzymeml:file", default)]
    pub file: Vec<FileAnnot>,
}

/// Data format specification defining column structure and interpretation
///
/// Defines how tabular data files should be parsed and interpreted, specifying
/// the meaning, units, and data types for each column. This enables automatic
/// data processing and validation during file loading.
///
/// # Column Organization
///
/// Formats typically include:
/// - **Time column**: Independent variable for time-series data
/// - **Species columns**: Dependent variables for concentration measurements
/// - **Replica columns**: Multiple measurements of the same species
/// - **Calculated columns**: Derived quantities from other measurements
///
/// # Data Validation
///
/// Format definitions enable:
/// - Unit consistency checking across measurements
/// - Data type validation during file parsing
/// - Species identifier verification against model
/// - Column index validation for data integrity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "format")]
pub struct FormatAnnot {
    /// Unique identifier for this format specification
    ///
    /// Used to link measurements and files to their corresponding
    /// column structure definitions.
    #[serde(rename = "@id")]
    pub id: String,

    /// Ordered list of column specifications defining data structure
    ///
    /// Each column definition specifies how to interpret a specific
    /// column in associated data files.
    #[serde(rename = "column", alias = "enzymeml:column", default)]
    pub columns: Vec<ColumnAnnot>,
}

/// Individual column specification within a data format
///
/// Defines the complete interpretation context for a single column in tabular
/// data files. This includes the data type, units, species association, and
/// metadata flags that control how the data should be processed.
///
/// # Column Types and Usage
///
/// - **Time**: Independent variable for kinetic experiments
/// - **Concentration**: Species amount measurements over time
/// - **Absorption**: Spectroscopic measurements for indirect quantification
/// - **Biomass**: Cell density or total protein measurements
/// - **Feed**: Substrate addition profiles for fed-batch experiments
/// - **Peak Area**: Chromatographic quantification data
/// - **Conversion**: Reaction progress measurements
///
/// # Replica Handling
///
/// Multiple measurements of the same species can be distinguished using
/// replica identifiers, enabling statistical analysis and error estimation.
///
/// # Calculated vs Measured Data
///
/// The `is_calculated` flag distinguishes between:
/// - **Measured data**: Direct experimental observations
/// - **Calculated data**: Derived quantities from mathematical models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "column")]
pub struct ColumnAnnot {
    /// SBML species identifier for concentration-related columns
    ///
    /// Links column data to specific chemical species in the model.
    /// Not applicable for time columns or other non-species data.
    #[serde(rename = "@species", default, skip_serializing_if = "Option::is_none")]
    pub species_id: Option<String>,

    /// Data type classification for interpretation and processing
    ///
    /// Determines how column values should be interpreted and what
    /// mathematical operations are appropriate.
    #[serde(rename = "@type")]
    pub column_type: ColumnType,

    /// Unit identifier referencing SBML unit definitions
    ///
    /// Enables dimensional analysis and unit conversion across
    /// different measurements and model parameters.
    #[serde(rename = "@unit")]
    pub unit: String,

    /// Zero-based column position in the data file
    ///
    /// Specifies the physical location of this data within
    /// CSV or TSV file structures.
    #[serde(rename = "@index")]
    pub index: usize,

    /// Identifier for distinguishing multiple measurements of the same species
    ///
    /// Enables statistical analysis and uncertainty quantification
    /// when multiple replicas are available.
    #[serde(rename = "@replica", default, skip_serializing_if = "Option::is_none")]
    pub replica: Option<String>,

    /// Flag indicating whether data is measured or computationally derived
    ///
    /// Supports both Python-style boolean strings ("True"/"False") and
    /// standard boolean values for cross-platform compatibility.
    #[serde(
        rename = "@isCalculated",
        default,
        deserialize_with = "deserialize_python_bool"
    )]
    pub is_calculated: bool,
}

/// Enumeration of supported column data types for experimental measurements
///
/// Defines the semantic meaning of different types of experimental data,
/// enabling appropriate mathematical treatment and visualization. Each type
/// corresponds to specific measurement techniques and data characteristics.
///
/// # Measurement Techniques
///
/// - **Concentration**: Direct chemical quantification (HPLC, GC-MS, etc.)
/// - **Absorption**: UV-Vis spectroscopy for indirect measurement
/// - **Biomass**: Cell density via optical density or dry weight
/// - **Feed**: Substrate addition profiles in fed-batch reactors
/// - **Peak Area**: Chromatographic integration for quantification
/// - **Conversion**: Reaction progress based on substrate consumption
/// - **Time**: Independent variable for temporal experiments
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Variantly)]
pub enum ColumnType {
    /// Direct concentration measurements in specified units
    #[default]
    #[serde(rename = "conc")]
    Concentration,

    /// Spectroscopic absorption measurements for indirect quantification
    #[serde(rename = "abs")]
    Absorption,

    /// Substrate feeding profiles for fed-batch experimental setups
    #[serde(rename = "feed")]
    Feed,

    /// Biomass measurements via optical density or gravimetric methods
    #[serde(rename = "biomass")]
    Biomass,

    /// Reaction conversion based on substrate depletion or product formation
    #[serde(rename = "conversion")]
    Conversion,

    /// Chromatographic peak area for quantitative analysis
    #[serde(rename = "peak-area")]
    PeakArea,

    /// Time points for kinetic measurements and time-series analysis
    #[serde(rename = "time")]
    Time,
}

/// Conversion from v1 column types to EnzymeML data types
impl From<ColumnType> for Option<DataTypes> {
    fn from(column_type: ColumnType) -> Self {
        (&column_type).into()
    }
}

/// Reference-based conversion for efficiency
impl From<&ColumnType> for Option<DataTypes> {
    fn from(column_type: &ColumnType) -> Self {
        match column_type {
            ColumnType::Concentration => Some(DataTypes::Concentration),
            ColumnType::Absorption => Some(DataTypes::Absorbance),
            ColumnType::Biomass => Some(DataTypes::Amount),
            ColumnType::Conversion => Some(DataTypes::Conversion),
            ColumnType::PeakArea => Some(DataTypes::PeakArea),
            ColumnType::Feed => Some(DataTypes::Concentration),
            _ => None,
        }
    }
}

/// Conversion from EnzymeML data types to v1 column types
impl From<&DataTypes> for ColumnType {
    fn from(data_type: &DataTypes) -> Self {
        match data_type {
            DataTypes::Concentration => ColumnType::Concentration,
            DataTypes::Absorbance => ColumnType::Absorption,
            DataTypes::Amount => ColumnType::Biomass,
            DataTypes::Conversion => ColumnType::Conversion,
            DataTypes::PeakArea => ColumnType::PeakArea,
            _ => ColumnType::Concentration,
        }
    }
}

/// Individual measurement annotation with metadata and initial conditions
///
/// Represents a single experimental measurement with associated data file,
/// format specification, and initial condition information. This structure
/// links experimental metadata to actual time-series data.
///
/// # Measurement Organization
///
/// Each measurement includes:
/// - **Identification**: Unique ID and descriptive name
/// - **Data Link**: Reference to file containing time-series data
/// - **Initial State**: Starting concentrations for all species
/// - **Format Link**: Column structure for data interpretation
///
/// # Initial Condition Specification
///
/// Initial concentrations are crucial for:
/// - Kinetic parameter estimation algorithms
/// - Model simulation starting points  
/// - Mass balance validation
/// - Experimental reproducibility assessment
///
/// # XML Example
///
/// ```xml
/// <measurement id="m0" name="Substrate titration" file="file0">
///   <initConc protein="p0" value="0.1" unit="mg_per_ml"/>
///   <initConc reactant="s0" value="10.0" unit="mM"/>
///   <initConc reactant="s1" value="0.0" unit="mM"/>
/// </measurement>
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "measurement")]
pub struct MeasurementAnnot {
    /// Unique identifier for cross-referencing within the document
    #[serde(rename = "@id")]
    pub id: String,

    /// Human-readable descriptive name for the measurement
    #[serde(rename = "@name")]
    pub name: String,

    /// File identifier linking to data file location
    ///
    /// References a file definition that specifies the actual
    /// location and format of the time-series data.
    #[serde(rename = "@file")]
    pub file: String,

    /// List of initial concentration specifications for all species
    ///
    /// Defines the starting state of the system for kinetic modeling
    /// and parameter estimation workflows.
    #[serde(rename = "initConc", alias = "enzymeml:initConc", default)]
    pub init_concs: Vec<InitConcAnnot>,
}

/// Initial concentration specification for individual species
///
/// Defines the starting concentration of a specific chemical species at the
/// beginning of an experimental measurement. This information is essential
/// for kinetic modeling and parameter estimation.
///
/// # Species Types
///
/// Initial concentrations can be specified for:
/// - **Proteins**: Enzyme concentrations in catalytic reactions
/// - **Reactants**: Substrate and product starting concentrations
/// - **Complexes**: Pre-formed molecular assembly concentrations
///
/// # Unit Consistency
///
/// Units must be consistent with:
/// - SBML model unit definitions
/// - Column units in associated data files
/// - Parameter units in kinetic expressions
///
/// # Modeling Applications
///
/// Initial concentrations are used for:
/// - Ordinary differential equation initial conditions
/// - Parameter estimation algorithm constraints
/// - Mass balance validation checks
/// - Experimental design optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "initConc")]
pub struct InitConcAnnot {
    /// Protein species identifier for enzyme initial concentrations
    ///
    /// References a protein species defined in the SBML model.
    /// Mutually exclusive with reactant field.
    #[serde(rename = "@protein", default, skip_serializing_if = "Option::is_none")]
    pub protein: Option<String>,

    /// Small molecule species identifier for substrate/product concentrations
    ///
    /// References a small molecule species in the SBML model.
    /// Mutually exclusive with protein field.
    #[serde(rename = "@reactant", default, skip_serializing_if = "Option::is_none")]
    pub reactant: Option<String>,

    /// Numerical concentration value in specified units
    ///
    /// Must be non-negative and physically reasonable for the
    /// experimental system being modeled.
    #[serde(rename = "@value", default)]
    pub value: f64,

    /// Unit identifier for dimensional consistency
    ///
    /// References unit definitions in the SBML model to ensure
    /// proper dimensional analysis and unit conversion.
    #[serde(rename = "@unit")]
    pub unit: Option<String>,
}

/// File location and format association for data access
///
/// Links logical file identifiers to physical file locations within COMBINE
/// archives or file systems. Associates data files with their corresponding
/// format specifications for proper interpretation.
///
/// # File Management
///
/// - **Logical ID**: Internal reference for measurement linking
/// - **Physical Location**: Actual file path within archive or filesystem
/// - **Format Association**: Links to column structure definitions
///
/// # COMBINE Archive Integration
///
/// File paths typically follow patterns:
/// - `data/measurement_id.csv` for time-series data
/// - `metadata/protocols.xml` for experimental protocols
/// - `models/kinetic_model.xml` for SBML model files
///
/// # Cross-Platform Compatibility
///
/// File paths should use forward slashes for cross-platform compatibility
/// within COMBINE archives and relative paths for portability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename = "file")]
pub struct FileAnnot {
    /// Unique identifier for internal referencing
    ///
    /// Used by measurement annotations to link to specific data files
    /// without exposing physical file paths in metadata.
    #[serde(rename = "@id")]
    pub id: String,

    /// Physical file location within archive or filesystem
    ///
    /// Specifies the actual path to the data file, typically relative
    /// to the COMBINE archive root or document location.
    #[serde(rename = "@file")]
    pub location: String,

    /// Format identifier linking to column structure definition
    ///
    /// References a format specification that defines how to interpret
    /// the columns and data types within the file.
    #[serde(rename = "@format")]
    pub format: String,
}

/// Custom deserializer for Python-style boolean strings with cross-platform support
///
/// Handles the deserialization of boolean values that may originate from Python
/// systems using string representations "True" and "False" (case-insensitive).
/// Also supports standard JSON boolean values for maximum compatibility.
///
/// # Supported Formats
///
/// - Standard booleans: `true`, `false`
/// - Python strings: `"True"`, `"False"` (case-insensitive)
/// - Mixed case variations: `"TRUE"`, `"False"`, etc.
///
/// # Error Handling
///
/// Returns descriptive errors for invalid boolean representations to aid
/// in debugging malformed data files or incompatible formats.
fn deserialize_python_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct BoolVisitor;

    impl Visitor<'_> for BoolVisitor {
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
                _ => Err(E::custom(format!("Invalid boolean string: {value}"))),
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
        assert!(data.formats.format[0].columns[0].is_calculated);

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
        assert!(!data.formats.format[0].columns[0].is_calculated);

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
        assert!(data.formats.format[0].columns[0].is_calculated);
    }
}
