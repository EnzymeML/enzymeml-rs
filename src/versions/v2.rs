//! This file contains Rust struct definitions with serde serialization.
//!
//! WARNING: This is an auto-generated file.
//! Do not edit directly - any changes will be overwritten.

use derive_builder::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

//
// Type definitions
//
/// The EnzymeMLDocument is the root object that serves as a container
/// for all components of an enzymatic experiment. It includes
/// essential metadata about the document itself, such as its
/// title and creation/modification dates, as well as references to
/// related publications and databases. Additionally, it contains
/// comprehensive information about the experimental setup, including
/// reaction vessels, proteins, complexes, small molecules, reactions,
/// measurements, equations, and parameters.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct EnzymeMLDocument {
    /// Title of the EnzymeML Document.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// The version of the EnzymeML Document.
    #[serde(default)]
    #[builder(default = "2.0.to_string().into()", setter(into))]
    pub version: String,

    /// Description of the EnzymeML Document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub description: Option<String>,

    /// Date the EnzymeML Document was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub created: Option<String>,

    /// Date the EnzymeML Document was modified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub modified: Option<String>,

    /// Contains descriptions of all authors that are part of the experiment.
    ///
    #[builder(default, setter(into, each(name = "to_creators")))]
    pub creators: Vec<Creator>,

    /// Contains descriptions of all vessels that are part of the experiment.
    ///
    #[builder(default, setter(into, each(name = "to_vessels")))]
    pub vessels: Vec<Vessel>,

    /// Contains descriptions of all proteins that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_proteins")))]
    pub proteins: Vec<Protein>,

    /// Contains descriptions of all complexes that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_complexes")))]
    pub complexes: Vec<Complex>,

    /// Contains descriptions of all reactants that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_small_molecules")))]
    pub small_molecules: Vec<SmallMolecule>,

    /// Contains descriptions of all reactions that are part of the
    /// experiment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_reactions")))]
    pub reactions: Vec<Reaction>,

    /// Contains descriptions of all measurements that are part of the
    /// experiment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_measurements")))]
    pub measurements: Vec<Measurement>,

    /// Contains descriptions of all equations that are part of the
    /// experiment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_equations")))]
    pub equations: Vec<Equation>,

    /// Contains descriptions of all parameters that are part of the
    /// experiment and may be used in equations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_parameters")))]
    pub parameters: Vec<Parameter>,

    /// Contains references to publications, databases, and arbitrary links to
    /// the web.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    pub references: Vec<String>,
}

/// The Creator object represents an individual author or contributor who
/// has participated in creating or modifying the EnzymeML Document.
/// It captures essential personal information such as their name
/// and contact details, allowing proper attribution and enabling
/// communication with the document's creators.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Creator {
    /// Given name of the author or contributor.
    ///
    #[builder(setter(into))]
    pub given_name: String,

    /// Family name of the author or contributor.
    ///
    #[builder(setter(into))]
    pub family_name: String,

    /// Email address of the author or contributor.
    ///
    #[builder(setter(into))]
    pub mail: String,
}

/// The Vessel object represents containers used to conduct experiments,
/// such as reaction vessels, microplates, or bioreactors. It captures
/// key properties like volume and whether the volume remains constant
/// during the experiment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Vessel {
    /// Unique identifier of the vessel.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the used vessel.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Volumetric value of the vessel.
    ///
    #[builder(setter(into))]
    pub volume: f64,

    /// Volumetric unit of the vessel.
    ///
    #[builder(setter(into))]
    pub unit: UnitDefinition,

    /// Whether the volume of the vessel is constant or not. Default is True.
    #[serde(default)]
    #[builder(default = "true.into()", setter(into))]
    pub constant: bool,
}

/// The Protein object represents enzymes and other proteins involved in
/// the experiment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Protein {
    /// Identifier of the protein, such as a UniProt ID, or a custom
    /// identifier.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the protein.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Whether the concentration of the protein is constant through the
    /// experiment or not. Default is True.
    #[serde(default)]
    #[builder(default = "true.into()", setter(into))]
    pub constant: bool,

    /// Amino acid sequence of the protein
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub sequence: Option<String>,

    /// Identifier of the vessel this protein has been applied to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub vessel_id: Option<String>,

    /// EC number of the protein.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub ecnumber: Option<String>,

    /// Expression host organism of the protein.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub organism: Option<String>,

    /// Taxonomy identifier of the expression host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub organism_tax_id: Option<String>,

    /// List of references to publications, database entries, etc. that
    /// describe or reference the protein.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    pub references: Vec<String>,
}

/// The Complex object allows the grouping of multiple species using
/// their . This enables the representation of protein-small molecule
/// complexes (e.g., enzyme-substrate complexes) as well as buffer or
/// solvent mixtures (combinations of SmallMolecule species).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Complex {
    /// Unique identifier of the complex.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the complex.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Whether the concentration of the complex is constant through the
    /// experiment or not. Default is False.
    ///
    #[builder(setter(into))]
    pub constant: bool,

    /// Unique identifier of the vessel this complex has been used in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub vessel_id: Option<String>,

    /// Array of IDs the complex contains
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_participants")))]
    pub participants: Vec<String>,
}

/// The SmallMolecule object represents small chemical compounds that
/// participate in the experiment as substrates, products, or
/// modifiers. It captures key molecular identifiers like SMILES and
/// InChI.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct SmallMolecule {
    /// Identifier of the small molecule, such as a Pubchem ID, ChEBI ID, or a
    /// custom identifier.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the small molecule.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Whether the concentration of the small molecule is constant through
    /// the experiment or not. Default is False.
    ///
    #[builder(setter(into))]
    pub constant: bool,

    /// Identifier of the vessel this small molecule has been used in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub vessel_id: Option<String>,

    /// Canonical Simplified Molecular-Input Line-Entry System (SMILES)
    /// encoding of the small molecule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub canonical_smiles: Option<String>,

    /// International Chemical Identifier (InChI) encoding of the small
    /// molecule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub inchi: Option<String>,

    /// Hashed International Chemical Identifier (InChIKey) encoding of the
    /// small molecule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub inchikey: Option<String>,

    /// List of synonymous names for the small molecule.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_synonymous_names")))]
    pub synonymous_names: Vec<String>,

    /// List of references to publications, database entries, etc. that
    /// describe or reference the small molecule.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    pub references: Vec<String>,
}

/// The Reaction object represents a chemical or enzymatic reaction and
/// holds the different species and modifiers that are part of the
/// reaction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Reaction {
    /// Unique identifier of the reaction.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the reaction.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Whether the reaction is reversible or irreversible. Default is False.
    ///
    #[builder(setter(into))]
    pub reversible: bool,

    /// Mathematical expression of the reaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub kinetic_law: Option<Equation>,

    /// List of reactants that are part of the reaction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_reactants")))]
    pub reactants: Vec<ReactionElement>,

    /// List of products that are part of the reaction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_products")))]
    pub products: Vec<ReactionElement>,

    /// List of reaction elements that are not part of the reaction but
    /// influence it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_modifiers")))]
    pub modifiers: Vec<ModifierElement>,
}

/// This object is part of the object and describes a species
/// (SmallMolecule, Protein, Complex) participating in the reaction.
/// The stochiometry is of the species is specified in the field,
/// whereas negative values indicate that the species is a reactant
/// and positive values indicate that the species is a product of
/// the reaction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct ReactionElement {
    /// Internal identifier to either a protein or reactant defined in the
    /// EnzymeML Document.
    ///
    #[builder(setter(into))]
    pub species_id: String,

    /// Float number representing the associated stoichiometry.
    #[serde(default)]
    #[builder(default = "1.0.into()", setter(into))]
    pub stoichiometry: f64,
}

/// The ModifierElement object represents a species that is not part of
/// the reaction but influences it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct ModifierElement {
    /// Internal identifier to either a protein or reactant defined in the
    /// EnzymeML Document.
    ///
    #[builder(setter(into))]
    pub species_id: String,

    /// Role of the modifier in the reaction.
    ///
    #[builder(setter(into))]
    pub role: ModifierRole,
}

/// The Equation object describes a mathematical equation used to model
/// parts of a reaction system.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Equation {
    /// Identifier of a defined species (SmallMolecule, Protein, Complex).
    /// Represents the left hand side of the equation.
    ///
    #[builder(setter(into))]
    pub species_id: String,

    /// Mathematical expression of the equation. Represents the right hand
    /// side of the equation.
    ///
    #[builder(setter(into))]
    pub equation: String,

    /// Type of the equation.
    ///
    #[builder(setter(into))]
    pub equation_type: EquationType,

    /// List of variables that are part of the equation
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_variables")))]
    pub variables: Vec<Variable>,
}

/// This object describes a variable that is part of an equation.
/// Variables can represent species concentrations, time, or other
/// quantities that appear in mathematical expressions. Each variable
/// must have a unique identifier, name, and symbol that is used in
/// equations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Variable {
    /// Identifier of the variable.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the variable.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Equation symbol of the variable.
    ///
    #[builder(setter(into))]
    pub symbol: String,
}

/// This object describes parameters used in kinetic models, including
/// estimated values, bounds, and associated uncertainties.
/// Parameters can represent rate constants, binding constants, or
/// other numerical values that appear in rate equations or other
/// mathematical expressions.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Parameter {
    /// Identifier of the parameter.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the parameter.
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Equation symbol of the parameter.
    ///
    #[builder(setter(into))]
    pub symbol: String,

    /// Numerical value of the estimated parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub value: Option<f64>,

    /// Unit of the estimated parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub unit: Option<UnitDefinition>,

    /// Initial value that was used for the parameter estimation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub initial_value: Option<f64>,

    /// Upper bound for the parameter value that was used for the parameter
    /// estimation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub upper_bound: Option<f64>,

    /// Lower bound for the parameter value that was used for the parameter
    /// estimation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub lower_bound: Option<f64>,

    /// Standard error of the estimated parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub stderr: Option<f64>,

    /// Specifies if this parameter is constant. Default is True.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default = "true.into()", setter(into))]
    pub constant: Option<bool>,
}

/// This object describes a single measurement, which includes time
/// course data of any type defined in DataTypes. It contains initial
/// concentrations and measurement data for all species involved in
/// the experiment. Multiple measurements can be grouped together
/// using the group_id field to indicate they are part of the same
/// experimental series.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct Measurement {
    /// Unique identifier of the measurement.
    ///
    #[builder(setter(into))]
    pub id: String,

    /// Name of the measurement
    ///
    #[builder(setter(into))]
    pub name: String,

    /// Measurement data of all species that were part of the measurement. A
    /// species refers to a Protein, Complex, or SmallMolecule.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_species_data")))]
    pub species_data: Vec<MeasurementData>,

    /// User-defined group ID to signal relationships between measurements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub group_id: Option<String>,

    /// pH value of the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub ph: Option<f64>,

    /// Temperature of the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub temperature: Option<f64>,

    /// Unit of the temperature of the measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub temperature_unit: Option<UnitDefinition>,
}

/// This object describes a single entity of a measurement, which
/// corresponds to one species (Protein, Complex, SmallMolecule). It
/// contains time course data for that species, including the initial
/// amount, prepared amount, and measured data points over time.
/// Endpoint data is treated as a time course data point with only one
/// data point.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct MeasurementData {
    /// The identifier for the described reactant.
    ///
    #[builder(setter(into))]
    pub species_id: String,

    /// Amount of the the species before starting the measurement. This field
    /// can be used for specifying the prepared amount of a species
    /// in the reaction mix. Not to be confused with , specifying
    /// the concentration of a species at the first data point from
    /// the array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub prepared: Option<f64>,

    /// Initial amount of the measurement data. This must be the same as the
    /// first data point in the array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub initial: Option<f64>,

    /// SI unit of the data that was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub data_unit: Option<UnitDefinition>,

    /// Data that was measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_data")))]
    pub data: Vec<f64>,

    /// Corresponding time points of the .
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_time")))]
    pub time: Vec<f64>,

    /// Unit of the time points of the .
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub time_unit: Option<UnitDefinition>,

    /// Type of data that was measured (e.g. concentration, absorbance, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub data_type: Option<DataTypes>,

    /// Whether or not the data has been generated by simulation. Default
    /// is False.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub is_simulated: Option<bool>,
}

/// Represents a unit definition that is based on the SI unit system.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct UnitDefinition {
    /// Unique identifier of the unit definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub id: Option<String>,

    /// Common name of the unit definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub name: Option<String>,

    /// Base units that define the unit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_base_units")))]
    pub base_units: Vec<BaseUnit>,
}

/// Represents a base unit in the unit definition.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Default)]
#[allow(non_snake_case)]
pub struct BaseUnit {
    /// Kind of the base unit (e.g., meter, kilogram, second).
    ///
    #[builder(setter(into))]
    pub kind: UnitType,

    /// Exponent of the base unit in the unit definition.
    ///
    #[builder(setter(into))]
    pub exponent: i64,

    /// Multiplier of the base unit in the unit definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub multiplier: Option<f64>,

    /// Scale of the base unit in the unit definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub scale: Option<f64>,
}

//
// Enum definitions
//
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum ModifierRole {
    #[default]
    #[serde(rename = "activator")]
    ACTIVATOR,

    #[serde(rename = "additive")]
    ADDITIVE,

    #[serde(rename = "biocatalyst")]
    BIOCATALYST,

    #[serde(rename = "buffer")]
    BUFFER,

    #[serde(rename = "catalyst")]
    CATALYST,

    #[serde(rename = "inhibitor")]
    INHIBITOR,

    #[serde(rename = "solvent")]
    SOLVENT,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum EquationType {
    #[default]
    #[serde(rename = "assignment")]
    ASSIGNMENT,

    #[serde(rename = "initialAssignment")]
    INITIAL_ASSIGNMENT,

    #[serde(rename = "ode")]
    ODE,

    #[serde(rename = "rateLaw")]
    RATE_LAW,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum DataTypes {
    #[default]
    #[serde(rename = "absorbance")]
    ABSORBANCE,

    #[serde(rename = "amount")]
    AMOUNT,

    #[serde(rename = "concentration")]
    CONCENTRATION,

    #[serde(rename = "conversion")]
    CONVERSION,

    #[serde(rename = "fluorescence")]
    FLUORESCENCE,

    #[serde(rename = "peakarea")]
    PEAK_AREA,

    #[serde(rename = "transmittance")]
    TRANSMITTANCE,

    #[serde(rename = "turnover")]
    TURNOVER,

    #[serde(rename = "yield")]
    YIELD,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum UnitType {
    #[default]
    #[serde(rename = "ampere")]
    AMPERE,

    #[serde(rename = "avogadro")]
    AVOGADRO,

    #[serde(rename = "becquerel")]
    BECQUEREL,

    #[serde(rename = "candela")]
    CANDELA,

    #[serde(rename = "celsius")]
    CELSIUS,

    #[serde(rename = "coulomb")]
    COULOMB,

    #[serde(rename = "dimensionless")]
    DIMENSIONLESS,

    #[serde(rename = "farad")]
    FARAD,

    #[serde(rename = "gram")]
    GRAM,

    #[serde(rename = "gray")]
    GRAY,

    #[serde(rename = "henry")]
    HENRY,

    #[serde(rename = "hertz")]
    HERTZ,

    #[serde(rename = "item")]
    ITEM,

    #[serde(rename = "joule")]
    JOULE,

    #[serde(rename = "katal")]
    KATAL,

    #[serde(rename = "kelvin")]
    KELVIN,

    #[serde(rename = "kilogram")]
    KILOGRAM,

    #[serde(rename = "litre")]
    LITRE,

    #[serde(rename = "lumen")]
    LUMEN,

    #[serde(rename = "lux")]
    LUX,

    #[serde(rename = "metre")]
    METRE,

    #[serde(rename = "mole")]
    MOLE,

    #[serde(rename = "newton")]
    NEWTON,

    #[serde(rename = "ohm")]
    OHM,

    #[serde(rename = "pascal")]
    PASCAL,

    #[serde(rename = "radian")]
    RADIAN,

    #[serde(rename = "second")]
    SECOND,

    #[serde(rename = "siemens")]
    SIEMENS,

    #[serde(rename = "sievert")]
    SIEVERT,

    #[serde(rename = "steradian")]
    STERADIAN,

    #[serde(rename = "tesla")]
    TESLA,

    #[serde(rename = "volt")]
    VOLT,

    #[serde(rename = "watt")]
    WATT,

    #[serde(rename = "weber")]
    WEBER,
}
