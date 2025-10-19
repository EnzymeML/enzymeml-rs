//! This file contains Rust struct definitions with serde serialization.
//!
//! WARNING: This is an auto-generated file.
//! Do not edit directly - any changes will be overwritten.

use derivative::Derivative;
use derive_builder::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid;

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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct EnzymeMLDocument {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_enzymemldocument_jsonld_header()")]
    #[derivative(Default(value = "default_enzymemldocument_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// The version of the EnzymeML Document.

    #[builder(default = "2.0.to_string().into()", setter(into))]
    #[derivative(Default(value = "\"2.0\".to_string()"))]
    pub version: String,

    /// Description of the EnzymeML Document.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub description: Option<String>,

    /// Title of the EnzymeML Document.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Date the EnzymeML Document was created.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub created: Option<String>,

    /// Date the EnzymeML Document was modified.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub modified: Option<String>,

    /// Contains descriptions of all authors that are part of the experiment.

    #[builder(default, setter(into, each(name = "to_creators")))]
    #[derivative(Default)]
    pub creators: Vec<Creator>,

    /// Contains descriptions of all vessels that are part of the experiment.

    #[builder(default, setter(into, each(name = "to_vessels")))]
    #[derivative(Default)]
    pub vessels: Vec<Vessel>,

    /// Contains descriptions of all proteins that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_proteins")))]
    #[derivative(Default)]
    pub proteins: Vec<Protein>,

    /// Contains descriptions of all complexes that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_complexes")))]
    #[derivative(Default)]
    pub complexes: Vec<Complex>,

    /// Contains descriptions of all reactants that are part of the experiment
    /// that may be referenced in reactions, measurements, and
    /// equations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_small_molecules")))]
    #[derivative(Default)]
    pub small_molecules: Vec<SmallMolecule>,

    /// Contains descriptions of all reactions that are part of the
    /// experiment.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_reactions")))]
    #[derivative(Default)]
    pub reactions: Vec<Reaction>,

    /// Contains descriptions of all measurements that are part of the
    /// experiment.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_measurements")))]
    #[derivative(Default)]
    pub measurements: Vec<Measurement>,

    /// Contains descriptions of all equations that are part of the
    /// experiment.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_equations")))]
    #[derivative(Default)]
    pub equations: Vec<Equation>,

    /// Contains descriptions of all parameters that are part of the
    /// experiment and may be used in equations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_parameters")))]
    #[derivative(Default)]
    pub parameters: Vec<Parameter>,

    /// Contains references to publications, databases, and arbitrary links to
    /// the web.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    #[derivative(Default)]
    pub references: Vec<String>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Creator object represents an individual author or contributor who
/// has participated in creating or modifying the EnzymeML Document.
/// It captures essential personal information such as their name
/// and contact details, allowing proper attribution and enabling
/// communication with the document's creators.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Creator {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_creator_jsonld_header()")]
    #[derivative(Default(value = "default_creator_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Given name of the author or contributor.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub given_name: String,

    /// Family name of the author or contributor.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub family_name: String,

    /// Email address of the author or contributor.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub mail: String,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Vessel object represents containers used to conduct experiments,
/// such as reaction vessels, microplates, or bioreactors. It captures
/// key properties like volume and whether the volume remains constant
/// during the experiment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Vessel {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_vessel_jsonld_header()")]
    #[derivative(Default(value = "default_vessel_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Unique identifier of the vessel.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the used vessel.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Volumetric value of the vessel.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub volume: f64,

    /// Volumetric unit of the vessel.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub unit: UnitDefinition,

    /// Whether the volume of the vessel is constant or not. Default is True.

    #[builder(default = "true.into()", setter(into))]
    #[derivative(Default(value = "true"))]
    pub constant: bool,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Protein object represents enzymes and other proteins involved in
/// the experiment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Protein {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_protein_jsonld_header()")]
    #[derivative(Default(value = "default_protein_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Identifier of the protein, such as a UniProt ID, or a custom
    /// identifier.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the protein.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Whether the concentration of the protein is constant through the
    /// experiment or not. Default is True.

    #[builder(default = "true.into()", setter(into))]
    #[derivative(Default(value = "true"))]
    pub constant: bool,

    /// Amino acid sequence of the protein
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub sequence: Option<String>,

    /// Identifier of the vessel this protein has been applied to.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub vessel_id: Option<String>,

    /// EC number of the protein.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub ecnumber: Option<String>,

    /// Expression host organism of the protein.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub organism: Option<String>,

    /// Taxonomy identifier of the expression host.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub organism_tax_id: Option<String>,

    /// List of references to publications, database entries, etc. that
    /// describe or reference the protein.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    #[derivative(Default)]
    pub references: Vec<String>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Complex object allows the grouping of multiple species using
/// their . This enables the representation of protein-small molecule
/// complexes (e.g., enzyme-substrate complexes) as well as buffer or
/// solvent mixtures (combinations of SmallMolecule species).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Complex {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_complex_jsonld_header()")]
    #[derivative(Default(value = "default_complex_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Unique identifier of the complex.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the complex.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Whether the concentration of the complex is constant through the
    /// experiment or not. Default is False.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub constant: bool,

    /// Unique identifier of the vessel this complex has been used in.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub vessel_id: Option<String>,

    /// Array of IDs the complex contains
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_participants")))]
    #[derivative(Default)]
    pub participants: Vec<String>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The SmallMolecule object represents small chemical compounds that
/// participate in the experiment as substrates, products, or
/// modifiers. It captures key molecular identifiers like SMILES and
/// InChI.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct SmallMolecule {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_smallmolecule_jsonld_header()")]
    #[derivative(Default(value = "default_smallmolecule_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Identifier of the small molecule, such as a Pubchem ID, ChEBI ID, or a
    /// custom identifier.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the small molecule.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Whether the concentration of the small molecule is constant through
    /// the experiment or not. Default is False.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub constant: bool,

    /// Identifier of the vessel this small molecule has been used in.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub vessel_id: Option<String>,

    /// Canonical Simplified Molecular-Input Line-Entry System (SMILES)
    /// encoding of the small molecule.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub canonical_smiles: Option<String>,

    /// International Chemical Identifier (InChI) encoding of the small
    /// molecule.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub inchi: Option<String>,

    /// Hashed International Chemical Identifier (InChIKey) encoding of the
    /// small molecule.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub inchikey: Option<String>,

    /// List of synonymous names for the small molecule.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_synonymous_names")))]
    #[derivative(Default)]
    pub synonymous_names: Vec<String>,

    /// List of references to publications, database entries, etc. that
    /// describe or reference the small molecule.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_references")))]
    #[derivative(Default)]
    pub references: Vec<String>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Reaction object represents a chemical or enzymatic reaction and
/// holds the different species and modifiers that are part of the
/// reaction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Reaction {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_reaction_jsonld_header()")]
    #[derivative(Default(value = "default_reaction_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Unique identifier of the reaction.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the reaction.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Whether the reaction is reversible or irreversible. Default is False.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub reversible: bool,

    /// Mathematical expression of the reaction.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub kinetic_law: Option<Equation>,

    /// List of reactants that are part of the reaction.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_reactants")))]
    #[derivative(Default)]
    pub reactants: Vec<ReactionElement>,

    /// List of products that are part of the reaction.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_products")))]
    #[derivative(Default)]
    pub products: Vec<ReactionElement>,

    /// List of reaction elements that are not part of the reaction but
    /// influence it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_modifiers")))]
    #[derivative(Default)]
    pub modifiers: Vec<ModifierElement>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// This object is part of the object and describes a species
/// (SmallMolecule, Protein, Complex) participating in the reaction.
/// The stochiometry is of the species is specified in the field,
/// whereas negative values indicate that the species is a reactant
/// and positive values indicate that the species is a product of
/// the reaction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct ReactionElement {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_reactionelement_jsonld_header()")]
    #[derivative(Default(value = "default_reactionelement_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Internal identifier to either a protein or reactant defined in the
    /// EnzymeML Document.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub species_id: String,

    /// Float number representing the associated stoichiometry.

    #[builder(default = "1.0.into()", setter(into))]
    #[derivative(Default(value = "1.0"))]
    pub stoichiometry: f64,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The ModifierElement object represents a species that is not part of
/// the reaction but influences it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct ModifierElement {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_modifierelement_jsonld_header()")]
    #[derivative(Default(value = "default_modifierelement_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Internal identifier to either a protein or reactant defined in the
    /// EnzymeML Document.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub species_id: String,

    /// Role of the modifier in the reaction.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub role: ModifierRole,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// The Equation object describes a mathematical equation used to model
/// parts of a reaction system.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Equation {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_equation_jsonld_header()")]
    #[derivative(Default(value = "default_equation_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Identifier of a defined species (SmallMolecule, Protein, Complex).
    /// Represents the left hand side of the equation.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub species_id: String,

    /// Mathematical expression of the equation. Represents the right hand
    /// side of the equation.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub equation: String,

    /// Type of the equation.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub equation_type: EquationType,

    /// List of variables that are part of the equation
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_variables")))]
    #[derivative(Default)]
    pub variables: Vec<Variable>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// This object describes a variable that is part of an equation.
/// Variables can represent species concentrations, time, or other
/// quantities that appear in mathematical expressions. Each variable
/// must have a unique identifier, name, and symbol that is used in
/// equations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Variable {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_variable_jsonld_header()")]
    #[derivative(Default(value = "default_variable_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Identifier of the variable.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the variable.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Equation symbol of the variable.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub symbol: String,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// This object describes parameters used in kinetic models, including
/// estimated values, bounds, and associated uncertainties.
/// Parameters can represent rate constants, binding constants, or
/// other numerical values that appear in rate equations or other
/// mathematical expressions.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Parameter {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_parameter_jsonld_header()")]
    #[derivative(Default(value = "default_parameter_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Identifier of the parameter.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the parameter.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Equation symbol of the parameter.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub symbol: String,

    /// Numerical value of the estimated parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub value: Option<f64>,

    /// Unit of the estimated parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub unit: Option<UnitDefinition>,

    /// Initial value that was used for the parameter estimation.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub initial_value: Option<f64>,

    /// Upper bound for the parameter value that was used for the parameter
    /// estimation
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub upper_bound: Option<f64>,

    /// Lower bound for the parameter value that was used for the parameter
    /// estimation
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub lower_bound: Option<f64>,

    /// Whether this parameter should be varied or not in the context of an
    /// optimization.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = "true.into()", setter(into))]
    #[derivative(Default(value = "true.into()"))]
    pub fit: Option<bool>,

    /// Standard error of the estimated parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub stderr: Option<f64>,

    /// Specifies if this parameter is constant. Default is True.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = "true.into()", setter(into))]
    #[derivative(Default(value = "true.into()"))]
    pub constant: Option<bool>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// This object describes a single measurement, which includes time
/// course data of any type defined in DataTypes. It contains initial
/// concentrations and measurement data for all species involved in
/// the experiment. Multiple measurements can be grouped together
/// using the group_id field to indicate they are part of the same
/// experimental series.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct Measurement {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_measurement_jsonld_header()")]
    #[derivative(Default(value = "default_measurement_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Unique identifier of the measurement.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub id: String,

    /// Name of the measurement

    #[builder(setter(into))]
    #[derivative(Default)]
    pub name: String,

    /// Measurement data of all species that were part of the measurement. A
    /// species refers to a Protein, Complex, or SmallMolecule.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_species_data")))]
    #[derivative(Default)]
    pub species_data: Vec<MeasurementData>,

    /// User-defined group ID to signal relationships between measurements.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub group_id: Option<String>,

    /// pH value of the measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub ph: Option<f64>,

    /// Temperature of the measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub temperature: Option<f64>,

    /// Unit of the temperature of the measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub temperature_unit: Option<UnitDefinition>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// This object describes a single entity of a measurement, which
/// corresponds to one species (Protein, Complex, SmallMolecule). It
/// contains time course data for that species, including the initial
/// amount, prepared amount, and measured data points over time.
/// Endpoint data is treated as a time course data point with only one
/// data point.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct MeasurementData {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_measurementdata_jsonld_header()")]
    #[derivative(Default(value = "default_measurementdata_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// The identifier for the described reactant.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub species_id: String,

    /// Amount of the the species before starting the measurement. This field
    /// can be used for specifying the prepared amount of a species
    /// in the reaction mix. Not to be confused with , specifying
    /// the concentration of a species at the first data point from
    /// the array.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub prepared: Option<f64>,

    /// Initial amount of the measurement data. This must be the same as the
    /// first data point in the array.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub initial: Option<f64>,

    /// SI unit of the data that was measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub data_unit: Option<UnitDefinition>,

    /// Data that was measured.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_data")))]
    #[derivative(Default)]
    pub data: Vec<f64>,

    /// Corresponding time points of the .
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_time")))]
    #[derivative(Default)]
    pub time: Vec<f64>,

    /// Unit of the time points of the .
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub time_unit: Option<UnitDefinition>,

    /// Type of data that was measured (e.g. concentration, absorbance, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub data_type: Option<DataTypes>,

    /// Whether or not the data has been generated by simulation. Default
    /// is False.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub is_simulated: Option<bool>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// Represents a unit definition that is based on the SI unit system.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct UnitDefinition {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_unitdefinition_jsonld_header()")]
    #[derivative(Default(value = "default_unitdefinition_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Unique identifier of the unit definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub id: Option<String>,

    /// Common name of the unit definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub name: Option<String>,

    /// Base units that define the unit.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[builder(default, setter(into, each(name = "to_base_units")))]
    #[derivative(Default)]
    pub base_units: Vec<BaseUnit>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

/// Represents a base unit in the unit definition.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Builder, Derivative)]
#[derivative(Default)]
#[serde(default)]
#[allow(non_snake_case)]
pub struct BaseUnit {
    /// JSON-LD header
    #[serde(flatten)]
    #[builder(default = "default_baseunit_jsonld_header()")]
    #[derivative(Default(value = "default_baseunit_jsonld_header()"))]
    pub jsonld: Option<JsonLdHeader>,

    /// Kind of the base unit (e.g., meter, kilogram, second).

    #[builder(setter(into))]
    #[derivative(Default)]
    pub kind: UnitType,

    /// Exponent of the base unit in the unit definition.

    #[builder(setter(into))]
    #[derivative(Default)]
    pub exponent: i64,

    /// Multiplier of the base unit in the unit definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub multiplier: Option<f64>,

    /// Scale of the base unit in the unit definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    #[derivative(Default)]
    pub scale: Option<f64>,

    /// Additional properties outside of the schema
    #[serde(flatten)]
    #[builder(default)]
    pub additional_properties: Option<HashMap<String, Value>>,
}

//
// Enum definitions
//
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum ModifierRole {
    #[default]
    #[serde(rename = "activator")]
    Activator,

    #[serde(rename = "additive")]
    Additive,

    #[serde(rename = "biocatalyst")]
    Biocatalyst,

    #[serde(rename = "buffer")]
    Buffer,

    #[serde(rename = "catalyst")]
    Catalyst,

    #[serde(rename = "inhibitor")]
    Inhibitor,

    #[serde(rename = "solvent")]
    Solvent,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum EquationType {
    #[default]
    #[serde(rename = "assignment")]
    Assignment,

    #[serde(rename = "initialAssignment")]
    InitialAssignment,

    #[serde(rename = "ode")]
    Ode,

    #[serde(rename = "rateLaw")]
    RateLaw,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum DataTypes {
    #[default]
    #[serde(rename = "absorbance")]
    Absorbance,

    #[serde(rename = "amount")]
    Amount,

    #[serde(rename = "concentration")]
    Concentration,

    #[serde(rename = "conversion")]
    Conversion,

    #[serde(rename = "fluorescence")]
    Fluorescence,

    #[serde(rename = "peakarea")]
    PeakArea,

    #[serde(rename = "transmittance")]
    Transmittance,

    #[serde(rename = "turnover")]
    Turnover,

    #[serde(rename = "yield")]
    Yield_,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub enum UnitType {
    #[default]
    #[serde(rename = "ampere")]
    Ampere,

    #[serde(rename = "avogadro")]
    Avogadro,

    #[serde(rename = "becquerel")]
    Becquerel,

    #[serde(rename = "candela")]
    Candela,

    #[serde(rename = "celsius")]
    Celsius,

    #[serde(rename = "coulomb")]
    Coulomb,

    #[serde(rename = "dimensionless")]
    Dimensionless,

    #[serde(rename = "farad")]
    Farad,

    #[serde(rename = "gram")]
    Gram,

    #[serde(rename = "gray")]
    Gray,

    #[serde(rename = "henry")]
    Henry,

    #[serde(rename = "hertz")]
    Hertz,

    #[serde(rename = "item")]
    Item,

    #[serde(rename = "joule")]
    Joule,

    #[serde(rename = "katal")]
    Katal,

    #[serde(rename = "kelvin")]
    Kelvin,

    #[serde(rename = "kilogram")]
    Kilogram,

    #[serde(rename = "litre")]
    Litre,

    #[serde(rename = "lumen")]
    Lumen,

    #[serde(rename = "lux")]
    Lux,

    #[serde(rename = "metre")]
    Metre,

    #[serde(rename = "mole")]
    Mole,

    #[serde(rename = "newton")]
    Newton,

    #[serde(rename = "ohm")]
    Ohm,

    #[serde(rename = "pascal")]
    Pascal,

    #[serde(rename = "radian")]
    Radian,

    #[serde(rename = "second")]
    Second,

    #[serde(rename = "siemens")]
    Siemens,

    #[serde(rename = "sievert")]
    Sievert,

    #[serde(rename = "steradian")]
    Steradian,

    #[serde(rename = "tesla")]
    Tesla,

    #[serde(rename = "volt")]
    Volt,

    #[serde(rename = "watt")]
    Watt,

    #[serde(rename = "weber")]
    Weber,
}

// Default JSON-LD header function for each object

pub fn default_enzymemldocument_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "name".to_string(),
        TermDef::Simple("schema:title".to_string()),
    );
    context.terms.insert(
        "created".to_string(),
        TermDef::Simple("schema:dateCreated".to_string()),
    );
    context.terms.insert(
        "modified".to_string(),
        TermDef::Simple("schema:dateModified".to_string()),
    );
    context.terms.insert(
        "creators".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:creator".to_string()),
            type_: None,
            container: Some("@list".to_string()),
            context: None,
        }),
    );
    context.terms.insert(
        "references".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:citation".to_string()),
            type_: Some("@id".to_string()),
            container: Some("@list".to_string()),
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:EnzymeMLDocument/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:EnzymeMLDocument".to_string()])),
    })
}

pub fn default_creator_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "given_name".to_string(),
        TermDef::Simple("schema:givenName".to_string()),
    );
    context.terms.insert(
        "family_name".to_string(),
        TermDef::Simple("schema:familyName".to_string()),
    );
    context.terms.insert(
        "mail".to_string(),
        TermDef::Simple("schema:email".to_string()),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Creator/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec![
            "enzml:Creator".to_string(),
            "schema:person".to_string(),
        ])),
    })
}

pub fn default_vessel_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "name".to_string(),
        TermDef::Simple("schema:name".to_string()),
    );
    context.terms.insert(
        "volume".to_string(),
        TermDef::Simple("OBO:OBI_0002139".to_string()),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Vessel/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec![
            "enzml:Vessel".to_string(),
            "OBO:OBI_0400081".to_string(),
        ])),
    })
}

pub fn default_protein_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "name".to_string(),
        TermDef::Simple("schema:name".to_string()),
    );
    context.terms.insert(
        "sequence".to_string(),
        TermDef::Simple("OBO:GSSO_007262".to_string()),
    );
    context.terms.insert(
        "vessel_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "organism".to_string(),
        TermDef::Simple("OBO:OBI_0100026".to_string()),
    );
    context.terms.insert(
        "organism_tax_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: None,
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "references".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:citation".to_string()),
            type_: Some("@id".to_string()),
            container: Some("@list".to_string()),
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Protein/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec![
            "enzml:Protein".to_string(),
            "OBO:PR_000000001".to_string(),
        ])),
    })
}

pub fn default_complex_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "name".to_string(),
        TermDef::Simple("schema:name".to_string()),
    );
    context.terms.insert(
        "vessel_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "participants".to_string(),
        TermDef::Detailed(TermDetail {
            id: None,
            type_: Some("@id".to_string()),
            container: Some("@list".to_string()),
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Complex/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Complex".to_string()])),
    })
}

pub fn default_smallmolecule_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "name".to_string(),
        TermDef::Simple("schema:name".to_string()),
    );
    context.terms.insert(
        "vessel_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "references".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:citation".to_string()),
            type_: Some("@id".to_string()),
            container: Some("@list".to_string()),
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:SmallMolecule/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:SmallMolecule".to_string()])),
    })
}

pub fn default_reaction_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Reaction/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Reaction".to_string()])),
    })
}

pub fn default_reactionelement_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "species_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:ReactionElement/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:ReactionElement".to_string()])),
    })
}

pub fn default_modifierelement_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "species_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:ModifierElement/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:ModifierElement".to_string()])),
    })
}

pub fn default_equation_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "species_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Equation/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Equation".to_string()])),
    })
}

pub fn default_variable_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Variable/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Variable".to_string()])),
    })
}

pub fn default_parameter_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Parameter/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Parameter".to_string()])),
    })
}

pub fn default_measurement_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "id".to_string(),
        TermDef::Detailed(TermDetail {
            id: Some("schema:identifier".to_string()),
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );
    context.terms.insert(
        "group_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: None,
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:Measurement/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:Measurement".to_string()])),
    })
}

pub fn default_measurementdata_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms
    context.terms.insert(
        "species_id".to_string(),
        TermDef::Detailed(TermDetail {
            id: None,
            type_: Some("@id".to_string()),
            container: None,
            context: None,
        }),
    );

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:MeasurementData/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:MeasurementData".to_string()])),
    })
}

pub fn default_unitdefinition_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:UnitDefinition/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:UnitDefinition".to_string()])),
    })
}

pub fn default_baseunit_jsonld_header() -> Option<JsonLdHeader> {
    let mut context = SimpleContext::default();

    // Add main prefix and repository URL
    context.terms.insert(
        "enzml".to_string(),
        TermDef::Simple("http://www.enzymeml.org/v2/".to_string()),
    );

    // Add configured prefixes
    context.terms.insert(
        "OBO".to_string(),
        TermDef::Simple("http://purl.obolibrary.org/obo/".to_string()),
    );
    context.terms.insert(
        "schema".to_string(),
        TermDef::Simple("https://schema.org/".to_string()),
    );

    // Add attribute terms

    Some(JsonLdHeader {
        context: Some(JsonLdContext::Object(context)),
        id: Some(format!("enzml:BaseUnit/{}", uuid::Uuid::new_v4())),
        type_: Some(TypeOrVec::Multi(vec!["enzml:BaseUnit".to_string()])),
    })
}

/// JSON-LD Header
///
/// JSON-LD (JavaScript Object Notation for Linked Data) provides a way to express
/// linked data using JSON syntax, enabling semantic web technologies and structured
/// data interchange with context and meaning preservation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Eq, PartialEq)]
pub struct JsonLdHeader {
    /// JSON-LD context (IRI, object, or array)
    #[serde(rename = "@context", skip_serializing_if = "Option::is_none")]
    pub context: Option<JsonLdContext>,

    /// Node identifier (IRI or blank node)
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type IRI(s) for the node, e.g. schema:Person
    #[serde(rename = "@type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<TypeOrVec>,
}

impl Default for JsonLdHeader {
    /// Returns the default JSON-LD header.
    fn default() -> Self {
        Self {
            context: None,
            id: None,
            type_: None,
        }
    }
}

impl JsonLdHeader {
    /// Adds a new term definition to the JSON-LD context, creating a context object if none exists.
    ///
    /// This method provides a convenient way to extend the JSON-LD context with additional term
    /// mappings, allowing for semantic annotation of properties and values within the document.
    /// If the header does not already contain a context, a new SimpleContext object will be
    /// created automatically to hold the term definition.
    ///
    /// # Arguments
    ///
    /// * `name` - The term name to be defined in the context
    /// * `term` - The term definition, either a simple IRI mapping or a detailed definition
    ///
    /// # Example
    ///
    /// ```no_compile
    /// let mut header = JsonLdHeader::default();
    /// header.add_term("name", TermDef::Simple("https://schema.org/name".to_string()));
    /// ```
    pub fn add_term(&mut self, name: &str, term: TermDef) {
        let context = self
            .context
            .get_or_insert_with(|| JsonLdContext::Object(SimpleContext::default()));

        if let JsonLdContext::Object(object) = context {
            object.terms.insert(name.to_string(), term);
        }
    }

    /// Updates an existing term definition in the JSON-LD context or adds it if it doesn't exist.
    ///
    /// This method functions similarly to add_term but provides clearer semantics when the
    /// intention is to modify an existing term definition. The behavior is identical to add_term
    /// as HashMap::insert will overwrite existing entries with the same key, but this method
    /// name makes the intent more explicit in code that is updating rather than initially
    /// defining terms.
    ///
    /// # Arguments
    ///
    /// * `name` - The term name to be updated in the context
    /// * `term` - The new term definition to replace any existing definition
    ///
    /// # Example
    ///
    /// ```no_compile
    /// let mut header = JsonLdHeader::default();
    /// header.add_term("name", TermDef::Simple("https://schema.org/name".to_string()));
    /// header.update_term("name", TermDef::Simple("https://example.org/fullName".to_string()));
    /// ```
    pub fn update_term(&mut self, name: &str, term: TermDef) {
        let context = self
            .context
            .get_or_insert_with(|| JsonLdContext::Object(SimpleContext::default()));

        if let JsonLdContext::Object(object) = context {
            object.terms.insert(name.to_string(), term);
        }
    }

    /// Removes a term definition from the JSON-LD context if it exists.
    ///
    /// This method allows for the removal of previously defined terms from the JSON-LD context,
    /// which can be useful when dynamically managing context definitions or when certain terms
    /// are no longer needed in the semantic annotation of the document. The method will only
    /// attempt removal if the context exists and is an object type; it will silently do nothing
    /// if the context is missing or is not an object.
    ///
    /// # Arguments
    ///
    /// * `name` - The term name to be removed from the context
    ///
    /// # Returns
    ///
    /// Returns `true` if the term was found and removed, `false` if the term was not present
    /// or if the context is not an object type.
    ///
    /// # Example
    ///
    /// ```no_compile
    /// let mut header = JsonLdHeader::default();
    /// header.add_term("name", TermDef::Simple("https://schema.org/name".to_string()));
    /// let was_removed = header.remove_term("name");
    /// assert!(was_removed);
    /// ```
    pub fn remove_term(&mut self, name: &str) -> bool {
        if let Some(JsonLdContext::Object(object)) = &mut self.context {
            object.terms.remove(name).is_some()
        } else {
            false
        }
    }
}

/// Accept either a single type IRI or an array of them.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Eq, PartialEq)]
#[serde(untagged)]
pub enum TypeOrVec {
    Single(String),
    Multi(Vec<String>),
}

/// JSON-LD Context:
/// - a single IRI (remote context)
/// - an inline context object
/// - or an array of these (merged sequentially)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Eq, PartialEq)]
#[serde(untagged)]
pub enum JsonLdContext {
    Iri(String),
    Object(SimpleContext),
    Array(Vec<JsonLdContext>),
}

/// A simple inline @context object with essential global keys and term definitions.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, Eq, PartialEq)]
pub struct SimpleContext {
    /// Base IRI used for relative resolution.
    #[serde(rename = "@base", skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,

    /// Default vocabulary IRI for terms without explicit IRIs.
    #[serde(rename = "@vocab", skip_serializing_if = "Option::is_none")]
    pub vocab: Option<String>,

    /// Default language for string literals.
    #[serde(rename = "@language", skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Mapping of term → IRI or detailed definition.
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty", default)]
    pub terms: HashMap<String, TermDef>,
}

/// Term definition can be a simple mapping (string → IRI) or a detailed object.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Eq, PartialEq)]
#[serde(untagged)]
pub enum TermDef {
    /// Simple alias: `"name": "https://schema.org/name"`
    Simple(String),
    /// Expanded form with type coercion, container behavior, or nested context.
    Detailed(TermDetail),
}

/// Detailed term definition (subset of JSON-LD 1.1 features).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, Eq, PartialEq)]
pub struct TermDetail {
    /// Absolute or relative IRI that the term expands to, or a keyword like "@id".
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type coercion or value type ("@id", "@vocab", or datatype IRI).
    #[serde(rename = "@type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,

    /// Container behavior ("@list", "@set", "@index", etc.).
    #[serde(rename = "@container", skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,

    /// Optional nested (scoped) context.
    #[serde(rename = "@context", skip_serializing_if = "Option::is_none")]
    pub context: Option<Box<JsonLdContext>>,
}
