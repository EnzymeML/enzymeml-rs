---
repo: "http://www.enzymeml.org/v2/"
prefix: "enzml"
prefixes:
  schema: "https://schema.org/"
  OBO: "http://purl.obolibrary.org/obo/"
---

# EnzymeML

EnzymeML is a data exchange format that supports the comprehensive documentation of enzymatic data by describing
reaction conditions, time courses of substrate and product concentrations, the kinetic model, and the estimated kinetic
constants. EnzymeML is based on the Systems Biology Markup Language, which was extended by implementing the STRENDA
Guidelines. An EnzymeML document serves as a container to transfer data between experimental platforms, modeling tools,
and databases. EnzymeML supports the scientific community by introducing a standardized data exchange format to make
enzymatic data findable, accessible, interoperable, and reusable according to the FAIR data principles.

## Root objects

### EnzymeMLDocument

This is the root object that composes all objects found in an EnzymeML document. It also includes general metadata such
as the name of the document, when it was created/modified, and references to publications, databases, and arbitrary
links to the web.

- **name**
    - Type: string
    - Description: Title of the EnzymeML Document.
    - Term: schema:title
- references
    - Type: Identifier[]
    - Description: Contains references to publications, databases, and arbitrary links to the web.
    - Term: schema:citation
- created
    - Type: string
    - Description: Date the EnzymeML document was created.
    - Term: schema:dateCreated
- modified
    - Type: string
    - Description: Date the EnzymeML document was modified.
    - Term: schema:dateModified
- creators
    - Type: Creator[]
    - Description: Contains all authors that are part of the experiment.
    - Term: schema:creator
- vessels
    - Type: Vessel[]
    - Description: Contains all vessels that are part of the experiment.
- proteins
    - Type: Protein[]
    - Description: Contains all proteins that are part of the experiment.
- complexes
    - Type: Complex[]
    - Description: Contains all complexes that are part of the experiment.
- small_molecules
    - Type: SmallMolecule[]
    - Description: Contains all reactants that are part of the experiment.
- reactions
    - Type: Reaction[]
    - Description: Dictionary mapping from reaction IDs to reaction-describing objects.
- measurements
    - Type: Measurement[]
    - Description: Contains measurements that describe outcomes of an experiment.
- equations
    - Type: Equation[]
    - Description: Contains ordinary differential equations that describe the kinetic model.
- parameters
    - Type: Parameter[]
    - Description: List of parameters that are part of the equation

## General information

### Creator (schema:person)

The creator object contains all information about authors that contributed to the resulting document.

- **given_name**
    - Type: string
    - Description: Given name of the author or contributor.
    - Term: schema:givenName
- **family_name**
    - Type: string
    - Description: Family name of the author or contributor.
    - Term: schema:familyName
- **mail**
    - Type: string
    - Description: Email address of the author or contributor.
    - Term: schema:email

## Species

### Vessel (OBO:OBI_0400081)

This object describes vessels in which the experiment has been carried out. These can include any type of vessel used in
biocatalytic experiments.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the vessel.
    - Term: schema:identifier
- **name**
    - Type: string
    - Description: Name of the used vessel.
    - Term: schema:name
- **volume**
    - Type: float
    - Description: Volumetric value of the vessel.
    - Term: OBO:OBI_0002139
- **unit**
    - Type: UnitDefinition
    - Description: Volumetric unit of the vessel.
- **constant**
    - Type: boolean
    - Description: Whether the volume of the vessel is constant or not.
    - Default: True

### Protein (schema:Protein)

This object describes the proteins that were used or formed throughout the experiment.

- **id**
    - Type: Identifier
    - Description: Unique internal identifier of the protein.
    - schema:identifier
- **name**
    - Type: string
    - Term: schema:name
- **constant**
    - Type: boolean
    - Default: False
- sequence
    - Type: string
    - Description: Amino acid sequence of the protein
    - Term: OBO:GSSO_007262
- vessel_id
    - Type: Identifier
    - Description: Unique identifier of the vessel this protein has been used in.
    - Term: schema:identifier
- ecnumber
    - Type: string
    - Description: EC number of the protein.
- organism
    - Type: string
    - Description: Organism the protein was expressed in.
    - Term: OBO:OBI_0100026
- organism_tax_id
    - Type: Identifier
    - Description: Taxonomy identifier of the expression host.
- references
    - Type: Identifier[]
    - Description: Array of references to publications, database entries, etc. that describe the protein.
    - Term: schema:citation

### Complex

This object describes complexes made of reactants and/or proteins that were used or produced in the course of the
experiment.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the complex.
    - Term: schema:identifier
- **name**
    - Type: string
    - Term: schema:name
- **constant**
    - Type: boolean
    - Default: False
- vessel_id
    - Type: Identifier
    - Description: Unique identifier of the vessel this complex has been used in.
    - Term: schema:identifier
- participants
    - Type: Identifier[]
    - Description: Array of IDs the complex contains

### SmallMolecule

This object describes the reactants that were used or produced in the course of the experiment.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the small molecule.
    - Term: schema:identifier
- **name**
    - Type: string
    - Term: schema:name
- **constant**
    - Type: boolean
    - Default: False
- vessel_id
    - Type: Identifier
    - Description: Unique identifier of the vessel this small molecule has been used in.
    - Term: schema:identifier
- canonical_smiles
    - Type: string
    - Description: Canonical Simplified Molecular-Input Line-Entry System (SMILES) encoding of the reactant.
- inchi
    - Type: string
    - Description: International Chemical Identifier (InChI) encoding of the reactant.
- inchikey
    - Type: string
    - Description: Hashed International Chemical Identifier (InChIKey) encoding of the reactant.
- references
    - Type: Identifier[]
    - Description: Array of references to publications, database entries, etc. that describe the reactant.
    - Term: schema:citation

## EnzymeReaction

### Reaction

This object describes a chemical or enzymatic reaction that was investigated in the course of the experiment. All
species used within this object need to be part of the data model.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the reaction.
    - Term: schema:identifier
- **name**
    - Type: string
    - Description: Name of the reaction.
- **reversible**
    - Type: boolean
    - Description: Whether the reaction is reversible or irreversible
    - Default: False
- kinetic_law
    - Type: Equation
    - Description: Mathematical expression of the reaction.
- species
    - Type: ReactionElement[]
    - Description: List of reaction elements that are part of the reaction.
- modifiers
    - Type: Identifier[]
    - Description: List of reaction elements that are not part of the reaction but influence it.

### ReactionElement

This object is part of the Reaction object and describes either an educt, product or modifier. The latter includes
buffers, counter-ions as well as proteins/enzymes.

- **species_id**
    - Type: Identifier
    - Description: Internal identifier to either a protein or reactant defined in the EnzymeMLDocument.
    - schema:identifier
- **stoichiometry**
    - Type: float
    - Description: Float number representing the associated stoichiometry.

## Modelling

### Equation

This object describes an equation that can be used to model the kinetics of a reaction. There are different types of
equations that can be used to model the kinetics of a reaction. The equation can be an ordinary differential equation, a
rate law or assignment rule.

- species_id
    - Type: Identifier
    - Description: Internal identifier to a species defined in the EnzymeMLDocument, given it is a rate equation.
- **equation**
    - Type: string
    - Description: Mathematical expression of the equation.
- **equation_type**
    - Type: EquationType
    - Description: Type of the equation.
- variables
    - Type: Variable[]
    - Description: List of variables that are part of the equation

### Variable

This object describes a variable that is part of an equation.

- **id**
    - Type: string
    - Description: Unique identifier of the variable.
    - Term: schema:identifier
- **name**
    - Type: string
    - Description: Name of the variable.
- **symbol**
    - Type: string
    - Description: Symbol of the variable.

### Parameter

This object describes the parameters of the kinetic model and can include all estimated values.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the parameter.
    - Term: schema:identifier
- **name**
    - Type: string
    - Description: Name of the parameter.
- **symbol**
    - Type: string
    - Description: Symbol of the parameter.
- value
    - Type: float
    - Description: Numerical value of the estimated parameter.
- unit
    - Type: UnitDefinition
    - Description: Unit of the estimated parameter.
- initial_value
    - Type: float
    - Description: Initial value that was used for the parameter estimation.
- upper
    - Type: float
    - Description: Upper bound of the estimated parameter.
- lower
    - Type: float
    - Description: Lower bound of the estimated parameter.
- stderr
    - Type: float
    - Description: Standard error of the estimated parameter.
- constant
    - Type: boolean
    - Description: Specifies if this parameter is constant
    - Default: True

## Time course data handling

### Measurement

This object describes the result of a measurement, which includes time course data of any type defined in DataTypes. It
includes initial concentrations of all species used in a single measurement.

- **id**
    - Type: Identifier
    - Description: Unique identifier of the measurement.
    - Term: schema:identifier
- **name**
    - Type: string
    - Description: Name of the measurement
- species_data
    - Type: MeasurementData[]
    - Description: Measurement data of all species that were part of the measurement. A species can refer to a protein,
      complex, or small molecule.
- group_id
    - Type: Identifier
    - Description: User-defined group ID to signal relationships between measurements.
- ph
    - Type: float
    - Description: PH value of the measurement.
    - Minimum: 0
    - Maximum: 14
- temperature
    - Type: float
    - Description: Temperature of the measurement.
- temperature_unit
    - Type: UnitDefinition
    - Description: Unit of the temperature of the measurement.

### MeasurementData

This object describes a single entity of a measurement, which corresponds to one species. It also holds replicates that
contain time course data.

- **species_id**
    - Type: Identifier
    - Description: The identifier for the described reactant.
- prepared
    - Type: float
    - Description: Amount of the reactant before the measurement. This field should be used for specifying the prepared
      amount of a species in the reaction mix. Not to be confused with `initial`, specifying the concentration at the
      first data point from the `data` array.
- **initial**
    - Type: float
    - Description: Initial amount of the measurement data. This must be the same as the first data point in the `data`
      array.
- **data_unit**
    - Type: UnitDefinition
    - Description: SI unit of the data that was measured.
- data
    - Type: float[]
    - Description: Data that was measured.
- time
    - Type: float[]
    - Description: Time steps of the replicate.
- time_unit
    - Type: UnitDefinition
    - Description: Time unit of the replicate.
- **data_type**
    - Type: DataTypes
    - Description: Type of data that was measured (e.g. concentration)
- **is_simulated**
    - Type: boolean
    - Description: Whether or not the data has been generated by simulation.
    - Default: False

## Enumerations

### EquationType

These values are used to determine the type of equation.

```python
ODE = "ode"
ASSIGNMENT = "assignment"
INITIAL_ASSIGNMENT = "initialAssignment"
RATE_LAW = "rateLaw"
```

### DataTypes

These values are used to determine the type of time course data.

```python
ABSORBANCE = "http://purl.allotrope.org/ontologies/quality#AFQ_0000061"
CONCENTRATION = "http://purl.obolibrary.org/obo/PATO_0000033"
CONVERSION = "http://purl.allotrope.org/ontologies/quality#AFQ_0000226"
PEAK_AREA = "http://purl.allotrope.org/ontologies/result#AFR_0001073"
TRANSMITTANCE = "http://purl.allotrope.org/ontologies/result#AFR_0002261"
FLUORESCENCE = "http://purl.obolibrary.org/obo/PATO_0000018"
```