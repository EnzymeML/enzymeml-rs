//! SBML Document Writer for EnzymeML Conversion
//!
//! This module provides comprehensive functionality for converting EnzymeML documents
//! to SBML format and creating COMBINE archives, serving as the primary export mechanism
//! for enzymatic data to standards-compliant SBML documents. It handles the complete
//! conversion process from high-level EnzymeML entities to low-level SBML structures
//! while preserving experimental data and metadata through annotation serialization.
//!
//! ## Module Role in SBML Processing Pipeline
//!
//! This writer module works in conjunction with other SBML processing components:
//! - **Reader** (@reader.rs): Converts SBML → EnzymeML, extracting entities and annotations
//! - **Writer** (this module): Converts EnzymeML → SBML, generating compliant SBML documents
//! - **Version** (@version.rs): Manages version detection and compatibility across EnzymeML formats
//! - **Annotations** (@annotations.rs): Handles version-specific metadata serialization
//!
//! ## Conversion Process
//!
//! The writing process follows a structured pipeline:
//!
//! 1. **Document Structure Creation**: SBML document framework is established with proper
//!    namespaces and model containers
//! 2. **Entity Serialization**: EnzymeML entities (vessels, species, reactions, parameters)
//!    are converted to their SBML equivalents with appropriate attributes
//! 3. **Annotation Embedding**: EnzymeML-specific metadata is serialized as SBML annotations
//!    using version-appropriate schema formats
//! 4. **Data Export**: Measurement data is written to CSV files and referenced through
//!    annotation metadata within the SBML structure
//! 5. **Archive Assembly**: All components are packaged into a COMBINE archive with
//!    proper manifest entries and format declarations
//!
//! ## Supported EnzymeML → SBML Mappings
//!
//! The writer converts the following EnzymeML components to SBML elements:
//! - **Vessels** → Compartments (reaction environments with volume and units)
//! - **Small molecules** → Species (with SBO terms and chemical annotations)
//! - **Proteins** → Species (with enzymatic metadata and sequence information)
//! - **Complexes** → Species (with component composition and binding data)
//! - **Reactions** → Reactions (with stoichiometry, kinetics, and rate laws)
//! - **Parameters** → Parameters (with statistical metadata and unit definitions)
//! - **Equations** → Rules (mathematical relationships and assignments)
//!
//! ## Version Compatibility
//!
//! The writer supports both EnzymeML v1 and v2 formats, automatically applying
//! appropriate annotation schemas and data organization patterns. Version-specific
//! serialization ensures compatibility with corresponding readers while maintaining
//! data integrity across format transitions.
//!
//! ## COMBINE Archive Structure
//!
//! Generated archives follow COMBINE specification standards:
//! - `model.xml`: Main SBML document with EnzymeML annotations
//! - `data/*.csv`: Measurement data files referenced by annotations
//! - `manifest.xml`: Archive contents and format declarations (auto-generated)
//!
//! ## Error Handling
//!
//! Conversion errors are propagated through the `SBMLError` type, providing detailed
//! information about serialization failures, incompatible data structures, or
//! archive creation issues. The writer employs validation strategies to ensure
//! generated SBML documents conform to specification requirements.

use sbml::{
    combine::KnownFormats, model::Model, prelude::CombineArchive, Annotation, SBMLDocument,
};

use crate::{
    prelude::{
        Complex, EnzymeMLDocument, Equation, EquationType, Measurement, Parameter, Protein,
        Reaction, SmallMolecule, Vessel,
    },
    sbml::{
        annotations::DataAnnot,
        error::SBMLError,
        speciestype::{COMPLEX_SBO_TERM, PROTEIN_SBO_TERM, SMALL_MOLECULE_SBO_TERM},
        units::map_unit_definition,
        v1, v2,
        version::EnzymeMLVersion,
    },
};

/// Converts an EnzymeML document to a COMBINE archive with SBML representation.
///
/// This function serves as the primary export mechanism for EnzymeML documents,
/// creating standards-compliant COMBINE archives containing SBML models with
/// embedded EnzymeML annotations and associated measurement data files.
///
/// The generated archive includes:
/// - `model.xml`: SBML document with complete model structure and annotations
/// - `data/*.csv`: Individual measurement data files in CSV format
/// - `manifest.xml`: Archive manifest with content declarations (auto-generated)
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document to convert
/// * `version` - Target EnzymeML version for annotation format compatibility
///
/// # Returns
/// * `Result<CombineArchive, SBMLError>` - Complete COMBINE archive or conversion error
/// ```
pub fn to_omex(
    enzmldoc: &EnzymeMLDocument,
    version: EnzymeMLVersion,
) -> Result<CombineArchive, SBMLError> {
    let mut archive = CombineArchive::new();
    let sbml_doc = to_sbml(enzmldoc, &version)?;
    archive.add_entry(
        "./model.xml",
        KnownFormats::SBML,
        true,
        sbml_doc.to_xml_string().as_bytes(),
    )?;

    write_measurement_data(enzmldoc, &mut archive, &version)?;

    Ok(archive)
}

/// Converts an EnzymeML document to an SBML document with embedded annotations.
///
/// This function orchestrates the complete conversion process from EnzymeML's
/// high-level object model to SBML's XML-based representation. It handles entity
/// mapping, annotation embedding, and structural organization to produce a
/// standards-compliant SBML document.
///
/// The conversion process includes:
/// 1. **Model Framework**: Creates SBML document structure with proper versioning
/// 2. **Unit Definitions**: Extracts and maps all units from measurement data
/// 3. **Entity Conversion**: Maps vessels, species, reactions, and parameters
/// 4. **Annotation Embedding**: Serializes EnzymeML metadata as SBML annotations
/// 5. **Initial Conditions**: Sets species concentrations from measurement data
///
/// # Arguments
/// * `enzmldoc` - The source EnzymeML document
/// * `version` - Target EnzymeML version for annotation compatibility
///
/// # Returns
/// * `Result<SBMLDocument, SBMLError>` - Complete SBML document or conversion error
///
/// # SBML Structure
/// The generated document follows SBML Level 3 Version 2 specification with
/// EnzymeML-specific annotations embedded throughout the model hierarchy.
fn to_sbml(
    enzmldoc: &EnzymeMLDocument,
    version: &EnzymeMLVersion,
) -> Result<SBMLDocument, SBMLError> {
    let sbmldoc = SBMLDocument::new(3, 2, vec![]);
    let model = sbmldoc.create_model(&enzmldoc.name);
    model.set_name(&enzmldoc.name);

    collect_units(&model, &enzmldoc.measurements)?;

    // Vessels
    enzmldoc
        .vessels
        .iter()
        .try_for_each(|vessel| map_vessel(vessel, &model))?;

    // Small molecules
    enzmldoc
        .small_molecules
        .iter()
        .try_for_each(|sm| map_small_molecule(sm, &model, version))?;

    // Proteins
    enzmldoc
        .proteins
        .iter()
        .try_for_each(|protein| map_protein(protein, &model, version))?;

    // Complexes
    enzmldoc
        .complexes
        .iter()
        .try_for_each(|complex| map_complex(complex, &model, version))?;

    // Reactions
    enzmldoc
        .reactions
        .iter()
        .try_for_each(|reaction| map_reaction(reaction, &model))?;

    // Parameters
    enzmldoc
        .parameters
        .iter()
        .try_for_each(|param| map_parameter(param, &model, version))?;

    // Equations
    enzmldoc
        .equations
        .iter()
        .try_for_each(|equation| map_equation(equation, &model))?;

    DataAnnot::to_sbml(&sbmldoc, &enzmldoc.measurements, version)?;

    if let Some(measurement) = enzmldoc.measurements.first() {
        map_init_conc(&model, measurement)?;
    }

    Ok(sbmldoc)
}

/// Maps initial concentrations from measurement data to SBML species.
///
/// This function applies experimental initial conditions to the corresponding
/// species in the SBML model, ensuring that simulation starting points reflect
/// actual experimental conditions. It also propagates unit information for
/// dimensional consistency.
///
/// # Arguments
/// * `model` - The target SBML model to update
/// * `measurement` - The measurement containing initial concentration data
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Initial Condition Mapping
/// - Sets `initialConcentration` attribute for each species
/// - Applies measurement units to species declarations
/// - Uses zero as default for missing initial values
fn map_init_conc(model: &Model, measurement: &Measurement) -> Result<(), SBMLError> {
    if let Some(temperature_unit) = &measurement.temperature_unit {
        map_unit_definition(model, temperature_unit)?;
    }

    for data in measurement.species_data.iter() {
        if let Some(species) = model.get_species(&data.species_id) {
            species.set_initial_concentration(data.initial.unwrap_or(0.0));

            if let Some(unit) = &data.data_unit {
                species.set_units(map_unit_definition(model, unit)?);
            }
        }
    }

    Ok(())
}

/// Converts an EnzymeML vessel to an SBML compartment.
///
/// Vessels represent reaction environments in EnzymeML and are mapped to SBML
/// compartments with volume, units, and constancy information preserved.
/// This mapping ensures proper spatial organization of biochemical reactions.
///
/// # Arguments
/// * `vessel` - The EnzymeML vessel to convert
/// * `model` - The target SBML model for compartment creation
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Compartment Properties
/// - **ID**: Preserved from vessel identifier
/// - **Name**: Human-readable vessel description
/// - **Size**: Volume with appropriate unit definitions
/// - **Constant**: Indicates whether volume changes during simulation
fn map_vessel(vessel: &Vessel, model: &Model) -> Result<(), SBMLError> {
    let compartment = model.create_compartment(&vessel.id);
    compartment.set_name(&vessel.name);
    compartment.set_size(vessel.volume);
    compartment.set_constant(vessel.constant);
    compartment.set_unit(map_unit_definition(model, &vessel.unit)?);

    Ok(())
}

/// Converts an EnzymeML small molecule to an SBML species with chemical annotations.
///
/// Small molecules represent chemical substrates, products, and cofactors in
/// enzymatic reactions. This mapping preserves chemical identifiers, structural
/// information, and experimental metadata through version-appropriate annotations.
///
/// # Arguments
/// * `small_molecule` - The small molecule to convert
/// * `model` - The target SBML model for species creation
/// * `version` - EnzymeML version for annotation compatibility
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Species Properties
/// - **SBO Term**: Classified as small molecule (SBO:0000247)
/// - **Compartment**: Linked to containing vessel
/// - **Annotations**: Chemical identifiers (SMILES, InChI) and metadata
/// - **Initial State**: Zero concentration (set by measurement data)
fn map_small_molecule(
    small_molecule: &SmallMolecule,
    model: &Model,
    version: &EnzymeMLVersion,
) -> Result<(), SBMLError> {
    let species = model.create_species(&small_molecule.id);
    species.set_constant(small_molecule.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&small_molecule.name);
    species.set_sbo_term(SMALL_MOLECULE_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = &small_molecule.vessel_id {
        species.set_compartment(vessel_id);
    }

    match version {
        EnzymeMLVersion::V1 => {
            let annotation = v1::ReactantAnnot::from(small_molecule);
            species.set_annotation_serde::<v1::ReactantAnnot>(&annotation)?;
        }
        EnzymeMLVersion::V2 => {
            let annotation = v2::SmallMoleculeAnnot::from(small_molecule);
            species.set_annotation_serde::<v2::SmallMoleculeAnnot>(&annotation)?;
        }
    }

    Ok(())
}

/// Extracts and maps all unit definitions from measurement data to SBML.
///
/// This function ensures that all units referenced in measurement data are
/// properly defined in the SBML model before they are used by species,
/// parameters, or other model elements. It prevents validation errors and
/// ensures dimensional consistency.
///
/// # Arguments
/// * `model` - The target SBML model for unit definition creation
/// * `measurements` - The measurement data containing unit references
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Unit Collection Process
/// - Scans all measurement data for unit references
/// - Creates SBML unit definitions for each unique unit
/// - Maps both concentration and time units from experimental data
fn collect_units(model: &Model, measurements: &[Measurement]) -> Result<(), SBMLError> {
    for measurement in measurements {
        for data in measurement.species_data.iter() {
            if let Some(unit) = &data.data_unit {
                map_unit_definition(model, unit)?;
            }

            if let Some(unit) = &data.time_unit {
                map_unit_definition(model, unit)?;
            }
        }
    }

    Ok(())
}

/// Converts an EnzymeML protein to an SBML species with enzymatic annotations.
///
/// Proteins represent enzymes and other macromolecules in EnzymeML models.
/// This mapping preserves sequence information, enzymatic classifications,
/// and organism data through version-appropriate annotation formats.
///
/// # Arguments
/// * `protein` - The protein to convert
/// * `model` - The target SBML model for species creation
/// * `version` - EnzymeML version for annotation compatibility
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Species Properties
/// - **SBO Term**: Classified as protein (SBO:0000252)
/// - **Compartment**: Linked to containing vessel/cellular location
/// - **Annotations**: Amino acid sequences, EC numbers, organism information
/// - **Initial State**: Zero concentration (modified by measurement data)
fn map_protein(
    protein: &Protein,
    model: &Model,
    version: &EnzymeMLVersion,
) -> Result<(), SBMLError> {
    let species = model.create_species(&protein.id);
    species.set_constant(protein.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&protein.name);
    species.set_sbo_term(PROTEIN_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = &protein.vessel_id {
        species.set_compartment(vessel_id);
    }

    match version {
        EnzymeMLVersion::V1 => {
            let annotation = v1::ProteinAnnot::from(protein);
            species.set_annotation_serde::<v1::ProteinAnnot>(&annotation)?;
        }
        EnzymeMLVersion::V2 => {
            let annotation = v2::ProteinAnnot::from(protein);
            species.set_annotation_serde::<v2::ProteinAnnot>(&annotation)?;
        }
    }

    Ok(())
}

/// Converts an EnzymeML complex to an SBML species with composition annotations.
///
/// Complexes represent multi-component molecular assemblies formed through
/// protein-protein interactions or enzyme-substrate binding. This mapping
/// preserves component relationships and binding stoichiometry information.
///
/// # Arguments
/// * `complex` - The complex to convert
/// * `model` - The target SBML model for species creation
/// * `version` - EnzymeML version for annotation compatibility
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Species Properties
/// - **SBO Term**: Classified as complex (SBO:0000253)
/// - **Compartment**: Linked to formation environment
/// - **Annotations**: Component composition and binding information
/// - **Initial State**: Zero concentration (determined by formation kinetics)
fn map_complex(
    complex: &Complex,
    model: &Model,
    version: &EnzymeMLVersion,
) -> Result<(), SBMLError> {
    let species = model.create_species(&complex.id);
    species.set_constant(complex.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&complex.name);
    species.set_sbo_term(COMPLEX_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = complex.vessel_id.as_ref() {
        species.set_compartment(vessel_id)
    }

    match version {
        EnzymeMLVersion::V1 => {
            let annotation = v1::ComplexAnnot::from(complex);
            species.set_annotation_serde::<v1::ComplexAnnot>(&annotation)?;
        }
        EnzymeMLVersion::V2 => {
            let annotation = v2::ComplexAnnot::from(complex);
            species.set_annotation_serde::<v2::ComplexAnnot>(&annotation)?;
        }
    }

    Ok(())
}

/// Converts an EnzymeML reaction to an SBML reaction with kinetic laws.
///
/// Reactions represent enzymatic transformations with defined stoichiometry,
/// directionality, and kinetic mechanisms. This mapping preserves reaction
/// participants, stoichiometric coefficients, and rate law expressions.
///
/// # Arguments
/// * `reaction` - The reaction to convert
/// * `model` - The target SBML model for reaction creation
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Reaction Properties
/// - **Participants**: Reactants and products with stoichiometric coefficients
/// - **Reversibility**: Direction constraints for enzymatic mechanisms
/// - **Kinetic Law**: Mathematical expression for reaction rate
/// - **Modifiers**: Catalysts and regulators (handled separately)
fn map_reaction(reaction: &Reaction, model: &Model) -> Result<(), SBMLError> {
    let sbml_reaction = model.create_reaction(&reaction.id);
    sbml_reaction.set_name(&reaction.name);
    sbml_reaction.set_reversible(reaction.reversible);

    for reactant in reaction.reactants.iter() {
        let sbml_reactant =
            sbml_reaction.create_reactant(&reactant.species_id, reactant.stoichiometry);
        sbml_reactant.set_constant(false);
    }

    for product in reaction.products.iter() {
        let sbml_product = sbml_reaction.create_product(&product.species_id, product.stoichiometry);
        sbml_product.set_constant(false);
    }

    for modifier in reaction.modifiers.iter() {
        sbml_reaction.create_modifier(&modifier.species_id);
    }

    if let Some(kinetic_law) = &reaction.kinetic_law {
        sbml_reaction.create_kinetic_law(&kinetic_law.equation);
    }

    Ok(())
}

/// Converts an EnzymeML equation to an SBML rule.
///
/// Equations represent mathematical relationships between model variables,
/// including rate laws and assignment rules. This mapping classifies equation
/// types and preserves mathematical expressions for simulation frameworks.
///
/// # Arguments
/// * `equation` - The equation to convert
/// * `model` - The target SBML model for rule creation
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Rule Type Mapping
/// - **Assignment**: Algebraic equations for derived quantities
/// - **ODE**: Differential equations for dynamic species behavior
/// - **Rate Law**: Embedded in kinetic law expressions (not separate rules)
fn map_equation(equation: &Equation, model: &Model) -> Result<(), SBMLError> {
    match equation.equation_type {
        EquationType::Assignment => {
            model.create_assignment_rule(&equation.species_id, &equation.equation);
        }
        EquationType::Ode => {
            model.create_rate_rule(&equation.species_id, &equation.equation);
        }
        _ => {}
    }

    Ok(())
}

/// Converts an EnzymeML parameter to an SBML parameter with statistical metadata.
///
/// Parameters represent kinetic constants, experimental conditions, and model
/// variables with associated uncertainty information. This mapping preserves
/// parameter values, units, and statistical metadata through annotations.
///
/// # Arguments
/// * `parameter` - The parameter to convert
/// * `model` - The target SBML model for parameter creation
/// * `version` - EnzymeML version for annotation compatibility
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or mapping error
///
/// # Parameter Properties
/// - **Value**: Numerical parameter estimate
/// - **Units**: Dimensional information for consistency
/// - **Constancy**: Whether parameter varies during simulation
/// - **Annotations**: Statistical bounds, uncertainty, and fitting metadata
fn map_parameter(
    parameter: &Parameter,
    model: &Model,
    version: &EnzymeMLVersion,
) -> Result<(), SBMLError> {
    let sbml_parameter = model.create_parameter(&parameter.id);
    sbml_parameter.set_name(&parameter.name);
    sbml_parameter.set_constant(parameter.constant.unwrap_or(true));

    if let Some(value) = parameter.value {
        sbml_parameter.set_value(value)
    }

    if let Some(unit) = &parameter.unit {
        if let Ok(unit_def) = map_unit_definition(model, unit) {
            sbml_parameter.set_units(unit_def);
        }
    }

    match version {
        EnzymeMLVersion::V1 => {
            let annotation = v1::ParameterAnnot::from(parameter);
            sbml_parameter.set_annotation_serde::<v1::ParameterAnnot>(&annotation)?;
        }
        EnzymeMLVersion::V2 => {
            let annotation = v2::ParameterAnnot::from(parameter);
            sbml_parameter.set_annotation_serde::<v2::ParameterAnnot>(&annotation)?;
        }
    }

    Ok(())
}

/// Writes measurement data to CSV files within the COMBINE archive.
///
/// This function delegates to version-specific serialization modules that handle
/// the formatting and organization of experimental measurement data. The resulting
/// CSV files are referenced by annotations within the SBML model structure.
///
/// # Arguments
/// * `enzmldoc` - The source EnzymeML document containing measurement data
/// * `archive` - The target COMBINE archive for data file storage
/// * `version` - EnzymeML version determining data organization format
///
/// # Returns
/// * `Result<(), SBMLError>` - Success indication or serialization error
///
/// # Data Organization
/// - **V1**: Reaction-level organization with measurement grouping
/// - **V2**: Model-level organization with unified data structure
/// - **CSV Format**: Time-series data with species columns and metadata headers
pub(crate) fn write_measurement_data(
    enzmldoc: &EnzymeMLDocument,
    archive: &mut CombineArchive,
    version: &EnzymeMLVersion,
) -> Result<(), SBMLError> {
    match version {
        EnzymeMLVersion::V1 => v1::serializer::write_measurement_data(enzmldoc, archive),
        EnzymeMLVersion::V2 => v2::serializer::write_measurement_data(enzmldoc, archive),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use sbml::prelude::SBMLErrorSeverity;
    use std::path::PathBuf;

    use crate::{
        prelude::*,
        sbml::{
            reader::parse_omex,
            utils::{read_omex_file, read_sbml_file},
        },
    };

    use super::*;

    #[test]
    fn test_comprehensive_document_roundtrip() {
        // Create a comprehensive EnzymeML document
        let enzmldoc = create_document();

        // Convert to SBML v2
        let out_path = PathBuf::from("comprehensive_enzymeml_v2.omex");
        to_omex(&enzmldoc, EnzymeMLVersion::V2)
            .unwrap()
            .save(&out_path)
            .unwrap();

        // Read SBML back and verify it can be parsed
        let mut archive = read_omex_file(&out_path).unwrap();
        let sbml_doc = read_sbml_file(&mut archive).unwrap();
        let error_log = sbml_doc.check_consistency();

        if !error_log.valid {
            for error in error_log.errors {
                if error.severity == SBMLErrorSeverity::Error {
                    println!("Validation error: {error:#?}");
                }
            }
            panic!("SBML document has errors");
        }

        let parsed_doc = parse_omex(archive).unwrap();

        // Basic validation that the document was converted and parsed successfully
        assert_eq!(parsed_doc.name, "Comprehensive Test Document");
        assert!(!parsed_doc.vessels.is_empty());
        assert!(!parsed_doc.proteins.is_empty());
        assert!(!parsed_doc.complexes.is_empty());
        assert!(!parsed_doc.small_molecules.is_empty());
        assert!(!parsed_doc.reactions.is_empty());
        assert!(!parsed_doc.measurements.is_empty());
        assert!(!parsed_doc.parameters.is_empty());
        assert!(!parsed_doc.equations.is_empty());

        // Now check if the document is the same as the original
        let original_json = serde_json::to_value(&enzmldoc).unwrap();
        let parsed_json = serde_json::to_value(&parsed_doc).unwrap();
        assert_eq!(parsed_json, original_json);

        // Clean up test file
        std::fs::remove_file(out_path).ok();
    }

    fn create_document() -> EnzymeMLDocument {
        use crate::versions::v2::*;

        // Create unit definitions
        let volume_unit = UnitDefinitionBuilder::default()
            .id("l".to_string())
            .name("litre".to_string())
            .base_units(vec![BaseUnitBuilder::default()
                .kind(UnitType::Litre)
                .exponent(1)
                .multiplier(1.0)
                .scale(1.0)
                .build()
                .expect("Failed to build base unit")])
            .build()
            .expect("Failed to build volume unit");

        let concentration_unit = UnitDefinitionBuilder::default()
            .id("M".to_string())
            .name("molar".to_string())
            .base_units(vec![
                BaseUnitBuilder::default()
                    .kind(UnitType::Mole)
                    .exponent(1)
                    .multiplier(1.0)
                    .scale(1.0)
                    .build()
                    .expect("Failed to build mole unit"),
                BaseUnitBuilder::default()
                    .kind(UnitType::Litre)
                    .exponent(-1)
                    .multiplier(1.0)
                    .scale(1.0)
                    .build()
                    .expect("Failed to build litre unit"),
            ])
            .build()
            .expect("Failed to build concentration unit");

        let time_unit = UnitDefinitionBuilder::default()
            .id("s".to_string())
            .name("second".to_string())
            .base_units(vec![BaseUnitBuilder::default()
                .kind(UnitType::Second)
                .exponent(1)
                .multiplier(1.0)
                .scale(1.0)
                .build()
                .expect("Failed to build time unit")])
            .build()
            .expect("Failed to build time unit");

        let temperature_unit = UnitDefinitionBuilder::default()
            .id("K".to_string())
            .name("kelvin".to_string())
            .base_units(vec![BaseUnitBuilder::default()
                .kind(UnitType::Kelvin)
                .exponent(1)
                .multiplier(1.0)
                .scale(1.0)
                .build()
                .expect("Failed to build temperature unit")])
            .build()
            .expect("Failed to build temperature unit");

        // Create vessel
        let vessel = VesselBuilder::default()
            .id("vessel_01".to_string())
            .name("Reaction Vessel".to_string())
            .volume(1.0)
            .unit(volume_unit.clone())
            .constant(true)
            .build()
            .expect("Failed to build vessel");

        // Create small molecules
        let substrate = SmallMoleculeBuilder::default()
            .id("substrate_01".to_string())
            .name("Substrate A".to_string())
            .constant(false)
            .vessel_id(vessel.id.clone())
            .canonical_smiles("C1=CC=CC=C1".to_string())
            .inchi("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H".to_string())
            .inchikey("UHOVQNZJYSORNB-UHFFFAOYSA-N".to_string())
            .build()
            .expect("Failed to build substrate");

        let product = SmallMoleculeBuilder::default()
            .id("product_01".to_string())
            .name("Product B".to_string())
            .constant(false)
            .vessel_id(vessel.id.clone())
            .canonical_smiles("CC(=O)O".to_string())
            .inchi("InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)".to_string())
            .inchikey("QTBSBXVTEAMEQO-UHFFFAOYSA-N".to_string())
            .build()
            .expect("Failed to build product");

        let cofactor = SmallMoleculeBuilder::default()
            .id("cofactor_01".to_string())
            .name("NADH".to_string())
            .constant(false)
            .vessel_id(vessel.id.clone())
            .canonical_smiles("NC(=O)c1ccc".to_string())
            .inchi("InChI=1S".to_string())
            .inchikey("BOPGDPNILDQYTO-NNYOXOHSSA-N".to_string())
            .build()
            .expect("Failed to build cofactor");

        // Create protein
        let enzyme = ProteinBuilder::default()
            .id("enzyme_01".to_string())
            .name("Test Enzyme".to_string())
            .constant(true)
            .sequence("MKT".to_string())
            .vessel_id(vessel.id.clone())
            .ecnumber("EC 1.1.1.1".to_string())
            .organism("Escherichia coli".to_string())
            .organism_tax_id("511145".to_string())
            .build()
            .expect("Failed to build enzyme");

        // Create complex
        let enzyme_substrate_complex = ComplexBuilder::default()
            .id("complex_01".to_string())
            .name("Enzyme-Substrate Complex".to_string())
            .constant(false)
            .vessel_id(vessel.id.clone())
            .participants(vec![enzyme.id.clone(), substrate.id.clone()])
            .build()
            .expect("Failed to build complex");

        let enzyme_product_complex = ComplexBuilder::default()
            .id("complex_02".to_string())
            .name("Enzyme-Product Complex".to_string())
            .constant(false)
            .vessel_id(vessel.id.clone())
            .participants(vec![enzyme.id.clone(), product.id.clone()])
            .build()
            .expect("Failed to build complex");

        // Create parameters
        let kcat_param = ParameterBuilder::default()
            .id("kcat".to_string())
            .name("kcat".to_string())
            .symbol("kcat".to_string())
            .value(7.0)
            .unit(
                UnitDefinitionBuilder::default()
                    .id("persecond".to_string())
                    .name("persecond".to_string())
                    .base_units(vec![BaseUnitBuilder::default()
                        .kind(UnitType::Second)
                        .exponent(-1)
                        .multiplier(1.0)
                        .scale(1.0)
                        .build()
                        .expect("Failed to build per second unit")])
                    .build()
                    .expect("Failed to build per second unit"),
            )
            .initial_value(5.0)
            .upper_bound(15.0)
            .lower_bound(1.0)
            .stderr(0.5)
            .constant(true)
            .build()
            .expect("Failed to build kcat parameter");

        let km_param = ParameterBuilder::default()
            .id("km".to_string())
            .name("km".to_string())
            .symbol("km".to_string())
            .value(100.0)
            .unit(concentration_unit.clone())
            .initial_value(80.0)
            .upper_bound(200.0)
            .lower_bound(10.0)
            .stderr(5.0)
            .constant(true)
            .build()
            .expect("Failed to build km parameter");

        // Create equation
        let rate_equation = EquationBuilder::default()
            .species_id("v".to_string())
            .equation("kcat * substrate_01 / (km + substrate_01)".to_string())
            .equation_type(EquationType::RateLaw)
            .build()
            .expect("Failed to build equation");

        let ode = EquationBuilder::default()
            .species_id("complex_01".to_string())
            .equation("kcat * substrate_01 / (km + substrate_01)".to_string())
            .equation_type(EquationType::Ode)
            .build()
            .expect("Failed to build equation");

        let assignment = EquationBuilder::default()
            .species_id("complex_02".to_string())
            .equation("enzyme_01".to_string())
            .equation_type(EquationType::Assignment)
            .build()
            .expect("Failed to build equation");

        // Create reaction elements
        let reactant_element = ReactionElementBuilder::default()
            .species_id(substrate.id.clone())
            .stoichiometry(1.0)
            .build()
            .expect("Failed to build reactant element");

        let product_element = ReactionElementBuilder::default()
            .species_id(product.id.clone())
            .stoichiometry(1.0)
            .build()
            .expect("Failed to build product element");

        let modifier_element = ModifierElementBuilder::default()
            .species_id(enzyme.id.clone())
            .role(ModifierRole::Biocatalyst)
            .build()
            .expect("Failed to build modifier element");

        // Create reaction
        let reaction = ReactionBuilder::default()
            .id("reaction_01".to_string())
            .name("Enzymatic Conversion".to_string())
            .reversible(false)
            .kinetic_law(rate_equation.clone())
            .reactants(vec![reactant_element])
            .products(vec![product_element])
            .modifiers(vec![modifier_element])
            .build()
            .expect("Failed to build reaction");

        // Create measurement data
        let substrate_measurement_data = MeasurementDataBuilder::default()
            .species_id(substrate.id.clone())
            .prepared(1000.0)
            .initial(1000.0)
            .data_unit(concentration_unit.clone())
            .data(vec![1000.0, 900.0, 800.0, 700.0, 600.0])
            .time(vec![0.0, 60.0, 120.0, 180.0, 240.0])
            .time_unit(time_unit.clone())
            .data_type(DataTypes::Concentration)
            .build()
            .expect("Failed to build substrate measurement data");

        let product_measurement_data = MeasurementDataBuilder::default()
            .species_id(product.id.clone())
            .prepared(0.0)
            .initial(0.0)
            .data_unit(concentration_unit.clone())
            .data(vec![0.0, 100.0, 200.0, 300.0, 400.0])
            .time(vec![0.0, 60.0, 120.0, 180.0, 240.0])
            .time_unit(time_unit.clone())
            .data_type(DataTypes::Concentration)
            .build()
            .expect("Failed to build product measurement data");

        let enzyme_measurement_data = MeasurementDataBuilder::default()
            .species_id(enzyme.id.clone())
            .prepared(10.0)
            .initial(10.0)
            .data_unit(concentration_unit.clone())
            .data(vec![10.0, 10.0, 10.0, 10.0, 10.0])
            .time(vec![0.0, 60.0, 120.0, 180.0, 240.0])
            .time_unit(time_unit.clone())
            .data_type(DataTypes::Concentration)
            .build()
            .expect("Failed to build enzyme measurement data");

        // Create measurement
        let measurement = MeasurementBuilder::default()
            .id("measurement_01".to_string())
            .name("Time Course Experiment".to_string())
            .species_data(vec![
                substrate_measurement_data,
                product_measurement_data,
                enzyme_measurement_data,
            ])
            .ph(7.4)
            .temperature(298.15)
            .temperature_unit(temperature_unit)
            .build()
            .expect("Failed to build measurement");

        // Create the main document
        EnzymeMLDocumentBuilder::default()
            .name("Comprehensive Test Document".to_string())
            .version("2.0".to_string())
            .vessels(vec![vessel])
            .proteins(vec![enzyme])
            .complexes(vec![enzyme_substrate_complex, enzyme_product_complex])
            .small_molecules(vec![substrate, product, cofactor])
            .reactions(vec![reaction])
            .measurements(vec![measurement])
            .equations(vec![ode, assignment])
            .parameters(vec![kcat_param, km_param])
            .build()
            .expect("Failed to build document")
    }
}
