/// SBML v1 serialization module
///
/// This module provides functionality to convert EnzymeML documents to SBML v1 format
/// and create COMBINE archives containing the SBML model and associated data.
use polars::{frame::DataFrame, io::SerWriter, prelude::CsvWriter};
use sbml::{
    combine::KnownFormats, model::Model, prelude::CombineArchive, Annotation, SBMLDocument,
};

use crate::{
    prelude::{
        Complex, DataTypes, EnzymeMLDocument, Equation, EquationType, Measurement, MeasurementData,
        Parameter, Protein, Reaction, SmallMolecule, Vessel,
    },
    sbml::{
        error::SBMLError,
        speciestype::{COMPLEX_SBO_TERM, PROTEIN_SBO_TERM, SMALL_MOLECULE_SBO_TERM},
        units::{map_unit_definition, replace_slashes},
    },
};

use super::schema::{
    ColumnAnnot, ColumnType, ComplexAnnot, DataAnnot, FileAnnot, FilesWrapper, FormatAnnot,
    FormatsWrapper, InitConcAnnot, IsEmpty, MeasurementAnnot, MeasurementsWrapper, ParameterAnnot,
    ProteinAnnot, ReactantAnnot, ENZYMEML_V1_NS,
};

/// Converts an EnzymeML document to a COMBINE archive in SBML v1 format
///
/// This function creates a COMBINE archive containing:
/// - The EnzymeML model as an SBML document
/// - CSV files for each measurement in the document
///
/// # Arguments
/// * `enzmldoc` - The EnzymeML document to convert
///
/// # Returns
/// * `Result<CombineArchive, SBMLError>` - The COMBINE archive or an error
pub fn to_v1_omex(enzmldoc: &EnzymeMLDocument) -> Result<CombineArchive, SBMLError> {
    let mut archive = CombineArchive::new();
    let sbml_doc = SBMLDocument::try_from(enzmldoc)?;
    archive.add_entry(
        "./model.xml",
        KnownFormats::SBML,
        true,
        sbml_doc.to_xml_string().as_bytes(),
    )?;

    for measurement in enzmldoc.measurements.iter() {
        let mut df: DataFrame = measurement.to_dataframe(false);
        let mut string_buffer = Vec::new();
        CsvWriter::new(&mut string_buffer)
            .include_header(false)
            .finish(&mut df)?;
        archive.add_entry(
            format!("data/{}.csv", measurement.id),
            KnownFormats::CSV,
            false,
            string_buffer.as_slice(),
        )?;
    }

    Ok(archive)
}

/// Implementation of TryFrom for converting EnzymeMLDocument to SBMLDocument
///
/// This implementation handles the conversion of all EnzymeML elements to their
/// corresponding SBML elements, including units, vessels, species, reactions,
/// parameters, and equations.
impl TryFrom<&EnzymeMLDocument> for SBMLDocument {
    type Error = SBMLError;

    fn try_from(enzmldoc: &EnzymeMLDocument) -> Result<Self, Self::Error> {
        let sbmldoc = SBMLDocument::new(3, 2, vec![]);
        let model = sbmldoc.create_model(&enzmldoc.name);
        model.set_name(&enzmldoc.name);

        collect_units(&model, &enzmldoc.measurements)?;

        for vessel in enzmldoc.vessels.iter() {
            map_vessel(vessel, &model)?;
        }

        for small_molecule in enzmldoc.small_molecules.iter() {
            map_small_molecule(small_molecule, &model)?;
        }

        for protein in enzmldoc.proteins.iter() {
            map_protein(protein, &model)?;
        }

        for complex in enzmldoc.complexes.iter() {
            map_complex(complex, &model)?;
        }

        for reaction in enzmldoc.reactions.iter() {
            map_reaction(reaction, &model)?;
        }

        for parameter in enzmldoc.parameters.iter() {
            map_parameter(parameter, &model)?;
        }

        for equation in enzmldoc.equations.iter() {
            map_equation(equation, &model)?;
        }

        let data_annot = DataAnnot::try_from(enzmldoc.measurements.as_slice())?;

        if !data_annot.is_empty() {
            model
                .set_reactions_annotation_serde::<DataAnnot>(&data_annot)
                .expect("Failed to set annotation");
        }

        if let Some(measurement) = enzmldoc.measurements.first() {
            map_init_conc(&model, measurement)?;
        }

        Ok(sbmldoc)
    }
}

/// Maps initial concentrations from a measurement to species in the SBML model
///
/// # Arguments
/// * `model` - The SBML model to update
/// * `measurement` - The measurement containing initial concentration data
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_init_conc(model: &Model, measurement: &Measurement) -> Result<(), SBMLError> {
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

/// Maps an EnzymeML Vessel to an SBML Compartment
///
/// # Arguments
/// * `vessel` - The EnzymeML vessel to map
/// * `model` - The SBML model to add the compartment to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_vessel(vessel: &Vessel, model: &Model) -> Result<(), SBMLError> {
    let compartment = model.create_compartment(&vessel.id);
    compartment.set_name(&vessel.name);
    compartment.set_size(vessel.volume);
    compartment.set_constant(vessel.constant);
    compartment.set_unit(map_unit_definition(model, &vessel.unit)?);

    Ok(())
}

/// Maps an EnzymeML SmallMolecule to an SBML Species
///
/// # Arguments
/// * `small_molecule` - The small molecule to map
/// * `model` - The SBML model to add the species to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_small_molecule(small_molecule: &SmallMolecule, model: &Model) -> Result<(), SBMLError> {
    let species = model.create_species(&small_molecule.id);
    species.set_constant(small_molecule.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&small_molecule.name);
    species.set_sbo_term(SMALL_MOLECULE_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = &small_molecule.vessel_id {
        species.set_compartment(vessel_id);
    }

    let annotation = ReactantAnnot::from(small_molecule);
    if !annotation.is_empty() {
        species
            .set_annotation_serde::<ReactantAnnot>(&annotation)
            .expect("Failed to set annotation");
    }

    Ok(())
}

/// Collects and maps all units from measurements
///
/// # Arguments
/// * `model` - The SBML model to add the units to
/// * `measurements` - The measurements containing units
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
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

/// Maps an EnzymeML Protein to an SBML Species
///
/// # Arguments
/// * `protein` - The protein to map
/// * `model` - The SBML model to add the species to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_protein(protein: &Protein, model: &Model) -> Result<(), SBMLError> {
    let species = model.create_species(&protein.id);
    species.set_constant(protein.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&protein.name);
    species.set_sbo_term(PROTEIN_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = &protein.vessel_id {
        species.set_compartment(vessel_id);
    }

    let annotation = ProteinAnnot::from(protein);
    if !annotation.is_empty() {
        species
            .set_annotation_serde::<ProteinAnnot>(&annotation)
            .expect("Failed to set annotation");
    }

    Ok(())
}

/// Maps an EnzymeML Complex to an SBML Species
///
/// # Arguments
/// * `complex` - The complex to map
/// * `model` - The SBML model to add the species to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_complex(complex: &Complex, model: &Model) -> Result<(), SBMLError> {
    let species = model.create_species(&complex.id);
    species.set_constant(complex.constant);
    species.set_has_only_substance_units(false);
    species.set_name(&complex.name);
    species.set_sbo_term(COMPLEX_SBO_TERM);
    species.set_initial_concentration(0.0);

    if let Some(vessel_id) = &complex.vessel_id {
        species.set_compartment(vessel_id);
    }

    let annotation = ComplexAnnot::from(complex);
    if !annotation.is_empty() {
        species
            .set_annotation_serde::<ComplexAnnot>(&annotation)
            .expect("Failed to set annotation");
    }

    Ok(())
}

/// Maps an EnzymeML Reaction to an SBML Reaction
///
/// # Arguments
/// * `reaction` - The reaction to map
/// * `model` - The SBML model to add the reaction to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_reaction(reaction: &Reaction, model: &Model) -> Result<(), SBMLError> {
    let sbml_reaction = model.create_reaction(&reaction.id);
    sbml_reaction.set_name(&reaction.name);
    sbml_reaction.set_reversible(reaction.reversible);

    for reactant in reaction.reactants.iter() {
        sbml_reaction.create_reactant(&reactant.species_id, reactant.stoichiometry);
    }

    for product in reaction.products.iter() {
        sbml_reaction.create_product(&product.species_id, product.stoichiometry);
    }

    if let Some(kinetic_law) = &reaction.kinetic_law {
        sbml_reaction.create_kinetic_law(&kinetic_law.equation);
    }

    Ok(())
}

/// Maps an EnzymeML Equation to an SBML Rule
///
/// # Arguments
/// * `equation` - The equation to map
/// * `model` - The SBML model to add the rule to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
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

/// Maps an EnzymeML Parameter to an SBML Parameter
///
/// # Arguments
/// * `parameter` - The parameter to map
/// * `model` - The SBML model to add the parameter to
///
/// # Returns
/// * `Result<(), SBMLError>` - Success or an error
fn map_parameter(parameter: &Parameter, model: &Model) -> Result<(), SBMLError> {
    let sbml_parameter = model.create_parameter(&parameter.id);
    sbml_parameter.set_name(&parameter.name);
    sbml_parameter.set_constant(parameter.constant.unwrap_or(true));

    if let Some(value) = parameter.value {
        sbml_parameter.set_value(value);
    }

    if let Some(unit) = &parameter.unit {
        sbml_parameter.set_units(map_unit_definition(model, unit)?);
    }

    let annotation = ParameterAnnot::from(parameter);
    if !annotation.is_empty() {
        sbml_parameter
            .set_annotation_serde::<ParameterAnnot>(&annotation)
            .expect("Failed to set annotation");
    }

    Ok(())
}

// Type conversions for annotations

/// Converts a SmallMolecule to a ReactantAnnot for SBML annotation
impl From<&SmallMolecule> for ReactantAnnot {
    fn from(small_molecule: &SmallMolecule) -> Self {
        ReactantAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            inchi: small_molecule.inchi.clone(),
            smiles: small_molecule.canonical_smiles.clone(),
            chebi_id: None,
        }
    }
}

/// Converts a Protein to a ProteinAnnot for SBML annotation
impl From<&Protein> for ProteinAnnot {
    fn from(protein: &Protein) -> Self {
        ProteinAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            sequence: protein.sequence.clone(),
            ecnumber: protein.ecnumber.clone(),
            uniprotid: None,
            organism: protein.organism.clone(),
            organism_tax_id: protein.organism_tax_id.clone(),
        }
    }
}

/// Converts a Complex to a ComplexAnnot for SBML annotation
impl From<&Complex> for ComplexAnnot {
    fn from(complex: &Complex) -> Self {
        ComplexAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            participants: complex.participants.clone(),
        }
    }
}

/// Converts a Parameter to a ParameterAnnot for SBML annotation
impl From<&Parameter> for ParameterAnnot {
    fn from(parameter: &Parameter) -> Self {
        ParameterAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            initial: parameter.initial_value,
            upper: parameter.upper_bound,
            lower: parameter.lower_bound,
        }
    }
}

/// Converts a slice of Measurements to a DataAnnot for SBML annotation
impl TryFrom<&[Measurement]> for DataAnnot {
    type Error = SBMLError;

    fn try_from(measurements: &[Measurement]) -> Result<Self, Self::Error> {
        let formats_wrapper = FormatsWrapper {
            format: measurements
                .iter()
                .map(FormatAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        };

        let measurements_wrapper = MeasurementsWrapper {
            measurement: measurements
                .iter()
                .map(MeasurementAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        };

        let files_wrapper = FilesWrapper {
            file: measurements.iter().map(FileAnnot::from).collect(),
        };

        Ok(DataAnnot {
            xmlns: ENZYMEML_V1_NS.to_string(),
            formats: formats_wrapper,
            measurements: measurements_wrapper,
            files: files_wrapper,
        })
    }
}

/// Converts a Measurement to a FileAnnot for SBML annotation
impl From<&Measurement> for FileAnnot {
    fn from(measurement: &Measurement) -> Self {
        FileAnnot {
            id: format!("file_{}", measurement.id),
            location: format!("data/{}.csv", measurement.id),
            format: measurement.id.clone(),
        }
    }
}

/// Converts a Measurement to a MeasurementAnnot for SBML annotation
impl TryFrom<&Measurement> for MeasurementAnnot {
    type Error = SBMLError;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        Ok(MeasurementAnnot {
            id: measurement.id.clone(),
            name: measurement.name.clone(),
            file: format!("file_{}", measurement.id),
            init_concs: measurement
                .species_data
                .iter()
                .map(InitConcAnnot::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// Converts MeasurementData to an InitConcAnnot for SBML annotation
impl TryFrom<&MeasurementData> for InitConcAnnot {
    type Error = SBMLError;

    fn try_from(measurement_data: &MeasurementData) -> Result<Self, Self::Error> {
        let initial = measurement_data
            .initial
            .ok_or(SBMLError::MissingInitialValue(
                measurement_data.species_id.clone(),
            ))?;

        let unit_id = measurement_data
            .data_unit
            .as_ref()
            .and_then(|unit| unit.id.clone());

        Ok(InitConcAnnot {
            protein: None,
            reactant: Some(measurement_data.species_id.clone()),
            value: initial,
            unit: unit_id.map(|id| replace_slashes(&id)),
        })
    }
}

/// Converts a Measurement to a FormatAnnot for SBML annotation
impl TryFrom<&Measurement> for FormatAnnot {
    type Error = SBMLError;

    fn try_from(measurement: &Measurement) -> Result<Self, Self::Error> {
        // This is very ugly, but it works :')
        let time_unit = measurement
            .species_data
            .iter()
            .find(|data| data.time_unit.is_some())
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .time_unit
            .as_ref()
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .id
            .as_ref()
            .ok_or(SBMLError::MissingTimeUnit(measurement.id.clone()))?
            .clone();

        let mut columns = vec![ColumnAnnot {
            species_id: None,
            column_type: ColumnType::Time,
            unit: time_unit,
            index: 0,
            replica: None,
            is_calculated: false,
        }];

        let mut index = 1;
        for data in measurement.species_data.iter() {
            if data.time.is_empty() {
                // If the time column is empty, we don't need to add it to the format
                continue;
            }

            let mut column = ColumnAnnot::try_from(data)?;
            column.index = index;
            index += 1;
            columns.push(column);
        }

        Ok(FormatAnnot {
            id: measurement.id.clone(),
            columns,
        })
    }
}

/// Converts MeasurementData to a ColumnAnnot for SBML annotation
impl TryFrom<&MeasurementData> for ColumnAnnot {
    type Error = SBMLError;

    fn try_from(measurement_data: &MeasurementData) -> Result<Self, Self::Error> {
        let column_type: ColumnType = measurement_data
            .data_type
            .as_ref()
            .unwrap_or(&DataTypes::Concentration)
            .into();

        let unit_id = &measurement_data
            .data_unit
            .as_ref()
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?;

        let unit_id = unit_id
            .id
            .as_ref()
            .ok_or(SBMLError::MissingUnit(measurement_data.species_id.clone()))?;

        Ok(ColumnAnnot {
            species_id: Some(measurement_data.species_id.clone()),
            column_type,
            unit: replace_slashes(unit_id),
            index: 0, // Indices will be set later
            replica: None,
            is_calculated: false,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::io::load_enzmldoc;

    use super::*;

    #[test]
    fn test_to_v1_sbml_document() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path().to_path_buf();
        let omex_path = temp_path.join("enzmldoc_reaction.omex");

        // Read the EnzymeML document and convert it to an OMEX file
        let original_enzmldoc = load_enzmldoc("tests/data/enzmldoc_reaction.json").unwrap();
        let mut archive = to_v1_omex(&original_enzmldoc).unwrap();
        archive.save(&omex_path).unwrap();

        // Read the OMEX file and convert it back to an EnzymeML document
        let roundtrip_enzmldoc = load_enzmldoc(&omex_path).unwrap();

        // Assert basic document properties are preserved
        assert_eq!(original_enzmldoc.name, roundtrip_enzmldoc.name);
        assert_eq!(
            original_enzmldoc.small_molecules.len(),
            roundtrip_enzmldoc.small_molecules.len()
        );
        assert_eq!(
            original_enzmldoc.proteins.len(),
            roundtrip_enzmldoc.proteins.len()
        );
        assert_eq!(
            original_enzmldoc.parameters.len(),
            roundtrip_enzmldoc.parameters.len()
        );
        assert_eq!(
            original_enzmldoc.measurements.len(),
            roundtrip_enzmldoc.measurements.len()
        );

        // Check that reactions are preserved (this will likely show data loss)
        assert_eq!(
            original_enzmldoc.reactions.len(),
            roundtrip_enzmldoc.reactions.len()
        );

        // Check specific reaction properties that are likely to be lost
        for (original_reaction, roundtrip_reaction) in original_enzmldoc
            .reactions
            .iter()
            .zip(roundtrip_enzmldoc.reactions.iter())
        {
            assert_eq!(original_reaction.id, roundtrip_reaction.id);
            assert_eq!(original_reaction.name, roundtrip_reaction.name);
            assert_eq!(original_reaction.reversible, roundtrip_reaction.reversible);
            assert_eq!(
                original_reaction.reactants.len(),
                roundtrip_reaction.reactants.len()
            );
            assert_eq!(
                original_reaction.products.len(),
                roundtrip_reaction.products.len()
            );

            // This assertion will likely fail - kinetic laws are not preserved in the SBML v1 serialization
            let kinetic_law = original_reaction.kinetic_law.as_ref().unwrap();
            let roundtrip_kinetic_law = roundtrip_reaction.kinetic_law.as_ref().unwrap();
            assert_eq!(kinetic_law.equation, roundtrip_kinetic_law.equation);
            assert_eq!(
                kinetic_law.equation_type,
                roundtrip_kinetic_law.equation_type
            );
        }

        // Check parameter values are preserved
        for (original_param, roundtrip_param) in original_enzmldoc
            .parameters
            .iter()
            .zip(roundtrip_enzmldoc.parameters.iter())
        {
            assert_eq!(original_param.id, roundtrip_param.id);
            assert_eq!(original_param.name, roundtrip_param.name);
            assert_eq!(original_param.value, roundtrip_param.value);
        }

        // Check measurement data integrity
        for (original_measurement, roundtrip_measurement) in original_enzmldoc
            .measurements
            .iter()
            .zip(roundtrip_enzmldoc.measurements.iter())
        {
            assert_eq!(original_measurement.id, roundtrip_measurement.id);
            assert_eq!(original_measurement.name, roundtrip_measurement.name);
            assert_eq!(
                original_measurement.species_data.len(),
                roundtrip_measurement.species_data.len()
            );

            // Check if time series data is preserved
            for (original_data, roundtrip_data) in original_measurement
                .species_data
                .iter()
                .zip(roundtrip_measurement.species_data.iter())
            {
                assert_eq!(original_data.species_id, roundtrip_data.species_id);
                assert_eq!(original_data.initial, roundtrip_data.initial);
                assert_eq!(original_data.data_type, roundtrip_data.data_type);
                assert_eq!(original_data.time, roundtrip_data.time);
                assert_eq!(original_data.data, roundtrip_data.data);
            }
        }

        println!("Conversion test completed. Check output for detected data loss.");
    }

    /// Tests the replace_slashes function for unit handling
    #[test]
    fn test_replace_slashes() {
        let id = "mmol / l";
        let replaced = replace_slashes(id);
        assert_eq!(replaced, "mmol__l");
    }
}
