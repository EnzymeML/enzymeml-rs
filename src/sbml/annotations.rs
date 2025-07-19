//! EnzymeML annotation management for SBML conversion.
//!
//! This module serves as the central hub for managing annotations during SBML ↔ EnzymeML conversion.
//! It handles multiple versions of the EnzymeML specification (v1 and v2) through internally tagged
//! enums that automatically detect and deserialize the appropriate version based on XML namespace attributes.
//!
//! ## Architecture Overview
//!
//! The SBML conversion system is organized into three main components:
//! - **reader.rs**: Converts SBML documents to EnzymeML entities using `TryFrom` implementations
//! - **writer.rs**: Converts EnzymeML entities to SBML documents using mapping functions  
//! - **annotations.rs** (this module): Manages version-specific annotation data extraction and application
//! - **version.rs**: Defines version enums and the `EnzymeMLAnnotation` trait
//!
//! ## Annotation Processing Flow
//!
//! ### During SBML → EnzymeML conversion (reader.rs):
//! 1. SBML entities are parsed and converted to base EnzymeML structures
//! 2. Version-specific annotations are extracted from SBML using this module's enums
//! 3. The `EnzymeMLAnnotation::apply()` trait method enhances EnzymeML entities with annotation data
//!
//! ### During EnzymeML → SBML conversion (writer.rs):
//! 1. EnzymeML entities are mapped to SBML structures
//! 2. Version-specific annotations are created from EnzymeML data
//! 3. Annotations are serialized and attached to the appropriate SBML elements
//!
//! ## Version Detection
//!
//! The module uses internally tagged enums with the `@xmlns` attribute for automatic version detection:
//! - `http://sbml.org/enzymeml/version1` or `http://sbml.org/enzymeml/version2` → EnzymeML v1
//! - `https://www.enzymeml.org/v2` → EnzymeML v2
//!
//! This allows seamless parsing of mixed-version documents and proper handling of legacy formats.

use crate::{
    prelude::{Complex, Measurement, Parameter, Protein, SmallMolecule, UnitDefinition},
    sbml::{
        error::SBMLError,
        utils::IsEmpty,
        v1::{self},
        v2,
        version::{EnzymeMLAnnotation, EnzymeMLVersion},
    },
    try_versions,
};
use quick_xml::impl_deserialize_for_internally_tagged_enum;
use sbml::{prelude::CombineArchive, Annotation, SBMLDocument};
use serde::{Deserialize, Serialize};
use variantly::Variantly;

/// Data annotations containing measurement information across EnzymeML versions.
///
/// This enum handles the extraction and serialization of measurement data that is embedded
/// within SBML documents as annotations. It automatically detects the appropriate version
/// and delegates processing to version-specific handlers.
///
/// ## Version-Specific Behavior
/// - **V1**: Measurement data is stored as reaction-level annotations
/// - **V2**: Measurement data is stored as model-level annotations
#[derive(Debug, Variantly, Serialize, Deserialize, PartialEq)]
#[serde(rename = "data", untagged)]
pub(crate) enum DataAnnot {
    /// EnzymeML v1 data annotation format
    V1(v1::schema::DataAnnot),
    /// EnzymeML v2 data annotation format
    V2(v2::schema::DataAnnot),
}

impl DataAnnot {
    /// Extracts measurement data from SBML documents and COMBINE archives.
    ///
    /// This method delegates to version-specific extraction logic that parses measurement
    /// data from both the SBML annotation and associated data files in the archive.
    ///
    /// # Arguments
    /// * `sbml` - The SBML document containing annotation metadata
    /// * `archive` - The COMBINE archive containing measurement data files
    ///
    /// # Returns
    /// Vector of measurements extracted from the archive, or an error if extraction fails
    pub(crate) fn extract_measurements(
        &self,
        sbml: &SBMLDocument,
        archive: &mut CombineArchive,
    ) -> Result<Vec<Measurement>, SBMLError> {
        match self {
            DataAnnot::V1(annot) => v1::extract::extract_measurements(sbml, annot, archive),
            DataAnnot::V2(annot) => v2::extract::extract_measurements(sbml, annot, archive),
        }
    }

    /// Serializes measurement data as SBML annotations.
    ///
    /// This method converts EnzymeML measurement data into version-appropriate SBML annotations
    /// and attaches them to the correct location within the SBML document structure.
    ///
    /// # Arguments  
    /// * `sbmldoc` - The target SBML document to annotate
    /// * `measurements` - The measurement data to serialize
    /// * `version` - The target EnzymeML version for annotation format
    ///
    /// # Returns
    /// Success or an error if serialization fails
    pub(crate) fn to_sbml(
        sbmldoc: &SBMLDocument,
        measurements: &[Measurement],
        version: &EnzymeMLVersion,
    ) -> Result<(), SBMLError> {
        let model = sbmldoc.model().unwrap();

        match version {
            EnzymeMLVersion::V1 => {
                let annot = v1::DataAnnot::try_from(measurements)?;
                if !annot.is_empty() {
                    model
                        .set_reactions_annotation_serde::<v1::DataAnnot>(&annot)
                        .map_err(SBMLError::from)?;
                }
            }
            EnzymeMLVersion::V2 => {
                let annot = v2::DataAnnot::try_from(measurements)?;
                if !annot.is_empty() {
                    model
                        .set_annotation_serde::<v2::DataAnnot>(&annot)
                        .map_err(SBMLError::from)?;
                }
            }
        }

        Ok(())
    }
}

impl TryFrom<&SBMLDocument> for DataAnnot {
    type Error = SBMLError;

    /// Attempts to extract data annotations from an SBML document.
    ///
    /// This function tries to parse data annotations in order of version preference (v2, then v1)
    /// to maintain compatibility with both current and legacy EnzymeML formats.
    ///
    /// # Arguments
    /// * `sbml` - The SBML document to extract annotations from
    ///
    /// # Returns
    /// The extracted data annotation, or an error if no valid annotation is found
    ///
    /// # Errors
    /// Returns `SBMLError::MissingDataAnnotation` if no recognizable data annotation is present
    fn try_from(sbml: &SBMLDocument) -> Result<Self, Self::Error> {
        try_versions!(
            sbml,
            (v2::DataAnnot, DataAnnot::V2),
            (v1::DataAnnot, DataAnnot::V1),
        );

        Err(SBMLError::MissingDataAnnotation)
    }
}

/// Small molecule annotations containing chemical identifiers and structural information.
///
/// This enum handles version-specific small molecule annotation data, providing a unified
/// interface for chemical information regardless of the underlying EnzymeML version format.
///
/// ## Version Differences
/// - **V1**: Uses `ReactantAnnot` with basic chemical identifiers (InChI, SMILES)
/// - **V2**: Uses dedicated `SmallMoleculeAnnot` with enhanced chemical information (InChI Key, canonical SMILES)
#[derive(Debug, PartialEq, Serialize, Clone)]
pub(crate) enum SmallMoleculeAnnot {
    /// EnzymeML v1 small molecule annotation (stored as ReactantAnnot)
    V1(v1::schema::ReactantAnnot),
    /// EnzymeML v2 small molecule annotation with enhanced chemical data
    V2(v2::schema::SmallMoleculeAnnot),
}

impl_deserialize_for_internally_tagged_enum! {
    SmallMoleculeAnnot, "@xmlns",
    ("http://sbml.org/enzymeml/version1"    => V1(v1::schema::ReactantAnnot)),
    ("http://sbml.org/enzymeml/version2" => V1(v1::schema::ReactantAnnot)),
    ("https://www.enzymeml.org/v2" => V2(v2::schema::SmallMoleculeAnnot)),
}

impl EnzymeMLAnnotation<SmallMolecule> for SmallMoleculeAnnot {
    /// Applies annotation data to enhance a SmallMolecule entity.
    ///
    /// This method extracts chemical identifiers and structural information from the annotation
    /// and applies them to the target small molecule, with version-specific field mapping.
    ///
    /// # Arguments
    /// * `smallmol` - The small molecule entity to enhance with annotation data
    fn apply(self, smallmol: &mut SmallMolecule) {
        match self {
            SmallMoleculeAnnot::V1(annot) => {
                smallmol.inchi = annot.inchi;
                smallmol.canonical_smiles = annot.smiles;
            }
            SmallMoleculeAnnot::V2(annot) => {
                smallmol.inchikey = annot.inchikey;
                smallmol.inchi = annot.inchi;
                smallmol.canonical_smiles = annot.canonical_smiles
            }
        }
    }

    fn expected_tags() -> Vec<String> {
        vec!["reactant".to_string(), "smallMolecule".to_string()]
    }
}

/// Protein annotations containing sequence and taxonomic information.
///
/// This enum manages protein-specific annotation data across EnzymeML versions, providing
/// access to sequence information, enzyme classification, and organism details.
///
/// ## Version Compatibility
/// Both v1 and v2 support the same core protein fields (sequence, EC number, organism, taxonomy ID),
/// ensuring consistent behavior across versions.
#[derive(Debug, PartialEq, Serialize, Clone)]
pub(crate) enum ProteinAnnot {
    /// EnzymeML v1 protein annotation
    V1(v1::schema::ProteinAnnot),
    /// EnzymeML v2 protein annotation
    V2(v2::schema::ProteinAnnot),
}

impl_deserialize_for_internally_tagged_enum! {
    ProteinAnnot, "@xmlns",
    ("http://sbml.org/enzymeml/version1"    => V1(v1::schema::ProteinAnnot)),
    ("http://sbml.org/enzymeml/version2" => V1(v1::schema::ProteinAnnot)),
    ("https://www.enzymeml.org/v2" => V2(v2::schema::ProteinAnnot)),
}

impl EnzymeMLAnnotation<Protein> for ProteinAnnot {
    /// Applies annotation data to enhance a Protein entity.
    ///
    /// This method extracts protein sequence, enzyme classification numbers, and organism
    /// information from the annotation and applies them to the target protein entity.
    ///
    /// # Arguments
    /// * `protein` - The protein entity to enhance with annotation data
    fn apply(self, protein: &mut Protein) {
        match self {
            ProteinAnnot::V1(annot) => {
                protein.sequence = annot.sequence;
                protein.ecnumber = annot.ecnumber;
                protein.organism = annot.organism;
                protein.organism_tax_id = annot.organism_tax_id;
            }
            ProteinAnnot::V2(annot) => {
                protein.sequence = annot.sequence;
                protein.ecnumber = annot.ecnumber;
                protein.organism = annot.organism;
                protein.organism_tax_id = annot.organism_tax_id;
            }
        }
    }

    fn expected_tags() -> Vec<String> {
        vec!["protein".to_string()]
    }
}

/// Complex annotations containing participant information.
///
/// This enum manages complex-specific annotation data that describes which molecular
/// entities participate in multi-component complexes within the enzymatic system.
///
/// ## Version Compatibility
/// Both v1 and v2 support the same participant structure, ensuring consistent
/// complex representation across EnzymeML versions.
#[derive(Debug, PartialEq, Serialize, Clone)]
pub(crate) enum ComplexAnnot {
    /// EnzymeML v1 complex annotation
    V1(v1::schema::ComplexAnnot),
    /// EnzymeML v2 complex annotation
    V2(v2::schema::ComplexAnnot),
}

impl_deserialize_for_internally_tagged_enum! {
    ComplexAnnot, "@xmlns",
    ("http://sbml.org/enzymeml/version1"    => V1(v1::schema::ComplexAnnot)),
    ("http://sbml.org/enzymeml/version2" => V1(v1::schema::ComplexAnnot)),
    ("https://www.enzymeml.org/v2" => V2(v2::schema::ComplexAnnot)),
}

impl EnzymeMLAnnotation<Complex> for ComplexAnnot {
    /// Applies annotation data to enhance a Complex entity.
    ///
    /// This method extracts participant information from the annotation and applies it
    /// to the target complex entity, defining which molecular entities form the complex.
    ///
    /// # Arguments
    /// * `complex` - The complex entity to enhance with annotation data
    fn apply(self, complex: &mut Complex) {
        match self {
            ComplexAnnot::V1(annot) => {
                complex.participants = annot.participants;
            }
            ComplexAnnot::V2(annot) => {
                complex.participants = annot.participants;
            }
        }
    }

    fn expected_tags() -> Vec<String> {
        vec!["complex".to_string()]
    }
}

/// Parameter annotations containing statistical and constraint information.
///
/// This enum manages parameter-specific annotation data including initial values,
/// boundary constraints, uncertainty estimates, and unit definitions that enhance
/// the basic parameter information stored in SBML.
///
/// ## Version Differences
/// - **V1**: Supports initial value, upper/lower bounds, and unit references
/// - **V2**: Adds standard error (stderr) field for enhanced statistical information
#[derive(Debug, PartialEq, Serialize, Clone)]
pub(crate) enum ParameterAnnot {
    /// EnzymeML v1 parameter annotation
    V1(v1::schema::ParameterAnnot),
    /// EnzymeML v2 parameter annotation with enhanced statistical data
    V2(v2::schema::ParameterAnnot),
}

impl_deserialize_for_internally_tagged_enum! {
    ParameterAnnot, "@xmlns",
    ("http://sbml.org/enzymeml/version1"    => V1(v1::schema::ParameterAnnot)),
    ("http://sbml.org/enzymeml/version2" => V1(v1::schema::ParameterAnnot)),
    ("https://www.enzymeml.org/v2" => V2(v2::schema::ParameterAnnot)),
}

impl EnzymeMLAnnotation<Parameter> for ParameterAnnot {
    /// Applies annotation data to enhance a Parameter entity.
    ///
    /// This method extracts statistical information, boundary constraints, and unit data
    /// from the annotation and applies them to the target parameter entity.
    ///
    /// # Arguments
    /// * `parameter` - The parameter entity to enhance with annotation data
    fn apply(self, parameter: &mut Parameter) {
        match self {
            ParameterAnnot::V1(annot) => {
                parameter.initial_value = annot.initial;
                parameter.upper_bound = annot.upper;
                parameter.lower_bound = annot.lower;
                // Only set unit if not already present with more complete information
                if parameter.unit.is_none() {
                    parameter.unit = annot.unit.map(|unit_id| UnitDefinition {
                        id: Some(unit_id),
                        ..Default::default()
                    });
                }
            }
            ParameterAnnot::V2(annot) => {
                parameter.initial_value = annot.initial;
                parameter.upper_bound = annot.upper_bound;
                parameter.lower_bound = annot.lower_bound;
                parameter.stderr = annot.stderr;
                // Only set unit if not already present with more complete information
                if parameter.unit.is_none() {
                    parameter.unit = annot.unit.map(|unit_id| UnitDefinition {
                        id: Some(unit_id),
                        ..Default::default()
                    });
                }
            }
        }
    }

    fn expected_tags() -> Vec<String> {
        vec!["parameter".to_string()]
    }
}

/// Variables annotations for EnzymeML version 2.
///
/// This annotation type is exclusive to EnzymeML v2 and provides enhanced variable
/// tracking and metadata for mathematical expressions and equations within the model.
/// Variables annotations are not available in EnzymeML v1.
#[derive(Debug, PartialEq, Serialize, Clone)]
pub(crate) enum VariablesAnnot {
    /// EnzymeML v2 variables annotation (not available in v1)
    V2(v2::schema::VariablesAnnot),
}

impl_deserialize_for_internally_tagged_enum! {
    VariablesAnnot, "@xmlns",
    ("https://www.enzymeml.org/v2" => V2(v2::schema::VariablesAnnot)),
}

// DEFAULT IMPLEMENTATIONS
//
// These provide sensible defaults for annotation types, defaulting to the latest
// version (v2) when creating new annotation instances programmatically.
impl Default for SmallMoleculeAnnot {
    fn default() -> Self {
        Self::V2(v2::schema::SmallMoleculeAnnot::default())
    }
}

impl Default for ProteinAnnot {
    fn default() -> Self {
        Self::V2(v2::schema::ProteinAnnot::default())
    }
}

impl Default for ComplexAnnot {
    fn default() -> Self {
        Self::V2(v2::schema::ComplexAnnot::default())
    }
}

impl Default for ParameterAnnot {
    fn default() -> Self {
        Self::V2(v2::schema::ParameterAnnot::default())
    }
}

impl Default for VariablesAnnot {
    fn default() -> Self {
        Self::V2(v2::schema::VariablesAnnot::default())
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::de::from_str;

    use super::*;

    /// Test automatic version detection for v2 small molecule annotations.
    #[test]
    fn test_parse_v2_small_mol_annot() {
        let xml = r#"
          <smallMolecule xmlns="https://www.enzymeml.org/v2">
            <inchiKey>QTBSBXVTEAMEQO-UHFFFAOYSA-N</inchiKey>
            <smiles>CC(=O)O</smiles>
          </smallMolecule>
        "#;

        let parsed: SmallMoleculeAnnot = from_str(xml).unwrap();

        assert!(matches!(parsed, SmallMoleculeAnnot::V2(_)));
    }

    #[test]
    fn test_parse_v2_small_mol_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let small_molecule = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:OBO="http://purl.obolibrary.org/obo/"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Protein/f753a055-ecc5-4de8-bf37-2950f8c8c777">
                <schema:name>Enzyme</schema:name>
                <OBO:GSSO_007262>MTEY</OBO:GSSO_007262>
                <OBO:OBI_0100026>E.coli</OBO:OBI_0100026>
                <rdf:type rdf:resource="http://purl.obolibrary.org/obo/PR_000000001" />
                <rdf:type rdf:resource="https://schema.org/Protein" />
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Protein" />
                </rdf:Description>
            </rdf:RDF>
            <smallMolecule xmlns="https://www.enzymeml.org/v2">
                <inchiKey>QTBSBXVTEAMEQO-UHFFFAOYSA-N</inchiKey>
                <smiles>CC(=O)O</smiles>
            </smallMolecule>
        "#;

        small_molecule.set_annotation(annotation).unwrap();
        let parsed = SmallMoleculeAnnot::extract(small_molecule.as_ref(), "smallMolecule").unwrap();
        assert!(matches!(*parsed, SmallMoleculeAnnot::V2(_)));
    }

    #[test]
    fn test_parse_v1_protein_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let protein = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:OBO="http://purl.obolibrary.org/obo/"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Protein/f753a055-ecc5-4de8-bf37-2950f8c8c777">
                <schema:name>Enzyme</schema:name>
                <OBO:GSSO_007262>MTEY</OBO:GSSO_007262>
                <OBO:OBI_0100026>E.coli</OBO:OBI_0100026>
                <rdf:type rdf:resource="http://purl.obolibrary.org/obo/PR_000000001" />
                <rdf:type rdf:resource="https://schema.org/Protein" />
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Protein" />
                </rdf:Description>
            </rdf:RDF>
            <enzymeml:protein xmlns:enzymeml="http://sbml.org/enzymeml/version1">
                    <enzymeml:sequence>MRR</enzymeml:sequence>
                    <enzymeml:ECnumber>3.1.1.43</enzymeml:ECnumber>
                    <enzymeml:uniprotID>B0RS62</enzymeml:uniprotID>
                    <enzymeml:organism>Xanthomonas campestris pv. campestris</enzymeml:organism>
            </enzymeml:protein>
        "#;

        protein.set_annotation(annotation).unwrap();
        let parsed = ProteinAnnot::extract(protein.as_ref(), "protein").unwrap();
        assert!(matches!(*parsed, ProteinAnnot::V1(_)));
    }

    #[test]
    fn test_parse_v2_protein_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let protein = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:OBO="http://purl.obolibrary.org/obo/"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Protein/f753a055-ecc5-4de8-bf37-2950f8c8c777">
                <schema:name>Enzyme</schema:name>
                <OBO:GSSO_007262>MTEY</OBO:GSSO_007262>
                <OBO:OBI_0100026>E.coli</OBO:OBI_0100026>
                <rdf:type rdf:resource="http://purl.obolibrary.org/obo/PR_000000001" />
                <rdf:type rdf:resource="https://schema.org/Protein" />
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Protein" />
                </rdf:Description>
            </rdf:RDF>
            <protein xmlns="https://www.enzymeml.org/v2">
                <ecnumber>1.1.1.1</ecnumber>
                <organism>E.coli</organism>
                <organismTaxId>12345</organismTaxId>
                <sequence>MTEY</sequence>
            </protein>
        "#;

        protein.set_annotation(annotation).unwrap();
        let parsed = ProteinAnnot::extract(protein.as_ref(), "protein").unwrap();
        assert!(matches!(*parsed, ProteinAnnot::V2(_)));
    }

    #[test]
    fn test_parse_v1_small_mol_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let small_molecule = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:OBO="http://purl.obolibrary.org/obo/"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/SmallMolecule/test">
                <schema:name>PGME</schema:name>
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/SmallMolecule" />
                </rdf:Description>
            </rdf:RDF>
            <enzymeml:reactant xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:inchi>1S/C9H11NO2/c1-12-9(11)7-10-8-5-3-2-4-6-8/h2-6,10H,7H2,1H3</enzymeml:inchi>
                <enzymeml:smiles>O(C([C@@H](C1=CC=CC=C1)N([H])[H])=O)C</enzymeml:smiles>
            </enzymeml:reactant>
        "#;

        small_molecule.set_annotation(annotation).unwrap();
        let parsed = SmallMoleculeAnnot::extract(small_molecule.as_ref(), "reactant").unwrap();
        assert!(matches!(*parsed, SmallMoleculeAnnot::V1(_)));
    }

    #[test]
    fn test_parse_v1_complex_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let complex = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:OBO="http://purl.obolibrary.org/obo/"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Complex/test">
                <schema:name>E·PGME</schema:name>
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Complex" />
                </rdf:Description>
            </rdf:RDF>
            <enzymeml:complex xmlns:enzymeml="http://sbml.org/enzymeml/version2">
                <enzymeml:participant>p0</enzymeml:participant>
                <enzymeml:participant>s0</enzymeml:participant>
            </enzymeml:complex>
        "#;

        complex.set_annotation(annotation).unwrap();
        let parsed = ComplexAnnot::extract(complex.as_ref(), "complex").unwrap();
        assert!(matches!(*parsed, ComplexAnnot::V1(_)));
    }

    #[test]
    fn test_parse_v2_complex_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let complex = model.create_species("test");
        let annotation = r#"
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:schema="https://schema.org/">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Complex/03d6a454-f53e-4c0b-9b3a-37e72a2c28a4">
                <schema:name>Enzyme-Substrate Complex</schema:name>
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Complex" />
                </rdf:Description>
            </rdf:RDF>
            <complex xmlns="https://www.enzymeml.org/v2">
                <participants>p0</participants>
                <participants>s0</participants>
            </complex>
        "#;

        complex.set_annotation(annotation).unwrap();
        let parsed = ComplexAnnot::extract(complex.as_ref(), "complex").unwrap();
        assert!(matches!(*parsed, ComplexAnnot::V2(_)));
    }

    #[test]
    fn test_parse_v1_parameter_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let parameter = model.create_parameter("test");
        let annotation = r#"
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v1/Parameter/test">
                <rdf:type rdf:resource="http://www.enzymeml.org/v1/Parameter" />
                </rdf:Description>
            </rdf:RDF>
            <enzymeml:parameter xmlns:enzymeml="http://sbml.org/enzymeml/version1">
                <enzymeml:initial>10.0</enzymeml:initial>
                <enzymeml:upper>100.0</enzymeml:upper>
                <enzymeml:lower>0.0</enzymeml:lower>
                <enzymeml:unit>u1</enzymeml:unit>
            </enzymeml:parameter>
        "#;

        parameter.set_annotation(annotation).unwrap();
        let parsed = ParameterAnnot::extract(parameter.as_ref(), "parameter").unwrap();
        assert!(matches!(*parsed, ParameterAnnot::V1(_)));
    }

    #[test]
    fn test_parse_v2_parameter_annot_from_sbml() {
        let sbmldoc = SBMLDocument::default();
        let model = sbmldoc.create_model("test");
        let parameter = model.create_parameter("test");
        let annotation = r#"
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description
                rdf:about="http://www.enzymeml.org/v2/Parameter/d5261296-b6d1-47cb-a397-238427422a44">
                <rdf:type rdf:resource="http://www.enzymeml.org/v2/Parameter" />
                </rdf:Description>
            </rdf:RDF>
            <parameter xmlns="https://www.enzymeml.org/v2">
                <lowerBound>0.0</lowerBound>
                <upperBound>100.0</upperBound>
                <stdDeviation>0.1</stdDeviation>
            </parameter>
        "#;

        parameter.set_annotation(annotation).unwrap();
        let parsed = ParameterAnnot::extract(parameter.as_ref(), "parameter").unwrap();
        assert!(matches!(*parsed, ParameterAnnot::V2(_)));
    }

    /// Test serialization and deserialization roundtrip for data annotations.
    #[test]
    fn test_data_annot_serialization() {
        let annot = DataAnnot::V1(v1::DataAnnot::default());
        let serialized = serde_json::to_string(&annot).unwrap();
        let deserialized: DataAnnot = serde_json::from_str(&serialized).unwrap();
        assert_eq!(annot, deserialized);
    }
}
