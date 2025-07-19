use regex;
use sbml::Annotation;

use crate::sbml::error::SBMLError;

/// Enum representing different versions of EnzymeML format.
///
/// EnzymeML has evolved through different versions with varying annotation
/// schemas and data structures. This enum helps identify and handle
/// version-specific differences during SBML parsing and conversion.
pub enum EnzymeMLVersion {
    /// EnzymeML version 1 format
    V1,
    /// EnzymeML version 2 format
    V2,
}

impl AsRef<EnzymeMLVersion> for EnzymeMLVersion {
    fn as_ref(&self) -> &EnzymeMLVersion {
        self
    }
}

impl std::str::FromStr for EnzymeMLVersion {
    type Err = String;

    /// Parse a string into an EnzymeML version.
    ///
    /// # Arguments
    /// * `s` - The string to parse
    ///
    /// # Returns
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1.0" | "1" | "v1" => Ok(EnzymeMLVersion::V1),
            "2.0" | "2" | "v2" => Ok(EnzymeMLVersion::V2),
            _ => Err(format!("Invalid EnzymeML version: {s}")),
        }
    }
}

/// Macro to try extracting data annotations from an SBMLDocument across multiple EnzymeML versions.
///
/// This macro attempts to parse annotations using different version-specific annotation types,
/// returning the first successful match. It's used to maintain backward compatibility when
/// working with EnzymeML documents of different versions.
///
/// # Arguments
/// * `$sbml` - The SBML document to extract annotations from
/// * `$(($version:path, $variant:path))` - Tuples of (annotation type, enum variant) to try
///
/// # Example
/// ```ignore
/// try_versions!(
///     sbml,
///     (v2::DataAnnot, DataAnnot::V2),
///     (v1::DataAnnot, DataAnnot::V1),
/// );
/// ```
#[macro_export]
macro_rules! try_versions {
    ($sbml:expr, $(($version:path, $variant:path)),+ $(,)?) => {
        $(
            if let Ok(result) = <$version>::try_from($sbml) {
                return Ok($variant(result));
            }
        )+
    };
}

/// Trait for applying EnzymeML annotations to target entities.
///
/// This trait provides a unified interface for extracting annotation data from SBML
/// entities and applying them to enhance EnzymeML objects with version-specific
/// information. It handles the complexity of version differences and annotation
/// extraction while providing a clean API for data enhancement.
///
/// # Type Parameter
/// * `T` - The target entity type to be enhanced with annotation data
///
/// # Implementation Pattern
/// Implementations should handle version-specific differences in their `apply` method
/// and provide proper error handling in the `extract` method.
pub(crate) trait EnzymeMLAnnotation<T> {
    /// Apply annotation data to enhance the target entity.
    ///
    /// This method takes ownership of the annotation and applies its data to modify
    /// the properties of the target entity. Version-specific field mappings are
    /// handled within the implementation.
    ///
    /// # Arguments
    /// * `obj` - Mutable reference to the entity to be enhanced
    ///
    /// # Example
    /// ```ignore
    /// let annotation = SmallMoleculeAnnot::extract(&species, &species.id())?;
    /// annotation.apply(&mut small_molecule);
    /// ```
    fn apply(self, obj: &mut T);

    /// Get the expected tags for the annotations.
    ///
    /// This method returns a list of tags that are expected to be present in the annotation.
    /// It is used to filter out invalid annotations during extraction.
    ///
    /// # Returns
    /// A list of expected tags
    fn expected_tags() -> Vec<String>;

    /// Extract and deserialize annotation data from an SBML entity.
    ///
    /// This method searches for valid annotation data within an SBML entity,
    /// deserializes it, and returns the first recognized annotation found.
    /// It automatically filters out "Other" variants to ensure only valid
    /// data is processed.
    ///
    /// # Type Parameters
    /// * `S` - The SBML entity type that contains annotation data
    ///
    /// # Arguments
    /// * `sbml_entity` - The SBML entity to extract annotations from
    /// * `path` - Identifier path used for error reporting
    ///
    /// # Returns
    /// A boxed annotation instance containing the extracted data
    ///
    /// # Errors
    /// * `SBMLError::NoExistingAnnotation` - No valid annotation found
    /// * Serde deserialization errors from malformed annotation data
    ///
    /// # Example
    /// ```ignore
    /// let annotation = ProteinAnnot::extract(&species, &species.id())?;
    /// ```
    fn extract<S>(sbml_entity: &S, path: &str) -> Result<Box<Self>, SBMLError>
    where
        Self: Sized + serde::Serialize + for<'de> serde::Deserialize<'de> + Clone + std::fmt::Debug,
        S: Annotation + Clone,
    {
        // TODO: Find a better way to do this.
        // This is a dirty hack to get the annotations work, even when there are other annotations
        // that pollute the namespace. Unfortunately, the XML situtation is not very well defined in
        // Rust and we need to us ethi to ensure compatibility.

        // Extract the raw annotation and remove all the annotations that are not in the expected tags
        let annotation = sbml_entity.get_annotation();
        let expected_tags = Self::expected_tags();

        // First, remove the enclosing "annotation" tag
        let annotation = annotation
            .replace("<annotation>", "")
            .replace("</annotation>", "")
            .replace(":enzymeml", "")
            .replace("enzymeml:", "");
        // Extract XML fragments for each expected tag using regex
        let mut extracted_fragments = Vec::<String>::new();

        for tag in &expected_tags {
            // Create regex pattern to match the complete XML element including opening and closing tags
            // This handles both self-closing tags and tags with content, including attributes
            let pattern = format!(
                r"<{tag}(?:\s[^>]*)?(?:/>|>[\s\S]*?</{tag}>)",
                tag = regex::escape(tag)
            );

            if let Ok(re) = regex::RegexBuilder::new(&pattern)
                .dot_matches_new_line(true)
                .build()
            {
                for mat in re.find_iter(&annotation) {
                    let fragment = mat.as_str().to_string();
                    // Only add non-empty fragments
                    if !fragment.trim().is_empty() {
                        extracted_fragments.push(fragment);
                    }
                }
            }
        }

        // If we found any fragments, try to parse them
        if !extracted_fragments.is_empty() {
            // Try to parse each fragment as XML and deserialize to Self
            for fragment in &extracted_fragments {
                if let Ok(parsed) = quick_xml::de::from_str::<Self>(fragment) {
                    return Ok(Box::new(parsed));
                }
            }
        }

        Err(SBMLError::NoExistingAnnotation(path.to_string()))
    }
}
