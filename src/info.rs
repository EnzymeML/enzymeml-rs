//! Information display module for EnzymeML documents
//!
//! This module provides functionality for displaying EnzymeML documents and their components
//! in a human-readable format. It implements the `Display` trait for `EnzymeMLDocument` and
//! provides helper functions to format various components as tables.

use std::{
    collections::HashSet,
    fmt::{self, Display},
};

use tabled::{builder::Builder, settings::Style};

use crate::prelude::{
    EnzymeMLDocument, Equation, EquationType, Measurement, Parameter, Protein, Reaction,
    SmallMolecule, Vessel,
};

/// Trait for converting model components to table records
///
/// This trait defines methods that allow model components to be displayed
/// as rows in a formatted table. Implementors must provide column headers
/// and a way to convert their data to string values for each column.
trait TableRecord {
    /// Get the column headers for the table
    ///
    /// # Returns
    /// * A vector of strings representing the column headers
    fn columns() -> Vec<String>;

    /// Convert the instance to a record for display in a table
    ///
    /// # Returns
    /// * A vector of strings representing the values for each column
    fn to_record(&self) -> Vec<String>;
}

impl Display for EnzymeMLDocument {
    /// Formats an EnzymeML document for display
    ///
    /// Creates a formatted table representation of the document, including all its
    /// components (vessels, molecules, proteins, reactions, measurements, equations,
    /// and parameters) if they are present.
    ///
    /// # Arguments
    /// * `f` - The formatter to write the output to
    ///
    /// # Returns
    /// * `fmt::Result` - The result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = Builder::default();
        builder.push_record(vec!["EnzymeML Document"]);

        if !self.vessels.is_empty() {
            builder.push_record(vec!["Vessels"]);
            builder.push_record(vec![to_table(&self.vessels)]);
        }

        if !self.small_molecules.is_empty() {
            builder.push_record(vec!["Small Molecules"]);
            builder.push_record(vec![to_table(&self.small_molecules)]);
        }

        if !self.proteins.is_empty() {
            builder.push_record(vec!["Proteins"]);
            builder.push_record(vec![to_table(&self.proteins)]);
        }

        if !self.reactions.is_empty() {
            builder.push_record(vec!["Reactions"]);
            builder.push_record(vec![to_table(&self.reactions)]);
        }

        if !self.measurements.is_empty() {
            builder.push_record(vec!["Measurements"]);
            builder.push_record(vec![measurement_table(&self.measurements)]);
        }

        if !self.equations.is_empty() {
            builder.push_record(vec!["Equations"]);
            builder.push_record(vec![to_table(&self.equations)]);
        }

        if !self.parameters.is_empty() {
            builder.push_record(vec!["Parameters"]);
            builder.push_record(vec![to_table(&self.parameters)]);
        }

        let mut table = builder.build();
        table.with(Style::sharp());
        write!(f, "{}", table.to_string())?;

        Ok(())
    }
}

/// Converts a collection of TableRecord implementors to a formatted table string
///
/// # Type Parameters
/// * `T` - A type that implements the TableRecord trait
///
/// # Arguments
/// * `records` - A slice of objects implementing TableRecord
///
/// # Returns
/// * A formatted string containing the table representation
fn to_table<T: TableRecord>(records: &[T]) -> String {
    let columns = T::columns();
    let mut builder = Builder::default();

    builder.push_record(columns);

    for record in records {
        builder.push_record(record.to_record());
    }

    let mut table = builder.build();
    table.with(Style::rounded());
    table.to_string()
}

impl TableRecord for Vessel {
    /// Returns column headers for vessel tables
    ///
    /// # Returns
    /// * Vector of column names: ID, Name, Volume
    fn columns() -> Vec<String> {
        vec!["ID".to_string(), "Name".to_string(), "Volume".to_string()]
    }

    /// Converts a vessel to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the vessel's ID, name, and volume
    fn to_record(&self) -> Vec<String> {
        vec![
            self.id.to_string(),
            self.name.to_string(),
            self.volume.to_string(),
        ]
    }
}

impl TableRecord for SmallMolecule {
    /// Returns column headers for small molecule tables
    ///
    /// # Returns
    /// * Vector of column names: ID, Name, Constant, Vessel ID
    fn columns() -> Vec<String> {
        vec![
            "ID".to_string(),
            "Name".to_string(),
            "Constant".to_string(),
            "Vessel ID".to_string(),
        ]
    }

    /// Converts a small molecule to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the molecule's ID, name, constant flag, and vessel ID
    fn to_record(&self) -> Vec<String> {
        vec![
            self.id.to_string(),
            self.name.to_string(),
            self.constant.to_string(),
            self.vessel_id.clone().unwrap_or("None".to_string()),
        ]
    }
}

impl TableRecord for Protein {
    /// Returns column headers for protein tables
    ///
    /// # Returns
    /// * Vector of column names: ID, Name, Constant, Vessel ID
    fn columns() -> Vec<String> {
        vec![
            "ID".to_string(),
            "Name".to_string(),
            "Constant".to_string(),
            "Vessel ID".to_string(),
        ]
    }

    /// Converts a protein to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the protein's ID, name, constant flag, and vessel ID
    fn to_record(&self) -> Vec<String> {
        vec![
            self.id.to_string(),
            self.name.to_string(),
            self.constant.to_string(),
            self.vessel_id.clone().unwrap_or("None".to_string()),
        ]
    }
}

impl TableRecord for Reaction {
    /// Returns column headers for reaction tables
    ///
    /// # Returns
    /// * Vector of column names: ID, Name, Reversible, Scheme
    fn columns() -> Vec<String> {
        vec![
            "ID".to_string(),
            "Name".to_string(),
            "Reversible".to_string(),
            "Scheme".to_string(),
        ]
    }

    /// Converts a reaction to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the reaction's ID, name, reversibility, and reaction scheme
    fn to_record(&self) -> Vec<String> {
        vec![
            self.id.to_string(),
            self.name.to_string(),
            self.reversible.to_string(),
            self.reaction_scheme(),
        ]
    }
}

impl TableRecord for Equation {
    /// Returns column headers for equation tables
    ///
    /// # Returns
    /// * Vector of column names: Symbol, Equation, Equation Type
    fn columns() -> Vec<String> {
        vec![
            "Symbol".to_string(),
            "Equation".to_string(),
            "Equation Type".to_string(),
        ]
    }

    /// Converts an equation to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the equation's species ID, equation string,
    ///   equation type, and a list of variables used in the equation
    fn to_record(&self) -> Vec<String> {
        vec![
            self.species_id.to_string(),
            self.equation.to_string(),
            match self.equation_type {
                EquationType::ODE => "ODE".to_string(),
                EquationType::RATE_LAW => "RateLaw".to_string(),
                EquationType::INITIAL_ASSIGNMENT => "InitialAssignment".to_string(),
                EquationType::ASSIGNMENT => "Assignment".to_string(),
            },
            self.variables
                .iter()
                .map(|v| v.id.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        ]
    }
}

impl TableRecord for Parameter {
    /// Returns column headers for parameter tables
    ///
    /// # Returns
    /// * Vector of column names: Symbol, Name, Value, Initial, Lower Bound, Upper Bound
    fn columns() -> Vec<String> {
        vec![
            "Symbol".to_string(),
            "Name".to_string(),
            "Value".to_string(),
            "Initial".to_string(),
            "Lower Bound".to_string(),
            "Upper Bound".to_string(),
        ]
    }

    /// Converts a parameter to a table record
    ///
    /// # Returns
    /// * Vector of strings containing the parameter's symbol, name, current value,
    ///   initial value, and bounds (if available)
    fn to_record(&self) -> Vec<String> {
        vec![
            self.symbol.to_string(),
            self.name.to_string(),
            self.value.map(|v| v.to_string()).unwrap_or("-".to_string()),
            self.initial_value
                .map(|v| v.to_string())
                .unwrap_or("-".to_string()),
            self.lower_bound
                .map(|v| v.to_string())
                .unwrap_or("-".to_string()),
            self.upper_bound
                .map(|v| v.to_string())
                .unwrap_or("-".to_string()),
        ]
    }
}

/// Creates a formatted table displaying measurement data for species
///
/// This function generates a table where:
/// - Each row represents a measurement
/// - Columns include measurement ID, name, and all measured species
/// - Cell values show initial concentrations for each species in each measurement
///
/// # Arguments
/// * `measurements` - Slice of Measurement objects to display
///
/// # Returns
/// * A formatted string containing the table
fn measurement_table(measurements: &[Measurement]) -> String {
    // Extract and sort unique species IDs from all measurements
    let mut measured_species: Vec<String> = measurements
        .iter()
        .flat_map(|m| m.species_data.iter().map(|s| s.species_id.clone()))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    measured_species.sort();

    // Create table builder and add header row
    let mut builder = Builder::default();
    let mut header = vec!["id".to_string(), "name".to_string()];
    header.extend(measured_species.clone());
    builder.push_record(header);

    // Add a row for each measurement
    for measurement in measurements {
        let mut row = vec![measurement.id.to_string(), measurement.name.to_string()];

        // For each species, add its initial value or "-" if not present
        for species_id in &measured_species {
            let value = measurement
                .species_data
                .iter()
                .find(|data| &data.species_id == species_id)
                .map_or("-".to_string(), |data| data.initial.to_string());

            row.push(value);
        }

        builder.push_record(row);
    }

    // Build and style the table
    let mut table = builder.build();
    table.with(Style::rounded());
    table.to_string()
}

impl Reaction {
    /// Converts a reaction to a human-readable reaction scheme string
    ///
    /// Creates a formatted string representation of the reaction in the form:
    /// "reactant1 + reactant2 → product1 + product2" for irreversible reactions, or
    /// "reactant1 + reactant2 ⇄ product1 + product2" for reversible reactions.
    /// Includes stoichiometric coefficients for each species.
    ///
    /// # Returns
    /// * A string representing the reaction scheme
    fn reaction_scheme(&self) -> String {
        let reactants = self
            .species
            .iter()
            .filter(|s| s.stoichiometry < 0.0)
            .map(|s| format!("{} {}", s.stoichiometry.abs(), s.species_id))
            .collect::<Vec<_>>();

        let products = self
            .species
            .iter()
            .filter(|s| s.stoichiometry > 0.0)
            .map(|s| format!("{} {}", s.stoichiometry, s.species_id))
            .collect::<Vec<_>>();

        if self.reversible {
            format!("{} {} {}", reactants.join(" + "), "⇄", products.join(" + "))
        } else {
            format!("{} {} {}", reactants.join(" + "), "→", products.join(" + "))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_table_output() {
        let doc = load_enzmldoc("tests/data/enzmldoc_reaction.json")
            .expect("Failed to load EnzymeML document");

        let expected_table = include_str!("../tests/data/expected_table.txt");
        let table = doc.to_string();
        assert_eq!(table, expected_table);
    }
}
