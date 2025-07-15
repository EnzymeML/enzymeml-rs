//! MCMC sample output handling for storing and managing sampling results.
//!
//! This module provides traits and implementations for storing MCMC samples from multiple chains.
//! The primary implementation uses Polars DataFrames for efficient storage and manipulation of
//! numerical data.

use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use polars::{
    frame::DataFrame,
    io::{SerReader, SerWriter},
    prelude::{CsvReadOptions, CsvWriter, NamedFrom},
    series::Series,
};

use crate::mcmc::error::MCMCError;

/// Trait for handling MCMC sample output across multiple chains.
///
/// This trait defines the interface for storing and retrieving MCMC samples from multiple
/// parallel chains. Implementations should provide efficient storage and retrieval of
/// numerical samples organized by parameter names and chain identifiers.
pub trait SampleOutput {
    /// Add a single draw (sample) to a specific chain.
    ///
    /// # Arguments
    ///
    /// * `samples` - A slice of f64 values representing parameter samples for one iteration
    /// * `chain_id` - String identifier for the chain to add the sample to
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success, or an error if the operation fails
    ///
    /// # Errors
    ///
    /// * `MCMCError::SeriesError` - If the number of samples doesn't match the number of parameters
    /// * `MCMCError::DataFrameError` - If there's an issue with DataFrame operations
    fn add_draw(&mut self, samples: &[f64], chain_id: &str) -> Result<(), MCMCError>;

    /// Add a new empty chain with the given identifier.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - String identifier for the new chain
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success, or an error if the operation fails
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If there's an issue creating the DataFrame for the chain
    fn add_chain(&mut self, chain_id: &str) -> Result<(), MCMCError>;

    /// Add multiple new empty chains with the given identifiers.
    ///
    /// # Arguments
    ///
    /// * `chain_ids` - Slice of string identifiers for the new chains
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success, or an error if any chain creation fails
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If there's an issue creating DataFrames for any chain
    fn add_chains(&mut self, chain_ids: &[&str]) -> Result<(), MCMCError>;

    /// Get a reference to the DataFrame for a specific chain.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - String identifier for the chain to retrieve
    ///
    /// # Returns
    ///
    /// `Option<&DataFrame>` - Some(&DataFrame) if the chain exists, None otherwise
    fn get_chain(&self, chain_id: &str) -> Result<DataFrame, MCMCError>;

    /// Get references to all chain DataFrames.
    ///
    /// # Returns
    ///
    /// `Vec<&DataFrame>` - Vector of references to all stored chain DataFrames
    fn get_chains(&self) -> Result<Vec<DataFrame>, MCMCError>;
}

/// DataFrame-based implementation of MCMC sample output storage.
///
/// This implementation uses Polars DataFrames to store MCMC samples efficiently.
/// Each chain is stored as a separate DataFrame with columns corresponding to
/// model parameters and rows corresponding to individual samples.
///
/// # Fields
///
/// * `parameters` - Vector of parameter names that define the column structure
/// * `chains` - HashMap mapping chain IDs to their corresponding DataFrames
#[derive(Debug, Clone)]
pub struct DataFrameOutput {
    /// Names of the model parameters that will be sampled
    parameters: Vec<String>,
    /// Storage for chain data, keyed by chain identifier
    chains: HashMap<String, DataFrame>,
}

impl DataFrameOutput {
    /// Create a new DataFrameOutput with the specified parameter names.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Vector of parameter names that can be converted to strings
    ///
    /// # Returns
    ///
    /// `Result<Self, MCMCError>` - New DataFrameOutput instance on success
    pub fn new(parameters: Vec<impl Into<String>>) -> Result<Self, MCMCError> {
        let parameters = parameters.into_iter().map(|p| p.into()).collect();
        Ok(Self {
            parameters,
            chains: HashMap::new(),
        })
    }

    /// Export all chains to CSV files in the specified directory.
    ///
    /// Creates one CSV file per chain in the given directory. Each file is named
    /// using the chain ID with a .csv extension.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory path where CSV files will be written
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success
    ///
    /// # Errors
    ///
    /// * `MCMCError::IoError` - If directory creation or file writing fails
    /// * `MCMCError::DataFrameError` - If CSV serialization fails
    pub fn to_csv(&mut self, dir: impl Into<PathBuf>) -> Result<(), MCMCError> {
        let dir = dir.into();
        for (chain_id, chain) in self.chains.iter_mut() {
            let path = Path::join(&dir, format!("{chain_id}.csv"));
            let mut file = std::fs::File::create(&path)?;
            CsvWriter::new(&mut file).finish(chain)?;
        }

        Ok(())
    }
}

impl SampleOutput for DataFrameOutput {
    /// Add a single draw to the specified chain.
    ///
    /// This method creates a new DataFrame row from the provided samples and extends
    /// the existing chain DataFrame. It's optimized for performance by pre-allocating
    /// the series vector.
    ///
    /// # Arguments
    ///
    /// * `draw` - Slice of f64 values, one for each parameter
    /// * `chain_id` - Identifier of the chain to add the sample to
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success
    ///
    /// # Errors
    ///
    /// * `MCMCError::SeriesError` - If the draw length doesn't match parameter count
    /// * `MCMCError::DataFrameError` - If DataFrame operations fail
    ///
    /// # Panics
    ///
    /// Panics if the chain_id doesn't exist (chain must be added first with `add_chain`)
    #[inline(always)]
    fn add_draw(&mut self, draw: &[f64], chain_id: &str) -> Result<(), MCMCError> {
        let chain = self.chains.get_mut(chain_id).unwrap();

        // Pre-allocate series vector with known capacity
        let mut series_vec = Vec::with_capacity(self.parameters.len());

        for (i, parameter) in self.parameters.iter().enumerate() {
            let value = draw.get(i).ok_or_else(|| {
                MCMCError::SeriesError(format!("Failed to get sample for parameter {parameter}"))
            })?;

            series_vec.push(Series::new(parameter, &[*value]));
        }

        let df = DataFrame::new(series_vec)?;
        chain.extend(&df)?;

        Ok(())
    }

    /// Add a new empty chain with the specified identifier.
    ///
    /// Creates a new DataFrame with empty columns for each parameter and stores it
    /// in the chains HashMap with the given chain_id as the key.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - Unique identifier for the new chain
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If DataFrame creation fails
    fn add_chain(&mut self, chain_id: &str) -> Result<(), MCMCError> {
        let parameters = self
            .parameters
            .iter()
            .map(|p| Series::new(p, Vec::<f64>::new()))
            .collect::<Vec<_>>();
        let df = DataFrame::new(parameters)?;
        self.chains.insert(chain_id.to_string(), df);
        Ok(())
    }

    /// Add multiple new empty chains.
    ///
    /// Convenience method for adding multiple chains at once. Calls `add_chain`
    /// for each provided chain_id.
    ///
    /// # Arguments
    ///
    /// * `chain_ids` - Slice of chain identifiers to create
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) if all chains are created successfully
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If any DataFrame creation fails
    fn add_chains(&mut self, chain_ids: &[&str]) -> Result<(), MCMCError> {
        for chain_id in chain_ids {
            self.add_chain(chain_id)?;
        }
        Ok(())
    }

    /// Get a reference to the DataFrame for the specified chain.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - Identifier of the chain to retrieve
    ///
    /// # Returns
    ///
    /// `Result<DataFrame, MCMCError>` - Cloned DataFrame for the specified chain
    ///
    /// # Errors
    ///
    /// * `MCMCError::ChainNotFound` - If the specified chain ID doesn't exist
    fn get_chain(&self, chain_id: &str) -> Result<DataFrame, MCMCError> {
        Ok(self
            .chains
            .get(chain_id)
            .ok_or(MCMCError::ChainNotFound(chain_id.to_string()))?
            .clone())
    }

    /// Get references to all stored chain DataFrames.
    ///
    /// # Returns
    ///
    /// `Result<Vec<DataFrame>, MCMCError>` - Vector containing cloned DataFrames for all chains
    fn get_chains(&self) -> Result<Vec<DataFrame>, MCMCError> {
        Ok(self.chains.values().cloned().collect())
    }
}

/// CSV file-based implementation of MCMC sample output storage.
///
/// This implementation writes MCMC samples directly to CSV files on disk, with one file
/// per chain. Each CSV file has a header row with parameter names, followed by rows
/// containing the sample values for each iteration. This approach is memory-efficient
/// for large sample sets but requires disk I/O for each operation.
///
/// The CSV files are stored in a specified directory with filenames following the
/// pattern "{chain_id}.csv". Files are kept open during sampling for efficiency
/// and automatically flushed when the struct is dropped.
///
/// # Fields
///
/// * `dir` - Directory path where CSV files will be stored
/// * `chains` - HashMap mapping chain IDs to their corresponding open file handles
/// * `parameters` - Vector of parameter names that define the column structure
pub struct CSVOutput {
    /// Directory path where CSV files are stored
    dir: PathBuf,
    /// Open file handles for each chain, keyed by chain identifier
    chains: HashMap<String, File>,
    /// Names of the model parameters that will be sampled
    parameters: Vec<String>,
}

impl CSVOutput {
    /// Create a new CSVOutput instance with the specified directory and parameter names.
    ///
    /// Creates the output directory structure and initializes the parameter list.
    /// The directory will be created if it doesn't exist when the first chain is added.
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the directory where CSV files will be stored
    /// * `parameters` - Vector of parameter names that can be converted to strings
    ///
    /// # Returns
    ///
    /// `Result<Self, MCMCError>` - New CSVOutput instance ready for use
    ///
    /// # Errors
    ///
    /// * `MCMCError::IoError` - If directory creation fails
    pub fn new(
        dir: impl Into<PathBuf>,
        parameters: Vec<impl Into<String>>,
    ) -> Result<Self, MCMCError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(MCMCError::IoError)?;
        Ok(Self {
            dir,
            chains: HashMap::new(),
            parameters: parameters.into_iter().map(|p| p.into()).collect(),
        })
    }
}

impl SampleOutput for CSVOutput {
    /// Add a new empty chain by creating a CSV file with headers.
    ///
    /// Creates a new CSV file in the output directory with the specified chain ID.
    /// The file is immediately written with a header row containing all parameter names.
    /// The file handle is kept open for efficient subsequent writes.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - String identifier for the new chain, used as the filename base
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success
    ///
    /// # Errors
    ///
    /// * `MCMCError::IoError` - If file creation fails or directory is not writable
    /// * `MCMCError::IoError` - If writing the header row fails
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be created (will be replaced with proper error handling)
    fn add_chain(&mut self, chain_id: &str) -> Result<(), MCMCError> {
        let path = Path::join(&self.dir, format!("{chain_id}.csv"));
        let mut file = File::create(path).unwrap();

        // Add header
        let header = self.parameters.join(",");
        writeln!(file, "{header}")?;
        self.chains.insert(chain_id.to_string(), file);

        Ok(())
    }

    /// Add a single draw (sample) to the specified chain's CSV file.
    ///
    /// Appends a new row to the CSV file corresponding to the given chain ID.
    /// The sample values are converted to strings and joined with commas.
    /// The row is immediately written to disk and flushed.
    ///
    /// # Arguments
    ///
    /// * `draw` - Slice of f64 values representing parameter samples for one iteration
    /// * `chain_id` - String identifier for the chain to add the sample to
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) on success
    ///
    /// # Errors
    ///
    /// * `MCMCError::IoError` - If writing to the file fails
    ///
    /// # Panics
    ///
    /// Panics if the chain_id doesn't exist (will be replaced with proper error handling)
    fn add_draw(&mut self, draw: &[f64], chain_id: &str) -> Result<(), MCMCError> {
        let file = self.chains.get_mut(chain_id).unwrap();
        let line = draw
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Add multiple new empty chains by creating CSV files for each.
    ///
    /// Convenience method that calls `add_chain` for each provided chain ID.
    /// All chains will have the same parameter structure defined at construction.
    ///
    /// # Arguments
    ///
    /// * `chain_ids` - Slice of string identifiers for the chains to create
    ///
    /// # Returns
    ///
    /// `Result<(), MCMCError>` - Ok(()) if all chains are created successfully
    ///
    /// # Errors
    ///
    /// * `MCMCError::IoError` - If any file creation or header writing fails
    ///
    /// # Behavior
    ///
    /// If any chain creation fails, the method returns immediately without creating
    /// the remaining chains. Previously created chains in the same call will remain.
    fn add_chains(&mut self, chain_ids: &[&str]) -> Result<(), MCMCError> {
        for chain_id in chain_ids {
            self.add_chain(chain_id)?;
        }
        Ok(())
    }

    /// Read and return the DataFrame for the specified chain from its CSV file.
    ///
    /// Loads the entire CSV file into memory as a Polars DataFrame. The file must
    /// exist and be readable. The DataFrame will have columns matching the parameter
    /// names defined at construction.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - String identifier for the chain to retrieve
    ///
    /// # Returns
    ///
    /// `Result<DataFrame, MCMCError>` - DataFrame containing all samples for the chain
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If the CSV file cannot be parsed or read
    /// * File I/O errors are converted to DataFrameError via Polars error handling
    ///
    /// # Panics
    ///
    /// Panics if the CSV reader configuration fails (will be replaced with proper error handling)
    fn get_chain(&self, chain_id: &str) -> Result<DataFrame, MCMCError> {
        let path = Path::join(&self.dir, format!("{chain_id}.csv"));
        CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path))
            .unwrap()
            .finish()
            .map_err(MCMCError::DataFrameError)
    }

    /// Read and return DataFrames for all chains from their CSV files.
    ///
    /// Loads all chain CSV files in the output directory and returns them as a vector
    /// of DataFrames. The order of DataFrames in the vector corresponds to the iteration
    /// order of the internal HashMap keys.
    ///
    /// # Returns
    ///
    /// `Result<Vec<DataFrame>, MCMCError>` - Vector containing DataFrames for all chains
    ///
    /// # Errors
    ///
    /// * `MCMCError::DataFrameError` - If any CSV file cannot be parsed or read
    ///
    /// # Behavior
    ///
    /// If any individual chain file fails to load, the method returns an error immediately
    /// without loading the remaining chains.
    fn get_chains(&self) -> Result<Vec<DataFrame>, MCMCError> {
        let mut chains = Vec::new();
        for chain_id in self.chains.keys() {
            chains.push(self.get_chain(chain_id)?);
        }
        Ok(chains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test creating a chain and verifying its initial empty state.
    #[test]
    fn test_df_chain_output() {
        let mut output = DataFrameOutput::new(vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        let chain = output.get_chain("0").unwrap();
        assert_eq!(chain.column("a").unwrap().len(), 0);
        assert_eq!(chain.column("b").unwrap().len(), 0);
        assert_eq!(chain.column("c").unwrap().len(), 0);
    }

    /// Test adding multiple chains and verifying they're all empty initially.
    #[test]
    fn test_df_add_chains() {
        let mut output = DataFrameOutput::new(vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_chain("1").unwrap();
        let chain1 = output.get_chain("0").unwrap();
        let chain2 = output.get_chain("1").unwrap();
        assert_eq!(chain1.column("a").unwrap().len(), 0);
        assert_eq!(chain2.column("a").unwrap().len(), 0);
    }

    /// Test adding a single draw to a chain.
    #[test]
    fn test_df_add_draw() {
        let mut output = DataFrameOutput::new(vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_draw(&[1.0, 2.0, 3.0], "0").unwrap();
        let chain = output.get_chain("0").unwrap();
        assert_eq!(chain.column("a").unwrap().len(), 1);
    }

    /// Test adding multiple draws to verify chain length increases correctly.
    #[test]
    fn test_df_add_draw_multiple() {
        let mut output = DataFrameOutput::new(vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_draw(&[1.0, 2.0, 3.0], "0").unwrap();
        output.add_draw(&[4.0, 5.0, 6.0], "0").unwrap();
        let chain = output.get_chain("0").unwrap();
        assert_eq!(chain.column("a").unwrap().len(), 2);
        assert_eq!(chain.column("b").unwrap().len(), 2);
        assert_eq!(chain.column("c").unwrap().len(), 2);
    }

    /// Test creating a CSV output and verifying basic functionality.
    #[test]
    fn test_csv_chain_output() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut output = CSVOutput::new(temp_dir.path(), vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();

        // Check that the CSV file was created
        let csv_path = temp_dir.path().join("0.csv");
        assert!(csv_path.exists());

        // Read the file and check the header
        let content = std::fs::read_to_string(csv_path).unwrap();
        assert!(content.starts_with("a,b,c\n"));
    }

    /// Test adding multiple chains to CSV output.
    #[test]
    fn test_csv_add_chains() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut output = CSVOutput::new(temp_dir.path(), vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_chain("1").unwrap();

        // Check that both CSV files were created
        let csv_path_0 = temp_dir.path().join("0.csv");
        let csv_path_1 = temp_dir.path().join("1.csv");
        assert!(csv_path_0.exists());
        assert!(csv_path_1.exists());

        // Verify headers in both files
        let content_0 = std::fs::read_to_string(csv_path_0).unwrap();
        let content_1 = std::fs::read_to_string(csv_path_1).unwrap();
        assert!(content_0.starts_with("a,b,c\n"));
        assert!(content_1.starts_with("a,b,c\n"));
    }

    /// Test adding a single draw to CSV output.
    #[test]
    fn test_csv_add_draw() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut output = CSVOutput::new(temp_dir.path(), vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_draw(&[1.0, 2.0, 3.0], "0").unwrap();

        // Drop the output to ensure file is flushed
        drop(output);

        let csv_path = temp_dir.path().join("0.csv");
        let content = std::fs::read_to_string(csv_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
        assert_eq!(lines[0], "a,b,c");
        assert_eq!(lines[1], "1,2,3");
    }

    /// Test adding multiple draws to CSV output.
    #[test]
    fn test_csv_add_draw_multiple() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut output = CSVOutput::new(temp_dir.path(), vec!["a", "b", "c"]).unwrap();
        output.add_chain("0").unwrap();
        output.add_draw(&[1.0, 2.0, 3.0], "0").unwrap();
        output.add_draw(&[4.0, 5.0, 6.0], "0").unwrap();
        output.add_draw(&[7.0, 8.0, 9.0], "0").unwrap();

        // Drop the output to ensure file is flushed
        drop(output);

        let csv_path = temp_dir.path().join("0.csv");
        let content = std::fs::read_to_string(csv_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
        assert_eq!(lines[0], "a,b,c");
        assert_eq!(lines[1], "1,2,3");
        assert_eq!(lines[2], "4,5,6");
        assert_eq!(lines[3], "7,8,9");
    }

    /// Test CSV output with multiple chains and draws.
    #[test]
    fn test_csv_multiple_chains_draws() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut output = CSVOutput::new(temp_dir.path(), vec!["x", "y"]).unwrap();
        output.add_chain("chain1").unwrap();
        output.add_chain("chain2").unwrap();

        output.add_draw(&[1.1, 2.2], "chain1").unwrap();
        output.add_draw(&[3.3, 4.4], "chain2").unwrap();
        output.add_draw(&[5.5, 6.6], "chain1").unwrap();

        // Drop the output to ensure files are flushed
        drop(output);

        // Check chain1 file
        let csv_path_1 = temp_dir.path().join("chain1.csv");
        let content_1 = std::fs::read_to_string(csv_path_1).unwrap();
        let lines_1: Vec<&str> = content_1.trim().split('\n').collect();
        assert_eq!(lines_1.len(), 3); // header + 2 data rows
        assert_eq!(lines_1[0], "x,y");
        assert_eq!(lines_1[1], "1.1,2.2");
        assert_eq!(lines_1[2], "5.5,6.6");

        // Check chain2 file
        let csv_path_2 = temp_dir.path().join("chain2.csv");
        let content_2 = std::fs::read_to_string(csv_path_2).unwrap();
        let lines_2: Vec<&str> = content_2.trim().split('\n').collect();
        assert_eq!(lines_2.len(), 2); // header + 1 data row
        assert_eq!(lines_2[0], "x,y");
        assert_eq!(lines_2[1], "3.3,4.4");
    }
}
