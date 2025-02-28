//! EnzymeML Rust Library
//!
//! This library provides functionality for working with EnzymeML documents, including:
//! - Parsing and generating EnzymeML models from markdown
//! - Simulating enzyme kinetics through ODE systems
//! - Optimizing kinetic parameters
//! - Validating EnzymeML documents
//! - Reading/writing tabular data
//! - Plotting results
//!
//! The main modules are organized as follows:

/// Module for generating EnzymeML model code from markdown specifications
pub mod enzyme_ml {
    mdmodels_macro::parse_mdmodel!("model.md");
}

/// Legacy structs and functionality maintained for backwards compatibility
pub mod legacy {
    pub mod structs;
}

/// Commonly used types and functionality re-exported for convenience
pub mod prelude {
    pub use crate::enzyme_ml::*;
    pub use crate::io::*;
    pub use crate::simulation::output::*;
    pub use crate::simulation::result::*;
    pub use crate::simulation::setup::*;
    pub use crate::simulation::system::*;
    pub use crate::simulation::*;
}

/// Core equation handling and manipulation
pub mod equation;

/// Simulation functionality for enzyme kinetics
pub mod simulation {
    pub use crate::simulation::setup::SimulationSetup;

    /// Error types for simulation failures
    pub mod error;
    /// Initial condition handling for simulations
    pub mod init_cond;
    /// Interpolation functionality
    pub mod interpolation;
    /// Output formats for simulation results
    pub mod output;
    /// Simulation result data structures
    pub mod result;
    /// Simulation setup and configuration
    pub mod setup;
    /// Core ODE system implementation
    pub mod system;
}

pub mod optim {
    pub use crate::optim::error::*;
    pub use crate::optim::optimizers::*;
    pub use crate::optim::problem::*;
    pub use crate::optim::transformation::Transformation;
    pub use argmin::core::CostFunction;
    pub use argmin::core::Gradient;
    pub mod error;
    pub mod problem;
    pub mod system;
    pub mod transformation;

    pub mod optimizers {
        pub use crate::optim::optimizers::bfgs::*;
        pub use crate::optim::optimizers::lbfgs::*;
        pub use crate::optim::optimizers::optimizer::*;
        pub use crate::optim::optimizers::pso::*;
        pub mod bfgs;
        pub mod lbfgs;
        pub mod optimizer;
        pub mod pso;
    }
}

pub mod objective {
    pub mod error;
    pub mod loss;
    pub mod objfun;
}

/// Validation of EnzymeML documents and components
pub mod validation {
    /// Validation of equation specifications
    mod equations;
    /// Validation of measurement data
    mod measurements;
    /// Validation of kinetic parameters
    mod parameters;
    /// Validation of reaction specifications
    mod reactions;
    /// Main validation interface
    pub mod validator;
}

/// Procedural and helper macros
pub mod macros {
    /// Macros for reaction specifications
    #[macro_use]
    pub mod reaction_macro;
    /// Macros for data extraction
    #[macro_use]
    pub mod extract_macro;
    /// Macros for unit handling
    #[macro_use]
    pub mod unit_macro;
    /// Utility macros for unwrapping nested data
    #[macro_use]
    pub mod unwrap_list;
    /// Unit conversion and mapping utilities
    pub mod unit_maps;
}

/// Tabular data handling
pub mod tabular {
    /// DataFrame implementation
    mod dataframe;
    /// Reading tabular data from files
    pub mod reader;
    /// Writing data to tabular formats
    pub mod writer;
}

/// Plotting and visualization functionality
pub mod plot;

/// IO functionality
pub mod io;

/// Conversion functionality
pub mod conversion;
