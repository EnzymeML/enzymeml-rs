//! EnzymeML Rust Library
//!
//! This library provides functionality for working with EnzymeML documents, including:
//!
//! - Simulating enzyme kinetics through ODE systems
//! - Optimizing kinetic parameters
//! - Bayesian parameter estimation
//! - Profile likelihood computation
//! - Identifiability analysis
//! - Language model integration
//! - SBML format reading and writing
//! - Tabular data handling
//! - Plotting and visualization
//! - Validating EnzymeML documents

#![warn(unused_imports)]

/// EnzymeML version management and schema definitions
pub mod versions {
    pub use crate::versions::v2 as latest;
    /// EnzymeML version 2 schema and data structures
    pub mod v2;
}

/// Utility functions
pub mod utils;

/// Commonly used types and functionality re-exported for convenience
pub mod prelude {
    pub use crate::io::*;
    pub use crate::versions::latest::*;

    #[cfg(feature = "simulation")]
    pub use crate::simulation::init_cond::*;
    #[cfg(feature = "simulation")]
    pub use crate::simulation::output::*;
    #[cfg(feature = "simulation")]
    pub use crate::simulation::result::*;
    #[cfg(feature = "simulation")]
    pub use crate::simulation::setup::*;
    #[cfg(feature = "simulation")]
    pub use crate::simulation::system::*;

    #[cfg(feature = "optimization")]
    pub use crate::objective::loss::*;
    #[cfg(feature = "optimization")]
    pub use crate::objective::objfun::*;
    #[cfg(feature = "optimization")]
    pub use crate::optim::*;
}

/// Core equation handling and manipulation
pub mod equation;

/// Simulation functionality for enzyme kinetics
#[cfg(feature = "simulation")]
pub mod simulation {
    pub use crate::simulation::setup::SimulationSetup;
    pub use peroxide::fuga::{
        ImplicitSolver, BS23, DP45, GL4, RALS3, RALS4, RK4, RK5, RKF45, TSIT45,
    };

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
    /// Derivation of the stoichiometry matrix
    pub mod stoich;
    /// Core ODE system implementation
    pub mod system;
    /// Macros for creating JIT functions
    #[macro_use]
    pub mod macros;
}

/// Optimization algorithms and parameter estimation for enzyme kinetics
#[cfg(feature = "optimization")]
pub mod optim {
    pub use crate::optim::bound::*;
    pub use crate::optim::error::*;
    pub use crate::optim::optimizers::*;
    pub use crate::optim::problem::*;
    pub use crate::optim::transformation::*;
    pub use argmin::core::CostFunction;
    pub use argmin::core::Gradient;
    use argmin_math as _;
    pub use peroxide::fuga::{
        ImplicitSolver, BS23, DP45, GL4, RALS3, RALS4, RK4, RK5, RKF45, TSIT45,
    };

    /// Parameter bounds and constraints for optimization
    pub mod bound;
    /// Error types for optimization failures
    pub mod error;
    /// Performance metrics and convergence criteria
    pub mod metrics;
    /// Observer patterns for monitoring optimization progress
    pub mod observer;
    /// Problem formulation and setup for optimization tasks
    pub mod problem;
    /// Result reporting and analysis for optimization runs
    pub mod report;
    /// System dynamics and equation handling for optimization
    pub mod system;
    /// Parameter transformations for optimization algorithms
    pub mod transformation;

    /// Collection of optimization algorithms and solvers
    pub mod optimizers {
        pub use crate::optim::optimizers::bfgs::*;
        pub use crate::optim::optimizers::ego::*;
        pub use crate::optim::optimizers::lbfgs::*;
        pub use crate::optim::optimizers::optimizer::*;
        pub use crate::optim::optimizers::pso::*;
        pub use crate::optim::optimizers::srtrust::*;
        /// Broyden-Fletcher-Goldfarb-Shanno optimization algorithm
        pub mod bfgs;
        /// Efficient Global Optimization using Gaussian processes
        pub mod ego;
        /// Limited-memory BFGS optimization algorithm
        pub mod lbfgs;
        /// Common optimizer traits and interfaces
        pub mod optimizer;
        /// Particle Swarm Optimization algorithm
        pub mod pso;
        /// Stochastic Rust Trust region optimization algorithm
        pub mod srtrust;
        /// Utility functions for optimization algorithms
        pub(crate) mod utils;
    }
}

/// Markov Chain Monte Carlo sampling for Bayesian parameter estimation
#[cfg(feature = "optimization")]
pub mod mcmc {
    /// Diagnostic tools for MCMC chain quality assessment
    pub mod diagnostics;
    /// Error types for MCMC sampling failures
    pub mod error;
    /// Likelihood function definitions for Bayesian inference
    pub mod likelihood;
    /// Output formatting and storage for MCMC results
    pub mod output;
    /// Prior distribution specifications for Bayesian analysis
    pub mod priors;
    /// Problem setup and configuration for MCMC sampling
    pub mod problem;
}

/// Objective function definitions and loss calculations for optimization
#[cfg(feature = "optimization")]
pub mod objective {
    pub use crate::objective::loss::*;
    pub use crate::objective::objfun::*;

    /// Error types for objective function evaluation failures
    pub mod error;
    /// Loss function definitions and implementations
    pub mod loss;
    /// Objective function construction and evaluation
    pub mod objfun;
}

/// SBML format reading and writing functionality for EnzymeML documents
#[cfg(feature = "sbml")]
pub mod sbml {
    pub use crate::sbml::reader as sbml_reader;
    pub use crate::sbml::version::EnzymeMLVersion;
    pub use crate::sbml::writer as sbml_writer;

    /// Annotation handling for SBML metadata
    pub(crate) mod annotations;
    /// Error types for SBML processing failures
    pub mod error;
    /// SBML document reading and parsing functionality
    pub mod reader;
    /// Species type definitions and handling
    pub(super) mod speciestype;
    /// Unit handling and conversion for SBML
    pub(super) mod units;
    /// Utility functions for SBML processing
    pub(super) mod utils;
    /// Version detection and handling for EnzymeML formats
    pub mod version;
    /// SBML document writing and serialization functionality
    pub mod writer;
    /// EnzymeML version 1 support and schema definitions
    pub(super) mod v1 {
        pub(crate) use crate::sbml::v1::schema::*;
        /// Data extraction utilities for v1 format
        pub(crate) mod extract;
        /// Schema definitions for EnzymeML v1
        pub(super) mod schema;
        /// Serialization utilities for v1 format
        pub mod serializer;
    }
    /// EnzymeML version 2 support and schema definitions
    pub(super) mod v2 {
        pub(crate) use crate::sbml::v2::schema::*;
        /// Data extraction utilities for v2 format
        pub(super) mod extract;
        /// Schema definitions for EnzymeML v2
        pub(super) mod schema;
        /// Serialization utilities for v2 format
        pub mod serializer;
    }
}

/// Parameter identifiability analysis and profile likelihood computation
#[cfg(feature = "optimization")]
pub mod identifiability {
    pub use crate::identifiability::egoprofile::*;
    pub use crate::identifiability::parameter::*;
    pub use crate::identifiability::profile::*;

    /// Efficient Global Optimization for profile likelihood computation
    pub mod egoprofile;
    /// Grid-based parameter space exploration
    pub mod grid;
    /// Parameter definition and handling for identifiability analysis
    pub mod parameter;
    /// Profile likelihood computation and analysis
    pub mod profile;
    /// Result storage and analysis for identifiability studies
    pub mod results;
    /// Utility functions for identifiability analysis
    pub mod utils;
}

#[cfg(feature = "llm")]
/// Language model integration for automated EnzymeML document generation and analysis
pub mod llm;

/// Validation of EnzymeML documents and components
pub mod validation {
    pub use crate::validation::schema::*;
    /// Main consistency interface
    pub mod consistency;
    /// Validation of equation specifications
    mod equations;
    /// Validation of measurement data
    mod measurements;
    /// Validation of kinetic parameters
    mod parameters;
    /// Validation of reaction specifications
    mod reactions;
    /// Main schema validation interface
    pub mod schema;
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

/// Language bindings and interfaces for other programming environments
pub mod bindings {
    /// WebAssembly bindings for browser and Node.js environments
    #[cfg(feature = "wasm")]
    pub mod wasm;
}

/// Tabular data handling
#[cfg(feature = "tabular")]
pub mod tabular {
    /// DataFrame implementation
    mod dataframe;
    /// Reading tabular data from files
    pub mod reader;
    /// Writing data to tabular formats
    pub mod writer;
}

/// Plotting and visualization functionality
pub mod plotting;

/// IO functionality
pub mod io;

/// [`Display`] implementation for the EnzymeML library
pub mod info;

/// ODE functionality used to derive the system of ODEs from the EnzymeML reactions
pub mod system;

/// Conversion functionality
pub mod conversion;
