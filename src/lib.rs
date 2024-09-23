use mdmodels_macro::parse_mdmodel;

// Will generate the enzyme_ml mod
parse_mdmodel!("model.md");

pub mod prelude {
    pub use crate::enzyme_ml::*;
    pub use crate::simulation::*;
}

pub mod equation;
pub mod simulation {
    pub use crate::simulation::runner::simulate;
    pub use crate::simulation::runner::{SimulationSetup, SimulationSetupBuilder};

    pub mod result;
    pub mod runner;
    pub mod system;
}

pub mod validation {
    mod equations;
    mod measurements;
    mod parameters;
    mod reactions;
    pub mod validator;
}

pub mod macros {
    #[macro_use]
    pub mod reaction_macro;
    #[macro_use]
    pub mod extract_macro;
    #[macro_use]
    pub mod unit_macro;
    pub mod unit_maps;
}

pub mod tabular {
    mod dataframe;
    pub mod reader;
    pub mod writer;
}
