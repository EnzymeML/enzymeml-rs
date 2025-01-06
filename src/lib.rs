// Will generate the enzyme_ml mod
pub mod enzyme_ml {
    mdmodels_macro::parse_mdmodel!("model.md");
}

pub mod optim {
    pub mod error;
    pub mod initials;
    pub mod metrics;
    pub mod objective;
    pub mod observer;
    pub mod problem;
    pub mod report;
    pub mod runner;
    pub mod system;
}

pub mod legacy {
    pub mod structs;
}

pub mod prelude {
    pub use crate::enzyme_ml::*;
    pub use crate::optim::initials::Initials;
    pub use crate::optim::runner::*;
    pub use crate::simulation::*;
}

pub mod equation;
pub mod simulation {
    pub use crate::simulation::runner::simulate;
    pub use crate::simulation::setup::SimulationSetup;

    pub mod error;
    pub mod result;
    pub mod runner;
    pub mod setup;
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
    #[macro_use]
    pub mod unwrap_list;
    pub mod unit_maps;
}

pub mod tabular {
    mod dataframe;
    pub mod reader;
    pub mod writer;
}

pub mod plot;
