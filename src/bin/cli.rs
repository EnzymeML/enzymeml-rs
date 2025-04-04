//! Command-line interface for the EnzymeML library
//!
//! This binary provides a CLI interface to interact with EnzymeML documents, including:
//! - Extracting information from natural language descriptions using LLMs
//! - Fitting kinetic parameters using various optimization algorithms
//!
//! # Usage
//!
//! ```bash
//! # Extract information from text using LLM
//! enzymeml extract --prompt input.txt --output result.json
//!
//! # Fit parameters using EGO algorithm
//! enzymeml fit ego --path model.xml --max-iters 100
//!
//! # Fit parameters using PSO algorithm  
//! enzymeml fit pso --path model.xml --pop-size 50
//! ```

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use enzymeml::{
    io::load_enzmldoc,
    llm::{query_llm, PromptInput},
    optim::{EGOBuilder, InitialGuesses, Optimizer, PSOBuilder, ProblemBuilder},
};
use peroxide::fuga::RK5;

/// Main CLI configuration struct
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available CLI commands
#[derive(Subcommand)]
enum Commands {
    /// Fit a model using optimization algorithms
    Fit {
        #[command(subcommand)]
        algorithm: FitAlgorithm,
    },
    /// Extract information from an EnzymeML document
    Extract {
        /// Path to the file containing the EnzymeML document
        #[arg(
            short,
            long,
            help = "Path to the file containing the text to extract information from"
        )]
        prompt: String,

        /// Path to the file containing the natural language description
        #[arg(short, long, help = "Path to the file containing the system prompt")]
        system_prompt: Option<String>,

        /// Path to the output file
        #[arg(short, long, help = "Path to save the response to.")]
        output: Option<PathBuf>,

        /// LLM model to use
        #[arg(short, long, default_value = "gpt-4o", help = "LLM model to use")]
        llm_model: String,

        /// API key for the LLM service
        #[arg(short, long, help = "API key for the LLM service")]
        api_key: Option<String>,

        /// Path to the dataset to extend, if any
        #[arg(short, long, help = "Path to the dataset to extend, if any")]
        dataset: Option<PathBuf>,
    },
}

/// Available optimization algorithms
#[derive(Subcommand)]
enum FitAlgorithm {
    /// Efficient Global Optimization algorithm
    EGO {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 100)]
        max_iters: usize,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,
    },
    /// Particle Swarm Optimization algorithm
    PSO {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 100)]
        max_iters: u64,

        /// Population size for the algorithm
        #[arg(long, default_value_t = 50)]
        pop_size: usize,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,
    },
}

/// Main entry point for the CLI application
pub fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Extract {
            prompt,
            system_prompt,
            output,
            llm_model,
            api_key,
            dataset: _,
        } => {
            let prompt = if Path::new(prompt).exists() {
                PromptInput::File(PathBuf::from(prompt))
            } else {
                PromptInput::String(prompt.to_string())
            };

            let system_prompt = if let Some(system_prompt) = system_prompt {
                if Path::new(system_prompt).exists() {
                    Some(PromptInput::File(PathBuf::from(system_prompt)))
                } else {
                    Some(PromptInput::String(system_prompt.to_string()))
                }
            } else {
                None
            };

            let response = query_llm(
                prompt,
                system_prompt,
                Some(llm_model.to_string()),
                api_key.as_ref().map(|key| key.to_string()),
            )
            .expect("Failed to query LLM");

            let json_response = serde_json::to_string_pretty(&response)
                .expect("Failed to serialize response to JSON");

            match output {
                Some(path) => {
                    serde_json::to_writer_pretty(
                        File::create(path).expect("Failed to create response file"),
                        &response,
                    )
                    .expect("Failed to write response");
                }
                None => println!("{}", json_response),
            }
        }
        Commands::Fit { algorithm } => match algorithm {
            FitAlgorithm::EGO {
                path,
                max_iters,
                output_dir,
                dt,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let problem = ProblemBuilder::new(&enzmldoc, RK5::default())
                    .dt(*dt)
                    .build()
                    .expect("Failed to build problem");
                let bounds = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");
                let optimizer = EGOBuilder::default()
                    .bounds(bounds)
                    .max_iters(*max_iters)
                    .build();

                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");
                let report_path = output_dir.join("report.json");
                let report_file = File::create(report_path).expect("Failed to create report file");
                serde_json::to_writer_pretty(report_file, &report).expect("Failed to write report");
            }
            FitAlgorithm::PSO {
                path,
                max_iters,
                pop_size,
                output_dir,
                dt,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let problem = ProblemBuilder::new(&enzmldoc, RK5::default())
                    .dt(*dt)
                    .build()
                    .expect("Failed to build problem");
                let bounds = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");
                let optimizer = PSOBuilder::default()
                    .bounds(bounds)
                    .max_iters(*max_iters)
                    .pop_size(*pop_size)
                    .build();

                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");
                let report_path = output_dir.join("report.json");
                let report_file = File::create(report_path).expect("Failed to create report file");
                serde_json::to_writer_pretty(report_file, &report).expect("Failed to write report");
            }
        },
    }
}
