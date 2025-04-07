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
    str::FromStr,
};

use clap::{Parser, Subcommand};
use enzymeml::{
    io::load_enzmldoc,
    llm::{query_llm, PromptInput},
    optim::{Bound, EGOBuilder, InitialGuesses, Optimizer, PSOBuilder, ProblemBuilder},
};
use peroxide::fuga::{self, anyhow, ODEIntegrator, ODEProblem};

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
    Ego {
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

        /// Bounds for the optimization
        #[arg(short, long, value_parser = parse_key_bounds)]
        bound: Vec<(String, (f64, f64))>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,
    },
    /// Particle Swarm Optimization algorithm
    Pso {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 100)]
        max_iters: u64,

        /// Population size for the algorithm
        #[arg(long, default_value_t = 50)]
        pop_size: usize,

        /// Bounds for the optimization
        #[arg(short, long, value_parser = parse_key_bounds)]
        bound: Vec<(String, (f64, f64))>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,
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
            FitAlgorithm::Ego {
                path,
                max_iters,
                output_dir,
                dt,
                bound,
                solver,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .build()
                    .expect("Failed to build problem");

                // Convert EnzymeML document to bounds
                let mut bounds = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");

                // Add CLI bounds
                add_cli_bounds(bound, &mut bounds);

                // Build optimizer
                let optimizer = EGOBuilder::default()
                    .bounds(bounds)
                    .max_iters(*max_iters)
                    .build();

                // Optimize
                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");

                // Write report
                let report_path = output_dir.join("report.json");
                let report_file = File::create(report_path).expect("Failed to create report file");
                serde_json::to_writer_pretty(report_file, &report).expect("Failed to write report");
            }
            FitAlgorithm::Pso {
                path,
                max_iters,
                pop_size,
                output_dir,
                dt,
                bound,
                solver,
            } => {
                // Load EnzymeML document
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");

                // Build problem
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .build()
                    .expect("Failed to build problem");

                // Convert EnzymeML document to bounds
                let mut bounds = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");

                // Add CLI bounds
                add_cli_bounds(bound, &mut bounds);

                // Build optimizer
                let optimizer = PSOBuilder::default()
                    .bounds(bounds)
                    .max_iters(*max_iters)
                    .pop_size(*pop_size)
                    .build();

                // Optimize
                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");

                // Write report
                let report_path = output_dir.join("report.json");
                let report_file = File::create(report_path).expect("Failed to create report file");
                serde_json::to_writer_pretty(report_file, &report).expect("Failed to write report");
            }
        },
    }
}

fn parse_key_bounds(s: &str) -> Result<(String, (f64, f64)), String> {
    // Example k_cat=0.1,0.5
    let parts = s.split('=').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err("Invalid format. Expected key=lower,upper".to_string());
    }
    let key = parts[0].to_string();
    let bounds = parts[1]
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    Ok((key, (bounds[0], bounds[1])))
}

/// Adds the bounds from the CLI to the EnzymeML document bounds
///
/// # Arguments
///
/// * `cli_bounds` - The bounds from the CLI
/// * `doc_bounds` - The bounds from the EnzymeML document
fn add_cli_bounds(cli_bounds: &Vec<(String, (f64, f64))>, doc_bounds: &mut Vec<Bound>) {
    for (key, (lower, upper)) in cli_bounds {
        if let Some(bound) = doc_bounds.iter_mut().find(|b| b.param() == key) {
            bound.set_lower(*lower);
            bound.set_upper(*upper);
            bound.validate().expect("Invalid bounds");
        } else {
            println!("Parameter {} not found", key);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Solvers {
    RK5(fuga::RK5),
    RK4(fuga::RK4),
    RKF45(fuga::RKF45),
    TSIT45(fuga::TSIT45),
    DP45(fuga::DP45),
    BS23(fuga::BS23),
    RALS3(fuga::RALS3),
    RALS4(fuga::RALS4),
}

impl FromStr for Solvers {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rk5" => Ok(Self::RK5(fuga::RK5::default())),
            "rk4" => Ok(Self::RK4(fuga::RK4::default())),
            "rkf45" => Ok(Self::RKF45(fuga::RKF45::default())),
            "tsit45" => Ok(Self::TSIT45(fuga::TSIT45::default())),
            "dp45" => Ok(Self::DP45(fuga::DP45::default())),
            "bs23" => Ok(Self::BS23(fuga::BS23::default())),
            "rals3" => Ok(Self::RALS3(fuga::RALS3::default())),
            "rals4" => Ok(Self::RALS4(fuga::RALS4::default())),
            _ => Err(format!("Invalid solver: {}", s)),
        }
    }
}

impl ODEIntegrator for Solvers {
    fn step<P: ODEProblem>(
        &self,
        problem: &P,
        t: f64,
        y: &mut [f64],
        dt: f64,
    ) -> anyhow::Result<f64> {
        match self {
            Self::RK5(solver) => solver.step(problem, t, y, dt),
            Self::RK4(solver) => solver.step(problem, t, y, dt),
            Self::RKF45(solver) => solver.step(problem, t, y, dt),
            Self::TSIT45(solver) => solver.step(problem, t, y, dt),
            Self::DP45(solver) => solver.step(problem, t, y, dt),
            Self::BS23(solver) => solver.step(problem, t, y, dt),
            Self::RALS3(solver) => solver.step(problem, t, y, dt),
            Self::RALS4(solver) => solver.step(problem, t, y, dt),
        }
    }
}
