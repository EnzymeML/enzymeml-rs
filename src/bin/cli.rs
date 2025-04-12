//! Command-line interface for the EnzymeML library
//!
//! This binary provides a CLI interface to interact with EnzymeML documents, including:
//! - Extracting information from natural language descriptions using LLMs
//! - Fitting kinetic parameters using various optimization algorithms
//! - Converting EnzymeML documents to different formats
//! - Validating EnzymeML documents for schema compliance and consistency
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
//!
//! # Convert EnzymeML document to Excel format
//! enzymeml convert --input model.xml --target xlsx --output model.xlsx
//!
//! # Validate an EnzymeML document
//! enzymeml validate model.xml
//! ```
//!
//! # Commands
//!
//! - `convert`: Convert EnzymeML documents to different formats (currently supports XLSX)
//! - `validate`: Check EnzymeML documents for schema compliance and consistency
//! - `fit`: Fit kinetic parameters using various optimization algorithms
//!   - `ego`: Efficient Global Optimization algorithm
//!   - `pso`: Particle Swarm Optimization algorithm
//!   - `lbfgs`: Limited-memory BFGS algorithm
//!   - `bfgs`: Broyden–Fletcher–Goldfarb–Shanno algorithm
//! - `extract`: Extract structured information from natural language using LLMs
//!
//! # Parameter Optimization
//!
//! The CLI supports multiple optimization algorithms for parameter fitting:
//!
//! - **EGO**: Efficient Global Optimization, a surrogate-based algorithm suitable for expensive objective functions
//! - **PSO**: Particle Swarm Optimization, a population-based stochastic algorithm
//! - **LBFGS/BFGS**: Gradient-based optimization algorithms for smooth objective functions
//!
//! # Solvers
//!
//! The following ODE solvers are available for simulation:
//!
//! - `rk5`: 5th order Runge-Kutta method
//! - `rk4`: 4th order Runge-Kutta method
//! - `rkf45`: Runge-Kutta-Fehlberg method
//! - `tsit45`: Tsitouras 5/4 method
//! - `dp45`: Dormand-Prince 5/4 method
//! - `bs23`: Bogacki-Shampine 3/2 method
//! - `rals3`: 3rd order Rosenbrock-Armero-Laso-Simo method
//! - `rals4`: 4th order Rosenbrock-Armero-Laso-Simo method

use std::{
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use enzymeml::{
    io::{load_enzmldoc, save_enzmldoc},
    llm::{query_llm, PromptInput},
    optim::{
        BFGSBuilder, Bound, EGOBuilder, InitialGuesses, LBFGSBuilder, Optimizer, PSOBuilder,
        ProblemBuilder, SR1TrustRegionBuilder, SubProblem, Transformation,
    },
    prelude::{EnzymeMLDocument, LossFunction},
    validation::{consistency, schema},
};

use peroxide::fuga::{self, anyhow, ODEIntegrator, ODEProblem};

#[cfg(not(feature = "wasm"))]
use case::CaseExt;
#[cfg(not(feature = "wasm"))]
use log::error;
#[cfg(not(feature = "wasm"))]
use plotly::ImageFormat;

// List all available transformations
const AVAILABLE_TRANSFORMATIONS: &[&str] =
    &["log", "multscale", "pow", "abs", "neg-exp", "softplus"];

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
    /// Visualize an EnzymeML document
    Visualize {
        /// Path to the file containing the EnzymeML document
        #[arg(help = "Path to the file containing the EnzymeML document")]
        path: PathBuf,

        /// Measurement IDs to visualize
        #[arg(short, long, help = "Measurement IDs to visualize")]
        measurement_ids: Vec<String>,

        /// Output directory for the visualization
        #[arg(
            short,
            long,
            help = "Output directory for the visualization",
            conflicts_with = "show",
            default_value = "."
        )]
        output_dir: Option<PathBuf>,

        /// Show fit or not
        #[arg(short = 'F', long, help = "Show fit or not", default_value_t = true)]
        show_fit: bool,

        /// Show in browser or not
        #[arg(
            short,
            long,
            help = "Show in browser or not",
            default_value_t = false,
            conflicts_with = "output_dir"
        )]
        show: bool,
    },

    /// Convert an EnzymeML document to a target
    Convert {
        /// Path to the file containing the EnzymeML document
        #[arg(
            short,
            long,
            help = "Path to the file containing the EnzymeML document"
        )]
        input: PathBuf,

        /// Target format
        #[arg(short, long, help = "Target format")]
        target: ConversionTarget,

        /// Output path for the converted document
        #[arg(short, long, help = "Output path for the converted document.")]
        output: PathBuf,

        /// As template or not
        #[arg(long, help = "As template or not")]
        template: bool,
    },

    /// Validate an EnzymeML document
    Validate {
        /// Path to the file containing the EnzymeML document
        #[arg(help = "Path to the file containing the EnzymeML document")]
        path: PathBuf,
    },

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

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Bounds for the optimization
        #[arg(short, long, value_parser = parse_key_bounds)]
        bound: Vec<(String, (f64, f64))>,

        /// Whether to use the log transformation for the initial guesses
        #[arg(short, long, conflicts_with = "transform")]
        log_transform: bool,

        /// Transformations for the initial guesses
        #[arg(short, long, help = format!("Transformations for the initial guesses. Available transformations: [{}]", AVAILABLE_TRANSFORMATIONS.join(", ")), conflicts_with = "log_transform")]
        transform: Vec<Transformation>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Objective function to use
        #[arg(short = 'O', long, default_value = "mse", value_enum)]
        objective: LossFunction,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: Option<PathBuf>,
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

        /// Whether to use the log transformation for the initial guesses
        #[arg(short, long, conflicts_with = "transform")]
        log_transform: bool,

        /// Transformations for the initial guesses
        #[arg(short, long, help = format!("Transformations for the initial guesses. Available transformations: [{}]", AVAILABLE_TRANSFORMATIONS.join(", ")), conflicts_with = "log_transform")]
        transform: Vec<Transformation>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Objective function to use
        #[arg(short = 'O', long, default_value = "mse", value_enum)]
        objective: LossFunction,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: Option<PathBuf>,
    },

    /// LBFGS algorithm
    Lbfgs {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Initial guesses for the optimization
        #[arg(short, long, value_parser = parse_initial_guesses)]
        initial: Vec<(String, f64)>,

        /// Whether to use the log transformation for the initial guesses
        #[arg(short, long, conflicts_with = "transform")]
        log_transform: bool,

        /// Transformations for the initial guesses
        #[arg(short, long, help = format!("Transformations for the initial guesses. Available transformations: [{}]", AVAILABLE_TRANSFORMATIONS.join(", ")), conflicts_with = "log_transform")]
        transform: Vec<Transformation>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Objective function to use
        #[arg(short = 'O', long, default_value = "mse", value_enum)]
        objective: LossFunction,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 20)]
        max_iters: u64,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: Option<PathBuf>,
    },

    /// BFGS algorithm
    Bfgs {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Initial guesses for the optimization
        #[arg(short, long, value_parser = parse_initial_guesses)]
        initial: Vec<(String, f64)>,

        /// Whether to use the log transformation for the initial guesses
        #[arg(short, long, conflicts_with = "transform")]
        log_transform: bool,

        /// Transformations for the initial guesses
        #[arg(short, long, help = format!("Transformations for the initial guesses. Available transformations: [{}]", AVAILABLE_TRANSFORMATIONS.join(", ")), conflicts_with = "log_transform")]
        transform: Vec<Transformation>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Objective function to use
        #[arg(short = 'O', long, default_value = "mse", value_enum)]
        objective: LossFunction,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 20)]
        max_iters: u64,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: Option<PathBuf>,
    },

    /// SR1TrustRegion algorithm
    SR1 {
        /// Path to the EnzymeML document
        #[arg(short, long)]
        path: PathBuf,

        /// Initial guesses for the optimization
        #[arg(short, long, value_parser = parse_initial_guesses)]
        initial: Vec<(String, f64)>,

        /// Whether to use the log transformation for the initial guesses
        #[arg(short, long, conflicts_with = "transform")]
        log_transform: bool,

        /// Transformations for the initial guesses
        #[arg(short, long, help = format!("Transformations for the initial guesses. Available transformations: [{}]", AVAILABLE_TRANSFORMATIONS.join(", ")), conflicts_with = "log_transform")]
        transform: Vec<Transformation>,

        /// Solver to use
        #[arg(
            short,
            long,
            value_parser = Solvers::from_str,
            help = "Solver to use. Available solvers: [rk5, rk4, rkf45, tsit45, dp45, bs23, rals3, rals4]"
        )]
        solver: Solvers,

        /// Subproblem solver to use
        #[arg(short = 'S', long, default_value = "steihaug", value_enum)]
        subproblem: SubProblem,

        /// Objective function to use
        #[arg(short = 'O', long, default_value = "mse", value_enum)]
        objective: LossFunction,

        /// Time step for the design points
        #[arg(long, default_value_t = 0.1)]
        dt: f64,

        /// Maximum number of iterations before stopping
        #[arg(long, default_value_t = 40)]
        max_iters: u64,

        /// Output directory for the optimization report
        #[arg(short, long, default_value = ".")]
        output_dir: Option<PathBuf>,
    },
}

/// Main entry point for the CLI application
pub fn main() {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    match &cli.command {
        // TODO: This is very ugly, we should extract the WASM feature into a separate crate
        #[cfg(feature = "wasm")]
        Commands::Visualize { .. } => {
            eprintln!("Visualization is not supported in WASM mode");
        }

        #[cfg(not(feature = "wasm"))]
        Commands::Visualize {
            path,
            measurement_ids,
            output_dir,
            show_fit,
            show,
        } => {
            // Load the enzymeml document
            let enzmldoc = complete_check(path).expect("Failed to validate EnzymeML document");

            // Get the measurement ids
            let measurement_ids = if !measurement_ids.is_empty() {
                measurement_ids.clone()
            } else {
                enzmldoc.measurements.iter().map(|m| m.id.clone()).collect()
            };

            if let Some(output_dir) = output_dir {
                let output_dir = output_dir.to_owned();
                if output_dir.exists() && !output_dir.is_dir() {
                    panic!(
                        "Output directory is not a directory: {}",
                        output_dir.display()
                    );
                }
                std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
            }

            for meas_id in &measurement_ids {
                let plot = enzmldoc.plot_measurement(meas_id, *show, *show_fit, None, None);

                match plot {
                    Ok(plot) => {
                        if let Some(output_dir) = output_dir {
                            let fname = format!("{}.png", meas_id.to_snake());
                            let output_path = output_dir.join(fname);
                            plot.write_image(&output_path, ImageFormat::PNG, 800, 600, 1.0);

                            println!(
                                "Saved measurement {} plot to {}",
                                meas_id.cyan().bold(),
                                output_path.to_str().unwrap().to_string().cyan().bold()
                            );
                        }
                    }
                    Err(e) => {
                        error!("Failed to plot measurement {}: {}", meas_id, e);
                    }
                }
            }
        }
        Commands::Convert {
            input: path,
            target,
            output,
            template,
        } => {
            // Check if the file is a valid EnzymeML document
            let enzmldoc = complete_check(path);

            if let Err(e) = enzmldoc {
                // If the document is invalid, print the error and return
                println!("{}", e);
                return;
            }

            // We know the document is valid, so we can unwrap it
            let enzmldoc = enzmldoc.unwrap();

            match target {
                ConversionTarget::Xlsx => {
                    // Convert the document to XLSX
                    enzmldoc
                        .to_excel(output.to_path_buf(), *template)
                        .expect("Failed to convert EnzymeML document to XLSX");
                }
            }
        }

        Commands::Validate { path } => {
            // First, check if the file exists
            if !Path::new(path).exists() {
                eprintln!("File does not exist: {}", path.display());
                return;
            }

            if let Err(e) = complete_check(path) {
                println!("{}", e);
            }
        }
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
            FitAlgorithm::Lbfgs {
                path,
                initial,
                transform,
                solver,
                objective: loss_function,
                dt,
                max_iters,
                output_dir,
                log_transform,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let transformations = if *log_transform {
                    create_log_transformations(&enzmldoc)
                } else {
                    transform.clone()
                };

                // Build problem
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .objective(*loss_function)
                    .transformations(transformations)
                    .build()
                    .expect("Failed to build problem");

                // Convert EnzymeML document to initial guesses
                let param_order = problem.ode_system().get_sorted_params();
                let mut initial_guesses: InitialGuesses = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Override initial guesses
                override_initial_guesses(&param_order, &mut initial_guesses, initial);

                // Build optimizer
                let lbfgs = LBFGSBuilder::default()
                    .linesearch(1e-4, 0.9)
                    .max_iters(*max_iters)
                    .target_cost(1e-6)
                    .build();

                // Optimize
                let report = lbfgs
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                // Save the fitted EnzymeML document
                if let Some(output_dir) = output_dir {
                    save_fitted_enzmldoc(path, &report.doc, output_dir, "lbfgs");
                }
            }

            FitAlgorithm::Bfgs {
                path,
                initial,
                transform,
                solver,
                objective: loss_function,
                dt,
                max_iters,
                output_dir,
                log_transform,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let transformations = if *log_transform {
                    create_log_transformations(&enzmldoc)
                } else {
                    transform.clone()
                };

                // Build problem
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .objective(*loss_function)
                    .transformations(transformations)
                    .build()
                    .expect("Failed to build problem");

                // Convert EnzymeML document to initial guesses
                let param_order = problem.ode_system().get_sorted_params();
                let mut initial_guesses: InitialGuesses = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Override initial guesses
                override_initial_guesses(&param_order, &mut initial_guesses, initial);

                // Build optimizer
                let bfgs = BFGSBuilder::default()
                    .linesearch(1e-4, 0.9)
                    .max_iters(*max_iters)
                    .target_cost(1e-6)
                    .build();

                // Optimize
                let report = bfgs
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                // Save the fitted EnzymeML document
                if let Some(output_dir) = output_dir {
                    save_fitted_enzmldoc(path, &report.doc, output_dir, "bfgs");
                }
            }

            FitAlgorithm::SR1 {
                path,
                initial,
                transform,
                solver,
                objective: loss_function,
                dt,
                max_iters,
                output_dir,
                log_transform,
                subproblem,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let transformations = if *log_transform {
                    create_log_transformations(&enzmldoc)
                } else {
                    transform.clone()
                };

                // Build problem
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .objective(*loss_function)
                    .transformations(transformations)
                    .build()
                    .expect("Failed to build problem");

                // Convert EnzymeML document to initial guesses
                let param_order = problem.ode_system().get_sorted_params();
                let mut initial_guesses: InitialGuesses = (&enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Override initial guesses
                override_initial_guesses(&param_order, &mut initial_guesses, initial);

                // Build optimizer
                let sr1trustregion = SR1TrustRegionBuilder::default()
                    .max_iters(*max_iters)
                    .subproblem(*subproblem)
                    .build();

                // Optimize
                let report = sr1trustregion
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                // Save the fitted EnzymeML document
                if let Some(output_dir) = output_dir {
                    save_fitted_enzmldoc(path, &report.doc, output_dir, "sr1trustregion");
                }
            }
            FitAlgorithm::Ego {
                path,
                max_iters,
                output_dir,
                dt,
                bound,
                solver,
                objective,
                log_transform,
                transform,
            } => {
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let transformations = if *log_transform {
                    create_log_transformations(&enzmldoc)
                } else {
                    transform.clone()
                };

                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .objective(*objective)
                    .transformations(transformations)
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

                // Display the results
                println!("{}", report);

                // Save the fitted EnzymeML document
                if let Some(output_dir) = output_dir {
                    save_fitted_enzmldoc(path, &report.doc, output_dir, "ego");
                }
            }

            FitAlgorithm::Pso {
                path,
                max_iters,
                pop_size,
                output_dir,
                dt,
                bound,
                solver,
                objective,
                log_transform,
                transform,
            } => {
                // Load EnzymeML document
                let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
                let transformations = if *log_transform {
                    create_log_transformations(&enzmldoc)
                } else {
                    transform.clone()
                };

                // Build problem
                let problem = ProblemBuilder::new(&enzmldoc, *solver)
                    .dt(*dt)
                    .objective(*objective)
                    .transformations(transformations)
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

                // Display the results
                println!("{}", report);

                // Save the fitted EnzymeML document
                if let Some(output_dir) = output_dir {
                    save_fitted_enzmldoc(path, &report.doc, output_dir, "pso");
                }
            }
        },
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConversionTarget {
    Xlsx,
}

fn complete_check(path: &PathBuf) -> Result<EnzymeMLDocument, String> {
    // Check if the file is a valid EnzymeML document
    let content = std::fs::read_to_string(path).expect("Failed to read EnzymeML document");
    let report = schema::validate_json(&content).expect("Failed to validate EnzymeML document");

    if !report.valid {
        println!("{}", "EnzymeML document is invalid".bold().red());
        for error in report.errors {
            println!("   {}", error);
        }
        return Err("EnzymeML document is invalid".to_string());
    }

    // Check if the document is consistent
    let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");
    let report = consistency::check_consistency(&enzmldoc);

    if !report.is_valid {
        println!("{}", "EnzymeML document is inconsistent".bold().red());
        for error in report.errors {
            println!("   {}", error);
        }
        return Err("EnzymeML document is inconsistent".to_string());
    }

    println!("{}", "EnzymeML document is valid".bold().green());

    Ok(enzmldoc)
}

/// Override the initial guesses with the ones from the CLI
///
/// # Arguments
///
/// * `param_order` - The order of the parameters
/// * `initial_guesses` - The initial guesses
/// * `initial` - The initial guesses from the CLI
fn override_initial_guesses(
    param_order: &[String],
    initial_guesses: &mut InitialGuesses,
    initial: &[(String, f64)],
) {
    for (key, value) in initial {
        let index = param_order
            .iter()
            .position(|p| p == key)
            .expect("Parameter not found");

        initial_guesses.set_value_at(index, *value);
    }
}

/// Creates log transformations for the parameters in the EnzymeML document
fn create_log_transformations(enzmldoc: &EnzymeMLDocument) -> Vec<Transformation> {
    let parameter_names = enzmldoc
        .parameters
        .iter()
        .map(|p| p.symbol.clone())
        .collect::<Vec<_>>();
    parameter_names
        .iter()
        .map(|p| Transformation::Log(p.to_string()))
        .collect::<Vec<_>>()
}

fn parse_initial_guesses(s: &str) -> Result<(String, f64), String> {
    // Example k_cat=0.1
    let parts = s.split('=').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err("Invalid format. Expected key=initial".to_string());
    }
    let key = parts[0].to_string();
    let initial = parts[1].to_string().parse::<f64>().unwrap();
    Ok((key, initial))
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
#[allow(clippy::ptr_arg)]
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

/// Saves the fitted EnzymeML document to the output directory
///
/// # Arguments
///
/// * `doc_path` - The path to the EnzymeML document
/// * `enzmldoc` - The fitted EnzymeML document
/// * `output_dir` - The output directory
fn save_fitted_enzmldoc(
    doc_path: &Path,
    enzmldoc: &EnzymeMLDocument,
    output_dir: &Path,
    optimizer: &str,
) {
    let name = format!(
        "{}_{}.json",
        doc_path
            .file_name()
            .expect("Failed to get file name")
            .to_str()
            .expect("Failed to get file name as string")
            .split('.')
            .next()
            .expect("Failed to get file name without extension"),
        optimizer,
    );
    let output_path = output_dir.join(name);
    save_enzmldoc(&output_path, enzmldoc).expect("Failed to save EnzymeML document");
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

    #[allow(clippy::default_constructed_unit_structs)]
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
