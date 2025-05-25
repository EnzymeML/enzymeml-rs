//! Command-line interface for the EnzymeML library
//!
//! This binary provides a CLI interface to interact with EnzymeML documents, including:
//! - Extracting information from natural language descriptions using LLMs
//! - Fitting kinetic parameters using various optimization algorithms
//! - Converting EnzymeML documents to different formats
//! - Validating EnzymeML documents for schema compliance and consistency
//! - Visualizing measurement data and simulation results
//! - Performing profile likelihood analysis for parameter identifiability
//!
//! # Usage
//!
//! ```bash
//! # Extract information from text using LLM
//! enzymeml extract --prompt input.txt --output result.json
//!
//! # Fit parameters using EGO algorithm
//! enzymeml fit ego --path model.json --max-iters 100
//!
//! # Fit parameters using PSO algorithm  
//! enzymeml fit pso --path model.json --pop-size 50
//!
//! # Convert EnzymeML document to Excel format
//! enzymeml convert --input model.json --target xlsx --output model.xlsx
//!
//! # Validate an EnzymeML document
//! enzymeml validate model.json
//!
//! # Visualize measurement data
//! enzymeml visualize model.json --measurement-ids meas1 meas2 --show-fit
//!
//! # Profile likelihood analysis for parameter identifiability
//! enzymeml profile sr1 --path model.json --fixed k_cat --from 0.1 --to 10.0 --steps 100
//! ```
//!
//! # Commands
//!
//! - `convert`: Convert EnzymeML documents to different formats (currently supports XLSX)
//! - `validate`: Check EnzymeML documents for schema compliance and consistency
//! - `visualize`: Generate plots of measurement data and simulation results
//! - `fit`: Fit kinetic parameters using various optimization algorithms
//!   - `ego`: Efficient Global Optimization algorithm
//!   - `pso`: Particle Swarm Optimization algorithm
//!   - `lbfgs`: Limited-memory BFGS algorithm
//!   - `bfgs`: Broyden–Fletcher–Goldfarb–Shanno algorithm
//!   - `sr1`: Symmetric Rank 1 Trust Region algorithm
//! - `extract`: Extract structured information from natural language using LLMs
//! - `profile`: Perform profile likelihood analysis for parameter identifiability
//!   - `sr1`: Using Symmetric Rank 1 Trust Region algorithm
//!   - `lbfgs`: Using Limited-memory BFGS algorithm
//!   - `bfgs`: Using Broyden–Fletcher–Goldfarb–Shanno algorithm
//!
//! # Parameter Optimization
//!
//! The CLI supports multiple optimization algorithms for parameter fitting:
//!
//! - **EGO**: Efficient Global Optimization, a surrogate-based algorithm suitable for expensive objective functions
//! - **PSO**: Particle Swarm Optimization, a population-based stochastic algorithm
//! - **LBFGS/BFGS**: Gradient-based optimization algorithms for smooth objective functions
//! - **SR1**: Symmetric Rank 1 Trust Region method for constrained optimization problems
//!
//! # Profile Likelihood Analysis
//!
//! Profile likelihood analysis is used to assess parameter identifiability and confidence intervals:
//!
//! - Systematically varies a single parameter while re-optimizing all others
//! - Helps identify practical and structural non-identifiability
//! - Provides confidence intervals for parameter estimates
//! - Visualizes the likelihood profile for each parameter
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

use clap::{Args, Parser, Subcommand, ValueEnum};
use colored::Colorize;
use enzymeml::{
    identifiability::{
        egoprofile::ego_profile_likelihood, parameter::ProfileParameter,
        profile::profile_likelihood, results::plot_pair_contour,
    },
    io::{load_enzmldoc, save_enzmldoc},
    llm::{query_llm, PromptInput},
    optim::{
        report::OptimizationReport, BFGSBuilder, EGOBuilder, InitialGuesses, LBFGSBuilder,
        OptimizeError, Optimizer, PSOBuilder, Problem, ProblemBuilder, SR1TrustRegionBuilder,
        SubProblem, Transformation,
    },
    prelude::{EnzymeMLDocument, LossFunction, NegativeLogLikelihood},
    validation::{consistency, schema},
};

use peroxide::fuga::{self, anyhow, ODEIntegrator, ODEProblem};

// TODO: This is very ugly, we should extract the WASM feature into a separate crate
use rust_xlsxwriter::workbook::Workbook;

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
    /// Print information about an EnzymeML document
    Info {
        /// Path to the file containing the EnzymeML document
        #[arg(help = "Path to the file containing the EnzymeML document")]
        path: PathBuf,
    },

    /// Visualize an EnzymeML document
    Visualize {
        /// Path to the file containing the EnzymeML document
        #[arg(help = "Path to the file containing the EnzymeML document")]
        path: PathBuf,

        /// Measurement IDs to visualize
        #[arg(
            short,
            long,
            help = "Measurement IDs to visualize. Default is all measurements."
        )]
        measurement: Option<Vec<String>>,

        /// Output directory for the visualization
        #[arg(short, long, help = "Path to the file to save the visualization to.")]
        output: Option<PathBuf>,

        /// Show fit or not
        #[arg(short = 'F', long, help = "Show fit or not")]
        fit: bool,

        /// Show in browser or not
        #[arg(short, long, help = "Show in browser or not", default_value_t = true)]
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
        algorithm: OptimizeAlgorithm,
    },

    /// Calculate the profile likelihood for a model
    Profile {
        #[command(subcommand)]
        algorithm: ProfileAlgorithm,
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
enum OptimizeAlgorithm {
    /// Efficient Global Optimization algorithm
    Ego {
        #[command(flatten)]
        params: EgoParams,

        #[command(flatten)]
        fit_params: BaseFitParams,
    },

    /// Particle Swarm Optimization algorithm
    Pso {
        #[command(flatten)]
        params: PSOParams,

        #[command(flatten)]
        fit_params: BaseFitParams,
    },

    /// LBFGS algorithm
    Lbfgs {
        #[command(flatten)]
        params: LbfgsParams,

        #[command(flatten)]
        fit_params: BaseFitParams,
    },

    /// BFGS algorithm
    Bfgs {
        #[command(flatten)]
        params: LbfgsParams,

        #[command(flatten)]
        fit_params: BaseFitParams,
    },

    /// SR1TrustRegion algorithm
    SR1 {
        #[command(flatten)]
        params: SR1Params,

        #[command(flatten)]
        fit_params: BaseFitParams,
    },
}

/// Available profile likelihood algorithms
#[derive(Subcommand)]
enum ProfileAlgorithm {
    /// SR1TrustRegion algorithm for profile likelihood analysis
    SR1 {
        #[command(flatten)]
        opt_params: SR1Params,

        #[command(flatten)]
        profile_params: BaseProfileParams,
    },

    /// LBFGS algorithm for profile likelihood analysis
    Lbfgs {
        #[command(flatten)]
        params: LbfgsParams,

        #[command(flatten)]
        profile_params: BaseProfileParams,
    },

    /// BFGS algorithm for profile likelihood analysis
    Bfgs {
        #[command(flatten)]
        params: LbfgsParams,

        #[command(flatten)]
        profile_params: BaseProfileParams,
    },
}

/// Base parameters for integrator configuration
#[derive(Args)]
struct BaseIntegratorParams {
    /// Path to the EnzymeML document
    #[arg(short, long)]
    path: PathBuf,

    /// Whether to use the log transformation for the initial guesses
    #[arg(short, long, conflicts_with = "transform", default_value_t = true)]
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

    /// Time step for the design points
    #[arg(long, default_value_t = 0.1)]
    dt: f64,

    /// Maximum number of iterations before stopping
    #[arg(long, default_value_t = 40)]
    max_iters: u64,
}

impl BaseIntegratorParams {
    /// Creates a Problem instance from the parameters
    ///
    /// # Arguments
    ///
    /// * `objective` - The objective function to use for optimization
    ///
    /// # Returns
    ///
    /// * `Result<Problem<Solvers, LossFunction>, OptimizeError>` - The problem instance or an error
    pub fn to_problem(
        &self,
        objective: impl Into<LossFunction>,
    ) -> Result<Problem<Solvers, LossFunction>, OptimizeError> {
        let enzmldoc = load_enzmldoc(&self.path).expect("Failed to load EnzymeML document");

        let transformations = if self.log_transform {
            create_log_transformations(&enzmldoc)
        } else {
            self.transform.clone()
        };

        // Build problem
        ProblemBuilder::new(&enzmldoc, self.solver, objective.into())
            .dt(self.dt)
            .transformations(transformations)
            .build()
    }
}

/// Base parameters for fitting operations
#[derive(Args)]
struct BaseFitParams {
    /// Fixed parameters
    #[arg(short, long, help = "Fixed parameters", default_value = "")]
    fix: Vec<String>,

    /// Output directory for the optimization report
    #[arg(short, long, default_value = ".")]
    output_dir: Option<PathBuf>,

    /// Objective function to use
    #[arg(
        short = 'O',
        long,
        default_value = "mse",
        value_parser = LossFunction::from_str,
        help = "Objective function to use. Available objective functions: [mse, sse, rmse, logcosh, mae, nll(sigma)]"
    )]
    objective: LossFunction,

    /// Show the fit
    #[arg(long, help = "Show the fit")]
    show: bool,
}

/// Base parameters for profile likelihood analysis
#[derive(Args)]
struct BaseProfileParams {
    /// Parameters to profile
    #[arg(short = 'P', long = "parameter", help = "Parameters to profile. Format: <name>=<from>:<to>", value_parser = ProfileParameter::from_str)]
    parameters: Vec<ProfileParameter>,

    /// Number of steps for the profile likelihood calculation
    #[arg(
        long,
        help = "Number of steps for the profile likelihood calculation",
        default_value_t = 10,
        conflicts_with = "ego"
    )]
    steps: usize,

    /// Sigma value for the negative log likelihood function
    #[arg(
        long,
        help = "Sigma value for the negative log likelihood function",
        default_value = "1.0"
    )]
    sigma: f64,

    /// Output directory for the profile likelihood plot
    #[arg(short, long, help = "Output directory for the profile likelihood plot")]
    out: Option<PathBuf>,

    /// Whether to use EGO-based profile likelihood
    #[arg(
        long,
        help = "Whether to use EGO-based profile likelihood",
        conflicts_with = "steps"
    )]
    ego: bool,

    /// Maximum number of iterations for the profile likelihood calculation
    #[arg(
        long = "ego-iters",
        help = "Maximum number of iterations for the profile likelihood calculation",
        conflicts_with = "steps",
        default_value_t = 10
    )]
    ego_iters: usize,

    /// Show the profile likelihood plot
    #[arg(long, help = "Show the profile likelihood plot")]
    show: bool,
}

/// Parameters for SR1 Trust Region optimization
#[derive(Args)]
struct SR1Params {
    #[command(flatten)]
    params: BaseIntegratorParams,

    /// Initial guesses for the optimization
    #[arg(short, long, value_parser = parse_initial_guesses)]
    initial: Vec<(String, f64)>,

    /// Subproblem solver to use
    #[arg(short = 'S', long, default_value = "steihaug", value_enum)]
    subproblem: SubProblem,
}

/// Parameters for LBFGS/BFGS optimization
#[derive(Args)]
struct LbfgsParams {
    #[command(flatten)]
    params: BaseIntegratorParams,

    /// Initial guesses for the optimization
    #[arg(short, long, value_parser = parse_initial_guesses)]
    initial: Vec<(String, f64)>,
}

/// Parameters for Particle Swarm Optimization
#[derive(Args)]
struct PSOParams {
    #[command(flatten)]
    params: BaseIntegratorParams,

    /// Population size for the algorithm
    #[arg(long, default_value_t = 50)]
    pop_size: usize,

    /// Bounds for the optimization
    #[arg(short, long, value_parser = parse_key_bounds)]
    bound: Vec<(String, (f64, f64))>,
}

/// Parameters for Efficient Global Optimization
#[derive(Args)]
struct EgoParams {
    #[command(flatten)]
    params: BaseIntegratorParams,

    /// Bounds for the optimization
    #[arg(short, long, value_parser = parse_key_bounds)]
    bound: Vec<(String, (f64, f64))>,
}

/// Main entry point for the CLI application
pub fn main() {
    // Initialize logger
    env_logger::init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Info { path } => {
            let enzmldoc = complete_check(path).expect("Failed to validate EnzymeML document");
            println!("{}", enzmldoc);
        }

        Commands::Visualize {
            path,
            measurement,
            output,
            fit,
            show,
        } => {
            // Load the enzymeml document
            validate_by_schema(path).expect("Failed to validate EnzymeML document");
            let enzmldoc = load_enzmldoc(path).expect("Failed to load EnzymeML document");

            // Create the output directory if it doesn't exist
            if let Some(output) = output {
                let parent = output.parent().unwrap();
                if !parent.exists() {
                    std::fs::create_dir_all(parent).expect("Failed to create output directory");
                }
            }

            // Plot the measurements
            let plot = enzmldoc
                .plot_measurements()
                .measurement_ids(measurement.clone())
                .show_fit(*fit)
                .call()
                .expect("Failed to plot measurements");

            // Save the plot to the output file
            if let Some(output) = output {
                // Replace the extension with .html if there is already an extension
                let output = output.with_extension("html");
                plot.write_html(&output);
                println!(
                    "Saved measurements plot to {}",
                    output.display().to_string().bold().green()
                );
            }

            if *show {
                plot.show();
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
            let mut enzmldoc = enzmldoc.unwrap();

            match target {
                ConversionTarget::Xlsx => {
                    if *template {
                        enzmldoc.measurements.clear();
                    }

                    // Convert the document to XLSX
                    let mut workbook = Workbook::try_from(enzmldoc)
                        .expect("Failed to convert EnzymeML document to XLSX");
                    workbook.save(output).expect("Failed to save XLSX file");
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
            OptimizeAlgorithm::Lbfgs { params, fit_params } => {
                let mut problem = params
                    .params
                    .to_problem(fit_params.objective)
                    .expect("Failed to build problem");

                // Override initial guesses
                override_initial_guesses(problem.enzmldoc_mut(), &params.initial);

                let enzmldoc = problem.enzmldoc();
                let initial_guesses: InitialGuesses = (enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Build optimizer
                let lbfgs = LBFGSBuilder::default()
                    .linesearch(1e-4, 0.9)
                    .max_iters(params.params.max_iters)
                    .target_cost(1e-6)
                    .build();

                // Optimize
                let report = lbfgs
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                if fit_params.show {
                    let plot = report
                        .doc
                        .plot_measurements()
                        .measurement_ids(None)
                        .show_fit(true)
                        .call()
                        .expect("Failed to plot measurements");
                    plot.show();
                }

                // Save the fitted EnzymeML document
                if let Some(output_dir) = &fit_params.output_dir {
                    save_results(
                        &params.params.path,
                        &report,
                        &report.doc,
                        output_dir,
                        "lbfgs",
                    );
                }
            }

            OptimizeAlgorithm::Bfgs { params, fit_params } => {
                let mut problem = params
                    .params
                    .to_problem(fit_params.objective)
                    .expect("Failed to build problem");

                // Override initial guesses
                override_initial_guesses(problem.enzmldoc_mut(), &params.initial);

                let enzmldoc = problem.enzmldoc();
                // Convert EnzymeML document to initial guesses
                let initial_guesses: InitialGuesses = (enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Build optimizer
                let bfgs = BFGSBuilder::default()
                    .linesearch(1e-4, 0.9)
                    .max_iters(params.params.max_iters)
                    .target_cost(1e-6)
                    .build();

                // Optimize
                let report = bfgs
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                if fit_params.show {
                    let plot = report
                        .doc
                        .plot_measurements()
                        .measurement_ids(None)
                        .show_fit(true)
                        .call()
                        .expect("Failed to plot measurements");
                    plot.show();
                }

                // Save the fitted EnzymeML document
                if let Some(output_dir) = &fit_params.output_dir {
                    save_results(
                        &params.params.path,
                        &report,
                        &report.doc,
                        output_dir,
                        "bfgs",
                    );
                }
            }

            OptimizeAlgorithm::SR1 { params, fit_params } => {
                let mut problem = params
                    .params
                    .to_problem(fit_params.objective)
                    .expect("Failed to build problem");

                // Override initial guesses
                override_initial_guesses(problem.enzmldoc_mut(), &params.initial);

                let enzmldoc = problem.enzmldoc();

                // Convert EnzymeML document to initial guesses
                let initial_guesses: InitialGuesses = (enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to initial guesses");

                // Build optimizer
                let sr1trustregion = SR1TrustRegionBuilder::default()
                    .max_iters(params.params.max_iters)
                    .subproblem(params.subproblem)
                    .build();

                // Optimize
                let report = sr1trustregion
                    .optimize(&problem, Some(initial_guesses))
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                if fit_params.show {
                    let plot = report
                        .doc
                        .plot_measurements()
                        .measurement_ids(None)
                        .show_fit(true)
                        .call()
                        .expect("Failed to plot measurements");
                    plot.show();
                }

                // Save the fitted EnzymeML document
                if let Some(output_dir) = &fit_params.output_dir {
                    save_results(
                        &params.params.path,
                        &report,
                        &report.doc,
                        output_dir,
                        "sr1trustregion",
                    );
                }
            }
            OptimizeAlgorithm::Ego { params, fit_params } => {
                let mut problem = params
                    .params
                    .to_problem(fit_params.objective)
                    .expect("Failed to build problem");

                // Add CLI bounds
                add_cli_bounds(problem.enzmldoc_mut(), &params.bound);

                let enzmldoc = problem.enzmldoc();

                // Convert EnzymeML document to bounds
                let bounds = (enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");

                // Build optimizer
                let optimizer = EGOBuilder::default()
                    .bounds(bounds)
                    .max_iters(params.params.max_iters as usize)
                    .build();

                // Optimize
                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                if fit_params.show {
                    let plot = report
                        .doc
                        .plot_measurements()
                        .measurement_ids(None)
                        .show_fit(true)
                        .call()
                        .expect("Failed to plot measurements");
                    plot.show();
                }

                // Save the fitted EnzymeML document
                if let Some(output_dir) = &fit_params.output_dir {
                    save_results(&params.params.path, &report, &report.doc, output_dir, "ego");
                }
            }

            OptimizeAlgorithm::Pso { params, fit_params } => {
                let mut problem = params
                    .params
                    .to_problem(fit_params.objective)
                    .expect("Failed to build problem");

                // Add CLI bounds
                add_cli_bounds(problem.enzmldoc_mut(), &params.bound);

                let enzmldoc = problem.enzmldoc();

                // Convert EnzymeML document to bounds
                let bounds = (enzmldoc)
                    .try_into()
                    .expect("Failed to convert EnzymeML document to bounds");

                // Build optimizer
                let optimizer = PSOBuilder::default()
                    .bounds(bounds)
                    .max_iters(params.params.max_iters)
                    .pop_size(params.pop_size)
                    .build();

                // Optimize
                let report = optimizer
                    .optimize(&problem, None::<InitialGuesses>)
                    .expect("Failed to optimize");

                // Display the results
                println!("{}", report);

                if fit_params.show {
                    let plot = report
                        .doc
                        .plot_measurements()
                        .measurement_ids(None)
                        .show_fit(true)
                        .call()
                        .expect("Failed to plot measurements");
                    plot.show();
                }

                // Save the fitted EnzymeML document
                if let Some(output_dir) = &fit_params.output_dir {
                    save_results(&params.params.path, &report, &report.doc, output_dir, "pso");
                }
            }
        },
        Commands::Profile { algorithm } => {
            match algorithm {
                ProfileAlgorithm::SR1 {
                    opt_params,
                    profile_params,
                } => {
                    let mut problem = opt_params
                        .params
                        .to_problem(NegativeLogLikelihood::new(profile_params.sigma))
                        .expect("Failed to build problem");

                    // Override initial guesses
                    override_initial_guesses(problem.enzmldoc_mut(), &opt_params.initial);

                    let enzmldoc = problem.enzmldoc();

                    // Convert EnzymeML document to initial guesses
                    let initial_guesses: InitialGuesses = (enzmldoc)
                        .try_into()
                        .expect("Failed to convert EnzymeML document to initial guesses");

                    // Build optimizer
                    let sr1trustregion = SR1TrustRegionBuilder::default()
                        .max_iters(opt_params.params.max_iters)
                        .subproblem(opt_params.subproblem)
                        .build();

                    let profile_result = if profile_params.ego {
                        ego_profile_likelihood()
                            .problem(&problem)
                            .optimizer(&sr1trustregion)
                            .initial_guess(initial_guesses)
                            .parameters(profile_params.parameters.clone())
                            .max_iters(profile_params.ego_iters)
                            .call()
                            .expect("Failed to profile likelihood")
                    } else {
                        profile_likelihood()
                            .problem(&problem)
                            .optimizer(&sr1trustregion)
                            .initial_guess(initial_guesses)
                            .parameters(profile_params.parameters.clone())
                            .n_steps(profile_params.steps)
                            .call()
                            .expect("Failed to profile likelihood")
                    };

                    let plot = if profile_result.len() > 1 {
                        plot_pair_contour(&profile_result)
                    } else {
                        profile_result.into()
                    };

                    if profile_params.show {
                        plot.show();
                    }

                    if let Some(out) = &profile_params.out {
                        let fname = opt_params
                            .params
                            .path
                            .file_name()
                            .expect("Failed to get file name")
                            .to_str()
                            .expect("Failed to get file name as string")
                            .split('.')
                            .next()
                            .expect("Failed to get file name without extension");

                        plot.write_html(out.join(format!("{}_profile.html", fname)));
                    }
                }
                _ => {
                    unimplemented!()
                }
            };
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConversionTarget {
    Xlsx,
}

/// Performs comprehensive validation of an EnzymeML document
///
/// Checks both schema compliance and data consistency:
/// 1. Validates against the EnzymeML JSON schema
/// 2. Verifies internal data consistency
///
/// # Arguments
///
/// * `path` - Path to the EnzymeML document
///
/// # Returns
///
/// * `Ok(EnzymeMLDocument)` - Valid document
/// * `Err(String)` - Error message if validation fails
fn complete_check(path: &PathBuf) -> Result<EnzymeMLDocument, String> {
    validate_by_schema(path)?;
    check_consistency(path)
}

fn check_consistency(path: &PathBuf) -> Result<EnzymeMLDocument, String> {
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

    Ok(enzmldoc)
}

/// Validates the EnzymeML document by the JSON schema
///
/// # Arguments
///
/// * `path` - Path to the EnzymeML document
///
/// # Returns
///
/// * `Ok(())` - Valid document
/// * `Err(String)` - Error message if validation fails
fn validate_by_schema(path: &PathBuf) -> Result<(), String> {
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

    Ok(())
}

/// Override the initial guesses with the ones from the CLI
///
/// # Arguments
///
/// * `enzmldoc` - The EnzymeML document
/// * `initial` - The initial guesses from the CLI
fn override_initial_guesses(enzmldoc: &mut EnzymeMLDocument, initial: &[(String, f64)]) {
    for param in enzmldoc.parameters.iter_mut() {
        if let Some(initial) = initial.iter().find(|(name, _)| name == &param.symbol) {
            param.initial_value = Some(initial.1);
        }
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

/// Parses initial parameter guesses from command-line string
///
/// Format: "parameter_name=value"
/// Example: "k_cat=0.1"
///
/// # Arguments
///
/// * `s` - Input string in the format "parameter=value"
///
/// # Returns
///
/// * `Ok((String, f64))` - Parameter name and value
/// * `Err(String)` - Error message if parsing fails
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
/// * `enzmldoc` - The EnzymeML document
#[allow(clippy::ptr_arg)]
fn add_cli_bounds(enzmldoc: &mut EnzymeMLDocument, cli_bounds: &[(String, (f64, f64))]) {
    for param in enzmldoc.parameters.iter_mut() {
        if let Some((_, (upper, lower))) = cli_bounds.iter().find(|(key, _)| key == &param.symbol) {
            param.lower_bound = Some(*lower);
            param.upper_bound = Some(*upper);
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
fn save_results(
    doc_path: &Path,
    report: &OptimizationReport,
    enzmldoc: &EnzymeMLDocument,
    output_dir: &Path,
    optimizer: &str,
) {
    let doc_name = format!(
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
    let output_path = output_dir.join(doc_name);
    save_enzmldoc(&output_path, enzmldoc).expect("Failed to save EnzymeML document");

    // Save the report
    let report_name = format!(
        "{}_{}_report.json",
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
    let report_path = output_dir.join(report_name);
    let report_file = File::create(&report_path).expect("Failed to create report file");
    serde_json::to_writer_pretty(report_file, report).expect("Failed to write report to file");

    println!(
        "Results saved to {} and {}",
        output_path.display(),
        report_path.display()
    );
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
