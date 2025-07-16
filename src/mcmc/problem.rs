//! Bayesian problem formulation for MCMC sampling.
//!
//! This module provides the `BayesianProblem` struct which combines an optimization problem
//! with prior distributions to enable Bayesian inference using MCMC methods. The implementation
//! is compatible with the `nuts-rs` crate for No-U-Turn Sampling (NUTS).

use std::collections::HashMap;
use std::thread;

use crossbeam::channel;
use finitediff::FiniteDiff;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array1;
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, Settings};
use peroxide::fuga::{ODEIntegrator, ThreadRng};
use rand::distributions::Distribution;
use rayon::prelude::*;

use crate::{
    mcmc::{error::MCMCError, likelihood::Likelihood, output::SampleOutput, priors::DiffablePrior},
    optim::problem::Problem,
    prelude::{Mode, ObjectiveFunction},
};

/// Message type for communicating samples between chains and the output handler.
#[derive(Debug)]
pub struct SampleMessage {
    /// Chain identifier
    pub chain_id: String,
    /// Parameter sample vector
    pub sample: Vec<f64>,
}

/// Per-chain context with pre-allocated buffers for efficient computation.
pub struct ChainContext<S, L, P>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Likelihood,
    P: DiffablePrior + Distribution<f64>,
{
    bprob: BayesianProblem<S, L, P>,
    param_buffer: Array1<f64>,
    likelihood_grad_buffer: Array1<f64>,
}

impl<S, L, P> ChainContext<S, L, P>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Likelihood,
    P: DiffablePrior + Distribution<f64>,
{
    pub fn new(problem: BayesianProblem<S, L, P>) -> Self {
        let dim = problem.problem.get_n_params();
        Self {
            bprob: problem,
            param_buffer: Array1::zeros(dim),
            likelihood_grad_buffer: Array1::zeros(dim),
        }
    }
}

/// A Bayesian problem that combines an optimization problem with prior distributions.
///
/// This struct wraps an optimization `Problem` and adds prior distributions for each parameter,
/// enabling Bayesian inference through MCMC sampling. It implements the `CpuLogpFunc` trait
/// from `nuts-rs` to be compatible with the No-U-Turn Sampler (NUTS).
///
/// # Type Parameters
///
/// * `S` - ODE integrator type that must implement `ODEIntegrator + Copy + Send + Sync`
/// * `L` - Likelihood function type that must implement both `ObjectiveFunction` and `Likelihood`
/// * `P` - Prior distribution type that must implement `DiffablePrior + Distribution<f64>`
#[derive(Clone)]
pub struct BayesianProblem<S, L, P>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Likelihood,
    P: DiffablePrior + Distribution<f64>,
{
    /// The underlying optimization problem containing the model and likelihood function
    problem: Problem<S, L>,
    /// Vector of prior distributions, one for each parameter
    priors: Vec<P>,
}

#[bon::bon]
impl<S, L, P> BayesianProblem<S, L, P>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Likelihood,
    P: DiffablePrior + Distribution<f64>,
{
    /// Creates a new Bayesian problem from an optimization problem and prior distributions.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem containing the model and likelihood function
    /// * `priors` - HashMap of prior distributions keyed by parameter name
    ///
    /// # Returns
    ///
    /// A new `BayesianProblem` instance ready for MCMC sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter in the problem lacks a corresponding prior distribution.
    pub fn new(problem: Problem<S, L>, priors: HashMap<String, P>) -> Result<Self, String> {
        let mut sorted_priors = vec![];
        for param in problem.ode_system().get_sorted_params() {
            if let Some(prior) = priors.get(&param).cloned() {
                sorted_priors.push(prior);
            } else {
                return Err(MCMCError::NoPriorError(param.clone()).to_string());
            }
        }
        Ok(Self {
            problem,
            priors: sorted_priors,
        })
    }

    /// Spawns a background thread to consume samples and write them to output.
    ///
    /// # Arguments
    ///
    /// * `receiver` - Channel receiver for sample messages
    /// * `output` - Output handler implementing `SampleOutput`
    /// * `num_chains` - Number of chains to initialize
    ///
    /// # Returns
    ///
    /// A join handle for the consumer thread
    fn spawn_draw_consumer<O>(
        receiver: channel::Receiver<SampleMessage>,
        mut output: O,
        num_chains: usize,
    ) -> thread::JoinHandle<Result<O, MCMCError>>
    where
        O: SampleOutput + Send + 'static,
    {
        thread::spawn(move || {
            let chain_ids: Vec<String> = (0..num_chains).map(|i| format!("chain_{i}")).collect();
            let chain_refs: Vec<&str> = chain_ids.iter().map(|s| s.as_str()).collect();
            output.add_chains(&chain_refs)?;

            while let Ok(message) = receiver.recv() {
                output.add_draw(&message.sample, &message.chain_id.to_string())?;
            }
            Ok(output)
        })
    }

    /// Updates the progress bar for a single MCMC chain draw.
    ///
    /// # Arguments
    ///
    /// * `progress_bar` - Progress bar to update
    /// * `chain_id` - Unique identifier for this chain
    /// * `draw_idx` - Current draw index
    /// * `num_tune` - Number of tuning draws
    /// * `info` - NUTS sampler information containing step size and number of steps
    #[inline]
    fn update_progress_bar(
        progress_bar: &ProgressBar,
        chain_id: usize,
        draw_idx: u64,
        num_tune: u64,
        info: &nuts_rs::Progress,
    ) {
        let is_tuning = draw_idx < num_tune;
        let phase = if is_tuning { "Tuning" } else { "Sampling" };

        let step_size = info.step_size;
        let steps = info.num_steps;

        let message = if (0.001..1000.0).contains(&step_size) {
            format!("{phase} {steps} steps of size {step_size:.3}")
        } else {
            format!("{phase} {steps} steps of size {step_size:.2e}")
        };

        progress_bar.set_message(message);
        progress_bar.set_position(draw_idx + 1);

        if draw_idx == num_tune && num_tune > 0 {
            let style = ProgressStyle::default_bar()
                .template(&format!(
                    "Chain {chain_id:02}: {{spinner:.green}} [{{bar:40.green/blue}}] {{pos}}/{{len}} | {{elapsed}}/{{eta}} | {{msg}}"
                ))
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ");
            progress_bar.set_style(style);
        }
    }

    /// Runs a single MCMC chain with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `chain_id` - Unique identifier for this chain
    /// * `settings` - NUTS sampler settings
    /// * `total_draws` - Total number of draws (tuning + sampling)
    /// * `progress_bar` - Progress bar for this chain
    /// * `sender` - Channel sender for sample transmission
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the chain execution
    #[inline]
    fn run_single_chain(
        &self,
        chain_id: usize,
        settings: &DiagGradNutsSettings,
        total_draws: u64,
        progress_bar: ProgressBar,
        sender: channel::Sender<SampleMessage>,
    ) -> Result<usize, MCMCError> {
        let chain_context = ChainContext::new(self.clone());

        let initial_position = chain_context.bprob.draw_initial_position();
        let math = CpuMath::new(chain_context);

        let mut rng = ThreadRng::default();
        let mut sampler = settings.new_chain(chain_id as u64, math, &mut rng);

        sampler
            .set_position(&initial_position)
            .expect("Unrecoverable error during init");

        let num_tune = settings.num_tune;
        let chain_id_string = format!("chain_{chain_id}");
        let mut divergences = 0;

        for draw_idx in 0..total_draws {
            let (draw, info) = sampler.draw().expect("Unrecoverable error during sampling");

            if !info.tuning && !info.diverging {
                let message = SampleMessage {
                    chain_id: chain_id_string.clone(),
                    sample: draw.into_vec(),
                };

                sender.send(message).map_err(|_| MCMCError::ChannelClosed)?;
            } else if !info.tuning && info.diverging {
                divergences += 1;
            }

            Self::update_progress_bar(&progress_bar, chain_id, draw_idx, num_tune, &info);
        }

        progress_bar.finish_with_message(format!(
            "✅ Completed ({} tuning + {} sampling)",
            num_tune,
            total_draws - num_tune
        ));
        Ok(divergences)
    }

    /// Runs MCMC sampling with multiple chains in parallel.
    ///
    /// # Arguments
    ///
    /// * `output` - Output handler for storing samples
    /// * `num_draws` - Number of sampling draws per chain
    /// * `num_tune` - Number of tuning draws per chain
    /// * `maxdepth` - Maximum tree depth for NUTS (default: 6)
    /// * `seed` - Random seed for reproducibility (default: 0)
    /// * `num_chains` - Number of parallel chains (default: 1)
    /// * `num_parallel` - Number of parallel threads (default: all available)
    /// * `target_accept` - Target acceptance rate for adaptation
    ///
    /// # Returns
    ///
    /// The populated output handler containing all samples
    ///
    /// # Errors
    ///
    /// Returns an error if sampling fails or if invalid parameters are provided
    #[builder]
    pub fn run<O>(
        &self,
        output: O,
        num_draws: u64,
        num_tune: u64,
        maxdepth: Option<u64>,
        seed: Option<u64>,
        num_chains: Option<usize>,
        num_parallel: Option<i32>,
        target_accept: Option<f64>,
    ) -> Result<(O, usize), MCMCError>
    where
        O: SampleOutput + Send + 'static,
    {
        let num_parallel = self.determine_num_parallel(num_parallel)?;
        let num_chains = num_chains.unwrap_or(1);

        let mut settings = DiagGradNutsSettings {
            num_tune,
            num_draws,
            maxdepth: maxdepth.unwrap_or(6),
            seed: seed.unwrap_or(0),
            num_chains,
            store_gradient: false,
            store_unconstrained: false,
            ..Default::default()
        };

        if let Some(target_accept) = target_accept {
            settings.adapt_options.dual_average_options.target_accept = target_accept;
        }

        let total_draws = num_tune + num_draws;

        println!("Running {num_chains} chains using {num_parallel} parallel threads");

        let (sender, receiver) = channel::bounded(1000);

        let consumer_handle = Self::spawn_draw_consumer(receiver, output, num_chains);

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_parallel)
            .build()
            .map_err(|e| MCMCError::ThreadPoolError(e.to_string()))?;

        let progress_bars = self.setup_progress_bars(num_chains, total_draws);

        let chain_results: Result<Vec<usize>, MCMCError> = thread_pool.install(|| {
            progress_bars
                .into_par_iter()
                .enumerate()
                .map(|(chain_id, progress_bar)| {
                    self.run_single_chain(
                        chain_id,
                        &settings,
                        total_draws,
                        progress_bar,
                        sender.clone(),
                    )
                })
                .collect()
        });

        let divergences: usize = chain_results?.iter().sum();

        drop(sender);

        let final_output = consumer_handle
            .join()
            .map_err(|_| MCMCError::ThreadPoolError("Consumer thread panicked".to_string()))??;

        for chain_id in 0..num_chains {
            println!("Chain {chain_id} completed with {total_draws} draws");
        }

        println!("All chains completed successfully!");
        Ok((final_output, divergences))
    }

    /// Determines the number of parallel threads to use.
    fn determine_num_parallel(&self, num_parallel: Option<i32>) -> Result<usize, MCMCError> {
        let available_threads = std::thread::available_parallelism().unwrap().get();

        match num_parallel.unwrap_or(-1) {
            -1 => Ok(available_threads),
            n if n <= 0 => Err(MCMCError::InvalidParallelism(n)),
            n => {
                let n = n as usize;
                if n > available_threads {
                    Err(MCMCError::TooManyThreads {
                        requested: n,
                        available: available_threads,
                    })
                } else {
                    Ok(n)
                }
            }
        }
    }

    /// Sets up progress bars for all chains.
    fn setup_progress_bars(&self, num_chains: usize, total_draws: u64) -> Vec<ProgressBar> {
        let multi_progress = MultiProgress::new();
        (0..num_chains)
            .map(|chain_id| {
                let pb = multi_progress.add(ProgressBar::new(total_draws));
                let style = ProgressStyle::default_bar()
                    .template(&format!(
                        "Chain {chain_id:02}: {{spinner:.green}} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} | {{elapsed}}/{{eta}} | {{msg}}"
                    ))
                    .unwrap()
                    .progress_chars("█▉▊▋▌▍▎▏ ");
                pb.set_style(style);
                pb.set_message("Initializing...");
                pb
            })
            .collect()
    }

    /// Computes the log-likelihood for given parameters.
    #[inline(always)]
    fn likelihood(&self, parameters: &[f64]) -> f64 {
        let (residuals, _) = self
            .problem
            .get_residuals(parameters, Some(Mode::Regular))
            .unwrap();
        self.problem.objective().log_likelihood(&residuals)
    }

    /// Computes the gradient of the log-likelihood using finite differences.
    #[inline(always)]
    fn grad_likelihood(&self, parameters: &Array1<f64>) -> Array1<f64> {
        parameters.central_diff(&|x| self.likelihood(x.as_slice().unwrap()))
    }

    /// Draws initial parameter values from the prior distributions.
    fn draw_initial_position(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        self.priors
            .iter()
            .map(|prior| prior.sample(&mut rng))
            .collect()
    }

    /// Returns the sorted parameter names from the underlying problem.
    pub fn get_sorted_params(&self) -> Vec<String> {
        self.problem.ode_system().get_sorted_params()
    }

    /// Creates a new ChainContext for this problem with pre-allocated buffers.
    pub fn create_chain_context(self) -> ChainContext<S, L, P> {
        ChainContext::new(self)
    }
}

/// Implementation of the `CpuLogpFunc` trait for NUTS sampling.
///
/// This implementation allows the `BayesianProblem` to be used with the `nuts-rs` crate
/// for efficient MCMC sampling using the No-U-Turn Sampler algorithm.
impl<S, L, P> CpuLogpFunc for ChainContext<S, L, P>
where
    S: ODEIntegrator + Copy + Send + Sync,
    L: ObjectiveFunction + Likelihood,
    P: DiffablePrior + Distribution<f64>,
{
    type LogpError = MCMCError;
    type TransformParams = ();

    /// Returns the dimensionality of the parameter space.
    #[inline(always)]
    fn dim(&self) -> usize {
        self.bprob.problem.get_n_params()
    }

    /// Computes the log posterior probability and its gradient.
    ///
    /// This method evaluates both the log-likelihood (from the optimization problem)
    /// and log-prior (from the prior distributions), then computes their sum
    /// (the log posterior) along with the gradient.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Current parameter values to evaluate
    /// * `grad` - Mutable slice to store the computed gradient
    ///
    /// # Returns
    ///
    /// The log posterior probability (log-likelihood + log-prior)
    #[inline(always)]
    fn logp(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        unsafe {
            self.param_buffer
                .as_slice_mut()
                .unwrap_unchecked()
                .copy_from_slice(parameters);
        }

        let likelihood = self.bprob.likelihood(parameters);
        let mut prior = 0.0;
        for (prior_dist, &param) in self.bprob.priors.iter().zip(parameters.iter()) {
            prior += prior_dist.ln_pdf(param);
        }

        let posterior = likelihood + prior;

        self.likelihood_grad_buffer
            .assign(&self.bprob.grad_likelihood(&self.param_buffer));

        let likelihood_grad_slice =
            unsafe { self.likelihood_grad_buffer.as_slice().unwrap_unchecked() };
        for (i, (prior_dist, &param)) in self.bprob.priors.iter().zip(parameters.iter()).enumerate()
        {
            unsafe {
                *grad.get_unchecked_mut(i) =
                    likelihood_grad_slice.get_unchecked(i) + prior_dist.ln_pdf_grad(param);
            }
        }

        Ok(posterior)
    }
}

#[cfg(test)]
mod tests {
    use peroxide::fuga::RK4;
    use statrs::distribution::Uniform;

    use crate::{io::load_enzmldoc, mcmc::likelihood::NormalLikelihood, optim::ProblemBuilder};

    use super::*;

    #[test]
    fn test_bayesian_problem() {
        let doc = load_enzmldoc("tests/data/enzmldoc.json").unwrap();
        let likelihood = NormalLikelihood::new(2.0);
        let problem = ProblemBuilder::new(&doc, RK4, likelihood)
            .dt(100.0)
            .build()
            .expect("Failed to build problem");

        let priors = HashMap::from([
            ("k_cat".to_string(), Uniform::new(0.10, 1.20).unwrap()),
            ("k_ie".to_string(), Uniform::new(0.0004, 0.005).unwrap()),
            ("K_M".to_string(), Uniform::new(60.0, 150.0).unwrap()),
        ]);

        let problem = BayesianProblem::new(problem, priors).unwrap();

        use crate::mcmc::output::CSVOutput;
        let output = CSVOutput::new("./output", problem.get_sorted_params()).unwrap();

        problem
            .run()
            .output(output)
            .num_tune(10)
            .num_draws(10)
            .maxdepth(6)
            .num_chains(2)
            .num_parallel(2)
            .seed(0)
            .call()
            .unwrap();
    }
}
