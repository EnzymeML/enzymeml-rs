use nuts_rs::LogpError;
use thiserror::Error;

/// Error types that can occur during posterior evaluation.
///
/// This enum defines the possible errors that can occur when evaluating the log posterior
/// probability and its gradient during MCMC sampling.
#[derive(Debug, Error)]
pub enum MCMCError {
    /// Error occurred during cost function evaluation.
    ///
    /// This can happen if:
    /// - The ODE integration fails
    /// - Parameter values are outside valid bounds
    /// - Numerical issues occur during likelihood computation
    #[error("Cost function evaluation failed")]
    CostError,

    #[error("No prior found for parameter: {0}")]
    NoPriorError(String),

    #[error("Invalid number of parallel threads requested: {0}")]
    InvalidParallelism(i32),

    #[error("Requested more parallel threads than available: requested={requested}, available={available}")]
    TooManyThreads { requested: usize, available: usize },

    #[error("Failed to initialize thread pool: {0}")]
    ThreadPoolError(String),

    #[error("Failed to create DataFrame: {0}")]
    DataFrameError(#[from] polars::error::PolarsError),

    #[error("Failed to extend series: {0}")]
    SeriesError(String),

    #[error("Sample output channel closed unexpectedly")]
    ChannelClosed,

    #[error("IO operation failed: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Chain {0} not found")]
    ChainNotFound(String),
}

/// Implementation of the `LogpError` trait for error handling in NUTS.
///
/// This implementation defines how errors should be handled during MCMC sampling.
impl LogpError for MCMCError {
    /// Determines whether the error is recoverable.
    ///
    /// Currently, all errors are considered non-recoverable, meaning the sampler
    /// will terminate if any error occurs. In future versions, this could be
    /// made more sophisticated to handle certain types of recoverable errors.
    ///
    /// # Returns
    ///
    /// `false` - All errors are currently treated as non-recoverable
    fn is_recoverable(&self) -> bool {
        true
    }
}
