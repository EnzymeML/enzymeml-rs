use argmin::core::{observers::Observe, Error, State, KV};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::panic::UnwindSafe;

/// A custom observer that allows passing a callback function to monitor optimization progress.
///
/// The `CallbackObserver` implements the `Observe` trait from the argmin library and provides
/// a way to execute custom logic during the optimization process. The callback function is called
/// after each iteration with the current cost and best cost values.
pub struct CallbackObserver {
    /// The callback function to be executed after each iteration.
    /// Takes two f64 parameters:
    /// - First parameter: Current cost value of the iteration
    /// - Second parameter: Best cost value found so far
    pub callback: Box<dyn Fn(f64, f64) + UnwindSafe + Send>,
}

impl<I> Observe<I> for CallbackObserver
where
    I: State,
{
    /// Called when optimization is initialized. Currently does nothing.
    ///
    /// # Arguments
    /// * `_msg` - Initialization message (unused)
    /// * `_state` - Current optimization state (unused)
    /// * `_kv` - Key-value storage (unused)
    fn observe_init(&mut self, _msg: &str, _state: &I, _kv: &KV) -> Result<(), Error> {
        Ok(())
    }

    /// Called after each optimization iteration.
    /// Extracts the current cost and best cost values and passes them to the callback function.
    ///
    /// # Arguments
    /// * `state` - Current optimization state containing cost values
    /// * `_kv` - Key-value storage (unused)
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        let cost = state.get_cost().to_string();
        let best_cost = state.get_best_cost().to_string();
        (self.callback)(
            cost.parse::<f64>().unwrap(),
            best_cost.parse::<f64>().unwrap(),
        );
        Ok(())
    }
}

impl CallbackObserver {
    pub fn new(callback: Box<dyn Fn(f64, f64) + UnwindSafe + Send>) -> Self {
        Self { callback }
    }
}

/// A progress bar observer that displays optimization progress.
///
/// The `ProgressObserver` implements the `Observe` trait from the argmin library and
/// provides a visual progress bar to monitor the optimization process. It shows the
/// current iteration and the best cost value found so far.
pub struct ProgressObserver {
    /// The progress bar used to display optimization progress
    pub pb: ProgressBar,
    /// The name of the optimizer
    pub name: String,
}

impl ProgressObserver {
    /// Creates a new ProgressObserver with the specified total number of iterations.
    ///
    /// # Arguments
    /// * `total` - The total number of iterations expected
    ///
    /// # Returns
    /// A new ProgressObserver instance with configured progress bar
    pub fn new(total: u64, name: &str) -> Self {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
            .template(&format!(
                "Fitting {name}: {{spinner:.green}} [{{bar:40.green/blue}}] {{pos}}/{{len}} | {{elapsed}}/{{eta}} | {{msg}}"
            ))
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        Self {
            pb,
            name: name.to_string(),
        }
    }
}

impl<I> Observe<I> for ProgressObserver
where
    I: State,
{
    /// Called when optimization is initialized.
    /// Resets the progress bar to zero.
    ///
    /// # Arguments
    /// * `_msg` - Initialization message (unused)
    /// * `_state` - Current optimization state (unused)
    /// * `_kv` - Key-value storage (unused)
    fn observe_init(&mut self, _msg: &str, _state: &I, _kv: &KV) -> Result<(), Error> {
        println!("\nFitting {}", self.name.bold().cyan());
        self.pb.reset();
        Ok(())
    }

    /// Called after each optimization iteration.
    /// Updates the progress bar position and displays the current best cost.
    ///
    /// # Arguments
    /// * `state` - Current optimization state containing iteration count and cost values
    /// * `_kv` - Key-value storage (unused)
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        self.pb.set_position(state.get_iter());
        let best_cost = state.get_best_cost().to_string();
        self.pb
            .set_message(format!("{:.6}", best_cost.parse::<f64>().unwrap_or(0.0)));
        Ok(())
    }
}
