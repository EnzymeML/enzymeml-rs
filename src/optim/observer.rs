use argmin::core::{observers::Observe, Error, State, KV};
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
