use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObjectiveError {
    /// Error indicating incompatible array shapes for operations
    #[error("Could not broadcast array for loss function")]
    ShapeError,
}
