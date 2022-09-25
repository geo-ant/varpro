use snafu::Snafu;

/// Errors pertaining to use errors of the [SeparableModel].
#[derive(Debug, Clone, Snafu, PartialEq, Eq)]
#[snafu(visibility(pub))]
pub enum ModelError {
    /// Base functions are expected to return a vector the same length as the location argument.
    /// A function did not adhere to that rule.
    #[snafu(display(
        "Base function gave vector of length {}, but expected output length was {}",
        actual_length,
        expected_length
    ))]
    UnexpectedFunctionOutput {
        expected_length: usize,
        actual_length: usize,
    },

    /// Indicates an evaluation for a parameter was requested that is not part of the model parameters
    #[snafu(display("Parameter '{}' is not in model", parameter))]
    ParameterNotInModel { parameter: String },

    /// Indicates that the given derivative index is out of bounds.
    #[snafu(display("Index {} for derivative is out of bounds", index))]
    DerivativeIndexOutOfBounds { index: usize },

    /// It was tried to evaluate a model with an incorrect number of parameters
    #[snafu(display("Model takes {} parameters, but {} were provide.", required, actual))]
    IncorrectParameterCount { required: usize, actual: usize },
}
