use snafu::Snafu;

/// An error structure containing error variants related to
/// the model and the set of model
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum ModelfunctionError {
    #[snafu(display("List of parameter contains duplictates."))]
    DuplicateParameterNames,

    #[snafu(display("List of parameters is empty."))]
    EmptyParameters,

    #[snafu(display("Subset contains parameters that are not in the full set"))]
    InvalidParametersInSubset,
}

/// An error structure containing error variants related to
/// the model and the set of model
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum ModelBuilderError {
    #[snafu(context(false))]
    ModelfunctionError {
        source : ModelfunctionError,
    }
}