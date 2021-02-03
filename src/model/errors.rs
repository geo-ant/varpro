use snafu::Snafu;

/// An error structure containing error variants related to
/// the model and the set of model
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum Error {
    #[snafu(display("List of parameter contains duplictates."))]
    DuplicateParameterNames,

    #[snafu(display("Parameter list for a model function with derivatives must not be empty."))]
    EmptyParameters,

    #[snafu(display("Subset contains parameters that are not in the full set"))]
    FunctionParameterNotInModel,

    #[snafu(display("Parameter '{}' for derivative not in set '{:?}'",parameter,function_parameters))]
    InvalidDerivative {
        parameter: String,
        function_parameters: Vec<String>,
    }

}