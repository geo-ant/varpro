use snafu::Snafu;

/// An error structure containing error variants related to
/// the model and the set of model
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum ModelError {
    #[snafu(display("Parameter list {:?} contains duplicates! Parameter lists must comprise only unique elements.",function_parameters))]
    DuplicateParameterNames {
        function_parameters : Vec<String>,
    },

    #[snafu(display("A function or model parameter list is empty! It must at least contain one parameter."))]
    EmptyParameters,

    #[snafu(display("Function parameter '{}' is not part of the model parameters.", function_parameter))]
    FunctionParameterNotInModel {
        function_parameter : String
    },

    #[snafu(display(
        "Parameter '{}' for derivative does not exist in parameter list '{:?}'.",
        parameter,
        function_parameters
    ))]
    InvalidDerivative {
        parameter: String,
        function_parameters: Vec<String>,
    },

    #[snafu(display("Derivative for parameter '{}' was already provided! Give each partial derivative exactly once.", parameter))]
    DuplicateDerivative { parameter: String },

    #[snafu(display("Function with paramter list {:?} is missing derivative for parametr '{}'.", function_parameters,missing_parameter))]
    MissingDerivative {
        missing_parameter : String,
        function_parameters : Vec<String>,
    },

    #[snafu(display("Fatal error: '{}'. This indicates a programming error in the library",message))]
    Fatal {
        message : String
    },
}
