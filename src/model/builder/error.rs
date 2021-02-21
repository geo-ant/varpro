use snafu::Snafu;

/// An error structure that contains error variants that occur when building a model.
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum ModelBuildError {
    /// Model or function parameters contain duplicates
    #[snafu(display("Parameter list {:?} contains duplicates! Parameter lists must comprise only unique elements.",function_parameters))]
    DuplicateParameterNames { function_parameters: Vec<String> },

    /// Model or function parameter list is empty. To add functions that are independent of
    /// model parameters, use the interface for adding invariant functions.
    #[snafu(display(
        "A function or model parameter list is empty! It must at least contain one parameter."
    ))]
    EmptyParameters,

    /// A function was added to the model which depends on parameters which are not in the model
    #[snafu(display(
        "Function parameter '{}' is not part of the model parameters.",
        function_parameter
    ))]
    FunctionParameterNotInModel { function_parameter: String },

    /// Tried to provide a partial derivative with respect to a parameter that a function does
    /// not depend on
    #[snafu(display(
        "Parameter '{}' given for partial derivative does not exist in parameter list '{:?}'.",
        parameter,
        function_parameters
    ))]
    InvalidDerivative {
        parameter: String,
        function_parameters: Vec<String>,
    },

    /// Tried to provide the same partial derivative twice.
    #[snafu(display("Derivative for parameter '{}' was already provided! Give each partial derivative exactly once.", parameter))]
    DuplicateDerivative { parameter: String },

    /// Not all partial derivatives for a function where given. Each function must be given
    /// a partial derivative with respect to each parameter it depends on.
    #[snafu(display(
        "Function with paramter list {:?} is missing derivative for parametr '{}'.",
        function_parameters,
        missing_parameter
    ))]
    MissingDerivative {
        missing_parameter: String,
        function_parameters: Vec<String>,
    },

    /// Tried to construct a model without base functions
    #[snafu(display(
        "Tried to construct model with no functions. A model must contain at least one function."
    ))]
    EmptyModel,

    /// The model depends on a certain parameter that none of the base functions depend on.
    #[snafu(display("Model depends on parameter '{}', but none of its functions use it. Each model parameter must occur in at least one function.",parameter))]
    UnusedParameter { parameter: String },

    /// This error indicates that the more or fewer string parameters where provided as function
    /// parameters than the actual variadic function takes. This might accidentally happen when giving
    /// a derivative that does not depend on a certain parameter, whereas the base function does.
    /// However, the library requires that the derivatives take all the parameters its base function
    /// takes in the same order.
    #[snafu(display(
        "Incorrect parameter count: Given function parameters '{:?}' have length {}, but the provided function takes {} parameter arguments.",
        params,
        string_params_count,
        function_argument_count,
    ))]
    IncorrectParameterCount {
        params: Vec<String>,
        string_params_count: usize,
        function_argument_count: usize,
    },

    /// Parameter names may not contain a comma separator, because this is most likely caused by a typo, i.e.
    /// `["tau,phi"]`, instead of actually `["tau","phi"]`. So this is forbidden in order to help spotting these
    /// hard to find errors.
    #[snafu(display("Parameter names may not contain comma separator: '{}'. Did you want to give two parameters?",param_name))]
    CommaInParameterNameNotAllowed { param_name: String },
}
