use crate::model::SeparableModel;
use crate::solvers::levmar::LevMarLeastSquaresProblem;
use nalgebra::{ComplexField, DVector, Dynamic, Scalar, SVD};
use num_traits::{Float, Zero};
use snafu::Snafu;
use levenberg_marquardt::LeastSquaresProblem;

/// Errors pertaining to use errors of the [LeastSquaresProblemBuilder]
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum LeastSquaresProblemBuilderError {
    /// the data for the independent variable was not given to the builder
    #[snafu(display("Data for vector x not provided."))]
    XDataMissing,

    /// the data for y variable was not given to the builder
    #[snafu(display("Data for vector y not provided."))]
    YDataMissing,

    /// no model was provided to the builder
    #[snafu(display("Model not provided."))]
    ModelMissing,

    /// no model was provided to the builder
    #[snafu(display("Initial guess for parameters not provided."))]
    InitialGuessMissing,

    /// x and y vector have different lengths
    #[snafu(display(
        "Vectors x and y must have same lengths. Given x length = {} and y length = {}",
        x_length,
        y_length
    ))]
    InvalidLengthOfData { x_length: usize, y_length: usize },

    /// the provided x and y vectors must not have zero elements
    #[snafu(display("x or y must have nonzero number of elements."))]
    InvalidZeroLength,

    /// the model has a different number of parameters than the provided initial guesses
    #[snafu(display("Initial guess vector must have same length as parameters. Model has {} parameters and {} initial guesses were provided.",model_count, provided_count))]
    InvalidParameterCount {
        model_count: usize,
        provided_count: usize,
    },
}

pub struct LevMarLeastSquaresProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField ,
    ScalarType::RealField : Float
{
    /// Required: the independent variable `$\vec{x}$
    x: Option<DVector<ScalarType>>,
    /// Required: the data `$\vec{y}(\vec{x})$` that we want to fit
    y: Option<DVector<ScalarType>>,
    /// Required: the model to be fitted to the data
    model: Option<&'a SeparableModel<ScalarType>>,
    /// Required: the initial guess for the model parameters
    parameter_initial_guess: Option<Vec<ScalarType>>,
    /// Optional: set epsilon below which two singular values
    /// are considered zero.
    /// if this is not given, when building the builder,
    /// the this is set to machine epsilon.
    epsilon: Option<ScalarType::RealField>,
}

/// Same as `Self::new()`.
impl<'a,ScalarType> Default for LevMarLeastSquaresProblemBuilder<'a,ScalarType>
    where
        ScalarType: Scalar + ComplexField + Zero,
        ScalarType::RealField : Float
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, ScalarType> LevMarLeastSquaresProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Zero,
ScalarType::RealField : Float
{
    /// Create a new builder with empty fields
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            model: None,
            parameter_initial_guess: None,
            epsilon: None,
        }
    }

    /// **Mandatory**: Set the value of the independent variable `$\vec{x}$` of the problem
    /// The length of `$\vec{x}$` and the data `$\vec{y}$` must be the same.
    pub fn x(self, xvec: DVector<ScalarType>) -> Self {
        Self {
            x: Some(xvec),
            ..self
        }
    }

    /// **Mandatory**: Set the data to fit with `$\vec{y}=\vec{y}(\vec{x})$` of the problem
    /// The length of `$\vec{x}$` and the data `$\vec{y}$` must be the same.
    pub fn y(self, yvec: DVector<ScalarType>) ->Self  {
        Self {
            y: Some(yvec),
            ..self
        }
    }

    /// **Mandatory**: Set the initial guesses of the model parameters
    /// The parameter must have the same length as the model parameters
    pub fn model(self, model: &'a SeparableModel<ScalarType>) -> Self
    {
        Self {
            model: Some(model),
            ..self
        }

    }

    /// **Mandatory**: provide initial guess for the parameters
    /// they must have the same number of elements as the model parameters
    pub fn initial_guess(self, params : Vec<ScalarType>) -> Self{
        Self {
            parameter_initial_guess: Some(params),
            ..self
        }
    }

    /// **Optional** set epsilon below which two singular values are considered zero
    /// If this value is not given, it will be set to machine epsilon. The epsilon is
    /// automatically converted to a non-negative number. This value
    /// can be used to mitigate the effects of basis functions becoming linearly dependent
    /// when model parameters align in an unfortunate way. Then set it to some small factor
    /// times the machine precision.
    pub fn epsilon(self, eps: ScalarType::RealField) -> Self {
        Self {
            epsilon: Some(<ScalarType::RealField as Float>::abs(eps)),
            ..self
        }
    }


    /// build the least squares problem from the builder.
    /// # Prerequisites
    /// * All mandatory parameters have been set (see individual builder methods for details)
    /// * `$\vec{x}$` and `$\vec{y}$` have the same number of elements
    /// * `$\vec{x}$` and `$\vec{y}$` have a nonzero number of elements
    /// * the length of the initial guesses vector is the same as the number of model parameters
    /// # Returns
    /// If all prerequisites are fullfilled, returns a [LevMarLeastSquaresProblem] with the given
    /// content and the parameters set to the initial guess. Otherwise returns an error variant.
    pub fn build(self) -> Result<LevMarLeastSquaresProblem<'a, ScalarType>, LeastSquaresProblemBuilderError> {
        let finalized_builder = self.finalize()?;

        let param_count = finalized_builder.model.parameter_count();
        let data_length = finalized_builder.y.len();

        let mut problem = LevMarLeastSquaresProblem {
            // these parameters all come from the builder
            location: finalized_builder.x,
            data: finalized_builder.y,
            model: finalized_builder.model,
            svd_epsilon: finalized_builder.epsilon,
            // these parameters are bs, but they will be overwritten immediately
            // by the call to set_params
            parameters: Vec::new(),
            current_residuals: DVector::from_element(data_length,Zero::zero()),
            current_svd: SVD {
                u: None,
                v_t: None,
                singular_values: DVector::from_element(data_length,Float::epsilon()),
            },
            current_linear_coeffs: DVector::from_element(param_count,Zero::zero()),
        };
        problem.set_params(&DVector::from(finalized_builder.parameter_initial_guess));
        Ok(problem)
    }
}

/// helper structure that has the same fields as the builder
/// but none of them are optional
struct FinalizedBuilder<'a,ScalarType>
    where
        ScalarType: Scalar + ComplexField ,
        ScalarType::RealField : Float
{
    x: DVector<ScalarType>,
    y: DVector<ScalarType>,
    model: &'a SeparableModel<ScalarType>,
    parameter_initial_guess: Vec<ScalarType>,
    epsilon: ScalarType::RealField,
}

// private implementations
impl<'a, ScalarType> LevMarLeastSquaresProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField ,
    ScalarType::RealField : Float
{
    /// helper function to check if all required fields have been set and pass the checks
    /// if so, this returns a destructured result of self
    fn finalize(self) -> Result<FinalizedBuilder<'a,ScalarType>, LeastSquaresProblemBuilderError> {
        match self {
            // in this case all required fields are set to something
            LevMarLeastSquaresProblemBuilder {
                x: Some(x),
                y: Some(y),
                model: Some(model),
                parameter_initial_guess: Some(parameter_initial_guess),
                epsilon,
            } => {
                if x.len() != y.len() {
                    Err(LeastSquaresProblemBuilderError::InvalidLengthOfData {
                        x_length: x.len(),
                        y_length: y.len(),
                    })
                } else if x.len() == 0 || y.len() == 0 {
                    Err(LeastSquaresProblemBuilderError::InvalidZeroLength)
                } else if model.parameter_count() != parameter_initial_guess.len() {
                    Err(LeastSquaresProblemBuilderError::InvalidParameterCount {
                        model_count: model.parameter_count(),
                        provided_count: parameter_initial_guess.len(),
                    })
                } else {
                    Ok( FinalizedBuilder {
                        x,
                        y,
                        model,
                        parameter_initial_guess,
                        epsilon: epsilon.unwrap_or(Float::epsilon()),
                    })
                }
            }
            // not all required fields are set, report missing parameters
            LevMarLeastSquaresProblemBuilder {
                x,
                y,
                model,
                parameter_initial_guess,
                epsilon,
            } => {
                if x.is_none() {
                    Err(LeastSquaresProblemBuilderError::XDataMissing)
                } else if y.is_none() {
                    Err(LeastSquaresProblemBuilderError::YDataMissing)
                } else if model.is_none() {
                    Err(LeastSquaresProblemBuilderError::ModelMissing)
                } else if parameter_initial_guess.is_none() {
                    Err(LeastSquaresProblemBuilderError::InitialGuessMissing)
                } else {
                    unreachable!()
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::solvers::levmar::LevMarLeastSquaresProblemBuilder;
    #[test]
    fn todo_add_tests_for_builder() {
        let builder = LevMarLeastSquaresProblemBuilder::<f64>::new();
        todo!();
    }
}