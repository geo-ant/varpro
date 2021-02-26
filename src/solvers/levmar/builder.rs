use crate::model::SeparableModel;
use crate::solvers::levmar::LevMarProblem;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, DVector, Scalar, SVD};
use num_traits::{Float, Zero};
use snafu::Snafu;
use std::ops::Mul;

/// Errors pertaining to use errors of the [LeastSquaresProblemBuilder]
#[derive(Debug, Clone, Snafu, PartialEq)]
#[snafu(visibility = "pub")]
pub enum LevMarBuilderError {
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
    ZeroLengthVector,

    /// the model has a different number of parameters than the provided initial guesses
    #[snafu(display("Initial guess vector must have same length as parameters. Model has {} parameters and {} initial guesses were provided.",model_count, provided_count))]
    InvalidParameterCount {
        model_count: usize,
        provided_count: usize,
    },
}

#[derive(Clone)]
pub struct LevMarBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    /// Required: the independent variable `$\vec{x}$
    x: Option<DVector<ScalarType>>,
    /// Required: the data `$\vec{y}(\vec{x})$` that we want to fit
    y: Option<DVector<ScalarType>>,
    /// Required: the model to be fitted to the data
    separable_model: Option<&'a SeparableModel<ScalarType>>,
    /// Required: the initial guess for the model parameters
    parameter_initial_guess: Option<Vec<ScalarType>>,
    /// Optional: set epsilon below which two singular values
    /// are considered zero.
    /// if this is not given, when building the builder,
    /// the this is set to machine epsilon.
    epsilon: Option<ScalarType::RealField>,
}

/// Same as `Self::new()`.
impl<'a, ScalarType> Default for LevMarBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Zero,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, ScalarType> LevMarBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Zero,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    /// Create a new builder with empty fields
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            separable_model: None,
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
    pub fn y(self, yvec: DVector<ScalarType>) -> Self {
        Self {
            y: Some(yvec),
            ..self
        }
    }

    /// **Mandatory**: Set the initial guesses of the model parameters
    /// The parameter must have the same length as the model parameters
    pub fn model(self, model: &'a SeparableModel<ScalarType>) -> Self {
        Self {
            separable_model: Some(model),
            ..self
        }
    }

    /// **Mandatory**: provide initial guess for the parameters
    /// they must have the same number of elements as the model parameters
    pub fn initial_guess(self, params: &[ScalarType]) -> Self {
        Self {
            parameter_initial_guess: Some(params.to_vec()),
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
    pub fn build(self) -> Result<LevMarProblem<'a, ScalarType>, LevMarBuilderError> {
        let finalized_builder = self.finalize()?;

        let param_count = finalized_builder.model.parameter_count();
        let data_length = finalized_builder.y.len();

        let mut problem = LevMarProblem {
            // these parameters all come from the builder
            x: finalized_builder.x,
            y: finalized_builder.y,
            model: finalized_builder.model,
            svd_epsilon: finalized_builder.epsilon,
            // these parameters are bs, but they will be overwritten immediately
            // by the call to set_params
            model_parameters: Vec::new(),
            current_residuals: DVector::from_element(data_length, Zero::zero()),
            current_svd: SVD {
                u: None,
                v_t: None,
                singular_values: DVector::from_element(data_length, Float::epsilon()),
            },
            linear_coefficients: DVector::from_element(param_count, Zero::zero()),
        };
        problem.set_params(&DVector::from(finalized_builder.parameter_initial_guess));
        Ok(problem)
    }
}

/// helper structure that has the same fields as the builder
/// but none of them are optional
struct FinalizedBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField,
    ScalarType::RealField: Float,
{
    x: DVector<ScalarType>,
    y: DVector<ScalarType>,
    model: &'a SeparableModel<ScalarType>,
    parameter_initial_guess: Vec<ScalarType>,
    epsilon: ScalarType::RealField,
}

// private implementations
impl<'a, ScalarType> LevMarBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    /// helper function to check if all required fields have been set and pass the checks
    /// if so, this returns a destructured result of self
    fn finalize(self) -> Result<FinalizedBuilder<'a, ScalarType>, LevMarBuilderError> {
        match self {
            // in this case all required fields are set to something
            LevMarBuilder {
                x: Some(x),
                y: Some(y),
                separable_model: Some(model),
                parameter_initial_guess: Some(parameter_initial_guess),
                epsilon,
            } => {
                if x.len() != y.len() {
                    Err(LevMarBuilderError::InvalidLengthOfData {
                        x_length: x.len(),
                        y_length: y.len(),
                    })
                } else if x.is_empty() || y.is_empty() {
                    Err(LevMarBuilderError::ZeroLengthVector)
                } else if model.parameter_count() != parameter_initial_guess.len() {
                    Err(LevMarBuilderError::InvalidParameterCount {
                        model_count: model.parameter_count(),
                        provided_count: parameter_initial_guess.len(),
                    })
                } else {
                    Ok(FinalizedBuilder {
                        x,
                        y,
                        model,
                        parameter_initial_guess,
                        epsilon: epsilon.unwrap_or_else(Float::epsilon),
                    })
                }
            }
            // not all required fields are set, report missing parameters
            LevMarBuilder {
                x,
                y,
                separable_model: model,
                parameter_initial_guess,
                epsilon: _,
            } => {
                if x.is_none() {
                    Err(LevMarBuilderError::XDataMissing)
                } else if y.is_none() {
                    Err(LevMarBuilderError::YDataMissing)
                } else if model.is_none() {
                    Err(LevMarBuilderError::ModelMissing)
                } else if parameter_initial_guess.is_none() {
                    Err(LevMarBuilderError::InitialGuessMissing)
                } else {
                    unreachable!()
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use crate::solvers::levmar::builder::LevMarBuilderError;
    use crate::solvers::levmar::LevMarBuilder;
    use crate::test_helpers::get_double_exponential_model_with_constant_offset;
    use nalgebra::DVector;

    #[test]
    fn new_builder_starts_with_empty_fields() {
        let builder = LevMarBuilder::<f64>::new();
        let LevMarBuilder {
            x,
            y,
            separable_model: model,
            parameter_initial_guess,
            epsilon,
        } = builder;
        assert!(x.is_none());
        assert!(y.is_none());
        assert!(model.is_none());
        assert!(parameter_initial_guess.is_none());
        assert!(epsilon.is_none());
    }

    #[test]
    #[allow(clippy::float_cmp)] //clippy moans, but it's wrong (again!)
    fn builder_assigns_fields_correctly() {
        let model = get_double_exponential_model_with_constant_offset();
        // octave x = linspace(0,10,11);
        let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        //octave y = 2*exp(-t/2)+exp(-t/4)+1;
        let y = DVector::from(vec![
            4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
        ]);
        let initial_guess = vec![1., 2.];

        // build a problem with default epsilon
        let builder = LevMarBuilder::new()
            .x(x.clone())
            .y(y.clone())
            .initial_guess(initial_guess.as_slice())
            .model(&model);
        let problem = builder
            .clone()
            .build()
            .expect("Valid builder should not fail build");

        assert_eq!(problem.x, x);
        assert_eq!(problem.y, y);
        assert_eq!(problem.model_parameters, initial_guess);
        //assert!(problem.model.as_ref().as_ptr()== model.as_ptr()); // this don't work, boss
        assert_eq!(problem.svd_epsilon, f64::EPSILON); //clippy moans, but it's wrong (again!)
        assert!(
            problem.current_svd.u.is_some(),
            "SVD U must have been calculated on initialization"
        );
        assert!(
            problem.current_svd.v_t.is_some(),
            "SVD V^T must have been calculated on initialization"
        );

        // now check that a given epsilon is also passed through to the model
        let problem = builder
            .epsilon(-1.0) // check that negative values are converted to absolutes
            .build()
            .expect("Valid builder should not fail");
        assert_eq!(problem.svd_epsilon, 1.0);
    }

    #[test]
    fn builder_gives_errors_for_missing_mandatory_parameters() {
        let model = get_double_exponential_model_with_constant_offset();
        // octave x = linspace(0,10,11);
        let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        //octave y = 2*exp(-t/2)+exp(-t/4)+1;
        let y = DVector::from(vec![
            4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
        ]);
        let initial_guess = vec![1., 2.];

        let _builder = LevMarBuilder::new()
            .x(x.clone())
            .y(y.clone())
            .initial_guess(initial_guess.as_slice())
            .model(&model);

        assert!(matches!(
            LevMarBuilder::new()
                .y(y.clone())
                .initial_guess(initial_guess.as_slice())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::XDataMissing)
        ));
        assert!(matches!(
            LevMarBuilder::new()
                .x(x.clone())
                .initial_guess(initial_guess.as_slice())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::YDataMissing)
        ));
        assert!(matches!(
            LevMarBuilder::new()
                .x(x.clone())
                .y(y.clone())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::InitialGuessMissing)
        ));
        assert!(matches!(
            LevMarBuilder::new()
                .x(x)
                .y(y)
                .initial_guess(initial_guess.as_slice())
                .build(),
            Err(LevMarBuilderError::ModelMissing)
        ));
    }

    #[test]
    fn builder_gives_errors_for_semantically_wrong_parameters() {
        let model = get_double_exponential_model_with_constant_offset();
        // octave x = linspace(0,10,11);
        let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        //octave y = 2*exp(-t/2)+exp(-t/4)+1;
        let y = DVector::from(vec![
            4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
        ]);
        let _initial_guess = vec![1., 2.];

        assert!(
            matches!(
                LevMarBuilder::new()
                    .x(x)
                    .y(y.clone())
                    .initial_guess(&[1.])
                    .model(&model)
                    .build(),
                Err(LevMarBuilderError::InvalidParameterCount { .. })
            ),
            "invalid parameter count must produce correct error"
        );

        assert!(
            matches!(
                LevMarBuilder::new()
                    .x(DVector::from(vec! {1.,2.,3.}))
                    .y(y)
                    .initial_guess(&[1.])
                    .model(&model)
                    .build(),
                Err(LevMarBuilderError::InvalidLengthOfData { .. })
            ),
            "invalid parameter count must produce correct error"
        );

        assert!(
            matches!(
                LevMarBuilder::new()
                    .x(DVector::from(Vec::<f64>::new()))
                    .y(DVector::from(Vec::<f64>::new()))
                    .initial_guess(&[1.])
                    .model(&model)
                    .build(),
                Err(LevMarBuilderError::ZeroLengthVector)
            ),
            "invalid parameter count must produce correct error"
        );
    }
}
