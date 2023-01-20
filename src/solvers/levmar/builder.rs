use crate::model::SeparableModel;
use crate::solvers::levmar::weights::Weights;
use crate::solvers::levmar::LevMarProblem;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, DVector, Scalar};
use num_traits::{Float, Zero};
use thiserror::Error as ThisError;
use std::ops::Mul;

/// Errors pertaining to use errors of the [LeastSquaresProblemBuilder]
#[derive(Debug, Clone, ThisError, PartialEq, Eq)]
pub enum LevMarBuilderError {
    /// the data for the independent variable was not given to the builder
    #[error("Data for vector x not provided.")]
    XDataMissing,

    /// the data for y variable was not given to the builder
    #[error("Data for vector y not provided.")]
    YDataMissing,

    /// no model was provided to the builder
    #[error("Model not provided.")]
    ModelMissing,

    /// no model was provided to the builder
    #[error("Initial guess for parameters not provided.")]
    InitialGuessMissing,

    /// x and y vector have different lengths
    #[error(
        "Vectors x and y must have same lengths. Given x length = {} and y length = {}",
        x_length,
        y_length
    )]
    InvalidLengthOfData { x_length: usize, y_length: usize },

    /// the provided x and y vectors must not have zero elements
    #[error("x or y must have nonzero number of elements.")]
    ZeroLengthVector,

    /// the model has a different number of parameters than the provided initial guesses
    #[error("Initial guess vector must have same length as parameters. Model has {} parameters and {} initial guesses were provided.",model_count, provided_count)]
    InvalidParameterCount {
        model_count: usize,
        provided_count: usize,
    },

    /// y vector and weights have different lengths
    #[error("The weights must have the same length as the data y.")]
    InvalidLengthOfWeights,
}

#[derive(Clone)]
/// A builder structure to create a [LevMarProblem](super::LevMarProblem), which can be used for
/// fitting a separable model to data.
/// # Example
/// The following code shows how to create an unweighted least squares problem to fit the separable model
/// `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` given by `model` to data `$\vec{y}$` (given as `y`) using the
/// independent variable `$\vec{x}$` (given as `x`). Furthermore we need an initial guess `params`
/// for the nonlinear model parameters `$\vec{\alpha}$`
/// ```rust
/// # use nalgebra::{DVector, Scalar};
/// # use varpro::model::SeparableModel;
/// # use varpro::solvers::levmar::{LevMarProblem, LevMarProblemBuilder};
/// # fn builder_example(x : DVector<f64>, y :DVector<f64>,model : & SeparableModel<f64>, params: &[f64]) {
///   let problem = LevMarProblemBuilder::new()
///                 .x(x)
///                 .y(y)
///                 .model(model)
///                 .initial_guess(params)
///                 .build()
///                 .unwrap();
/// # }
/// ```
/// # Building a Model
/// A new builder is constructed with the [new](LevMarProblemBuilder::new) method. It must be filled with
/// content using the methods described in the following. After all mandatory fields have been filled,
/// the [build](LevMarProblemBuilder::build) method can be called. This returns a [Result](std::result::Result)
/// type that contains the finished model iff all mandatory fields have been set with valid values. Otherwise
/// it contains an error variant.
/// ## Mandatory Building Blocks
/// The following methods must be called before building a problem with the builder.
/// * [x](LevMarProblemBuilder::x) to set the values of the independent variable `$\vec{x}$`
/// * [y](LevMarProblemBuilder::y) to set the values of the independent variable `$\vec{y}$`
/// * [model](LevMarProblemBuilder::model) to set the model function
/// * [initial_guess](LevMarProblemBuilder::initial_guess) provide an initial guess
/// ## Additional Building blocks
/// The other methods of the builder allow to manipulate further aspects, like adding weights to the data.
/// The methods are marked as **Optional**.
pub struct LevMarProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Copy,
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
    /// Optional: weights to be applied to the data for weightes least squares
    /// if no weights are given, the problem is unweighted, i.e. the same as if
    /// all weights were 1.
    /// Must have the same length as x and y.
    weights: Weights<ScalarType>,
}

/// Same as `Self::new()`.
impl<'a, ScalarType> Default for LevMarProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Zero + Copy,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, ScalarType> LevMarProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Zero + Copy,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    /// Create a new builder with empty fields and unit weights
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            separable_model: None,
            parameter_initial_guess: None,
            epsilon: None,
            weights: Weights::default(),
        }
    }

    /// **Mandatory**: Set the value of the independent variable `$\vec{x}$` of the problem.
    /// The length of `$\vec{x}$` and the data `$\vec{y}$` must be the same.
    pub fn x<VectorType>(self, xvec: VectorType) -> Self
    where
        DVector<ScalarType>: From<VectorType>,
    {
        Self {
            x: Some(DVector::from(xvec)),
            ..self
        }
    }

    /// **Mandatory**: Set the data which we want to fit: `$\vec{y}=\vec{y}(\vec{x})$`.
    /// The length of `$\vec{x}$` and the data `$\vec{y}$` must be the same.
    pub fn y<VectorType>(self, yvec: VectorType) -> Self
    where
        DVector<ScalarType>: From<VectorType>,
    {
        Self {
            y: Some(DVector::from(yvec)),
            ..self
        }
    }

    /// **Mandatory**: Set the actual model used for fitting
    pub fn model(self, model: &'a SeparableModel<ScalarType>) -> Self {
        Self {
            separable_model: Some(model),
            ..self
        }
    }

    /// **Mandatory**: provide initial guess for the parameters.
    /// The number of values in the initial guess  must match the number of model parameters
    pub fn initial_guess(self, params: &[ScalarType]) -> Self {
        Self {
            parameter_initial_guess: Some(params.to_vec()),
            ..self
        }
    }

    /// **Optional** This value is relevant for the solver, because it uses singular value decomposition
    /// internally. This method sets a value `\epsilon` for which smaller (i.e. absolute - wise) singular
    /// values are considered zero. In essence this gives a truncation of the SVD. This might be
    /// helpful if two basis functions become linear dependent when the nonlinear model parameters
    /// align in an unfortunate way. In this case a higher epsilon might increase the robustness
    /// of the fitting process.
    ///
    /// If this value is not given, it will be set to machine epsilon.
    ///
    /// The given epsilon is automatically converted to a non-negative number.
    pub fn epsilon(self, eps: ScalarType::RealField) -> Self {
        Self {
            epsilon: Some(<ScalarType::RealField as Float>::abs(eps)),
            ..self
        }
    }

    /// **Optional** Add diagonal weights to the problem (meaning data points are statistically
    /// independent). If this is not given, the problem is unweighted, i.e. each data point has
    /// unit weight.
    ///
    /// **Note** The weighted residual is calculated as `$||W(\vec{y}-\vec{f}(\vec{\alpha})||^2$`, so
    /// to make weights that have a statistical meaning, the diagonal elements of the weight matrix should be
    /// set to `w_{jj} = 1/\sigma_j` where `$\sigma_j$` is the (estimated) standard deviation associated with
    /// data point `$y_j$`.
    pub fn weights<VectorType>(self, weights: VectorType) -> Self
    where
        DVector<ScalarType>: From<VectorType>,
    {
        Self {
            weights: Weights::diagonal(weights),
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
    /// If all prerequisites are fulfilled, returns a [LevMarProblem](super::LevMarProblem) with the given
    /// content and the parameters set to the initial guess. Otherwise returns an error variant.
    pub fn build(self) -> Result<LevMarProblem<'a, ScalarType>, LevMarBuilderError> {
        let finalized_builder = self.finalize()?;

        Ok(LevMarProblem::from(finalized_builder))
    }
}

/// helper structure that has the same fields as the builder
/// but all of them are valid
#[doc(hidden)]
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
    weights: Weights<ScalarType>,
}

/// helper to create a [LevMarProblem] from a finalized builder. This initializes all fields
/// with valid values. Most importantly, the data is weighted with the weight matrix and
/// the initial parameters are set.
#[doc(hidden)]
#[allow(non_snake_case)]
impl<'a, ScalarType> From<FinalizedBuilder<'a, ScalarType>> for LevMarProblem<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Copy,
    ScalarType::RealField: Mul<ScalarType, Output = ScalarType> + Float,
{
    fn from(finalized_builder: FinalizedBuilder<'a, ScalarType>) -> Self {
        // 1) create weighted data
        let y_w = &finalized_builder.weights * finalized_builder.y;

        // 2) initialize the levmar problem. Some field values are dummy initialized
        // (like the SVD) because they are calculated in step 3 as part of set_params
        let mut problem = LevMarProblem {
            // these parameters all come from the builder
            x: finalized_builder.x,
            y_w,
            model: finalized_builder.model,
            svd_epsilon: finalized_builder.epsilon,
            // these parameters are dummies and they will be overwritten immediately
            // by the call to set_params
            model_parameters: Vec::new(),
            cached: None,
            weights: finalized_builder.weights,
        };
        // 3) step 3: overwrite the dummy values with the correct results for the given
        // parameters, such that the problem is in a valid state
        problem.set_params(&DVector::from(finalized_builder.parameter_initial_guess));

        problem
    }
}

// private implementations
impl<'a, ScalarType> LevMarProblemBuilder<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField + Copy,
    ScalarType::RealField: Float + Mul<ScalarType, Output = ScalarType>,
{
    /// helper function to check if all required fields have been set and pass the checks
    /// if so, this returns a destructured result of self
    fn finalize(self) -> Result<FinalizedBuilder<'a, ScalarType>, LevMarBuilderError> {
        match self {
            // in this case all required fields are set to something
            LevMarProblemBuilder {
                x: Some(x),
                y: Some(y),
                separable_model: Some(model),
                parameter_initial_guess: Some(parameter_initial_guess),
                epsilon,
                weights,
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
                } else if !weights.is_size_correct_for_data_length(y.len()) {
                    //check that weights have correct length if they were given
                    Err(LevMarBuilderError::InvalidLengthOfWeights)
                } else {
                    Ok(FinalizedBuilder {
                        x,
                        y,
                        model,
                        parameter_initial_guess,
                        epsilon: epsilon.unwrap_or_else(Float::epsilon),
                        weights,
                    })
                }
            }
            // not all required fields are set, report missing parameters
            LevMarProblemBuilder {
                x,
                y,
                separable_model: model,
                parameter_initial_guess,
                epsilon: _,
                weights: _,
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
                    unreachable!("Error in the library.")
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use crate::linalg_helpers::DiagDMatrix;
    use crate::solvers::levmar::builder::LevMarBuilderError;
    use crate::solvers::levmar::weights::Weights;
    use crate::solvers::levmar::LevMarProblemBuilder;
    use crate::test_helpers::get_double_exponential_model_with_constant_offset;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn new_builder_starts_with_empty_fields() {
        let builder = LevMarProblemBuilder::<f64>::new();
        let LevMarProblemBuilder {
            x,
            y,
            separable_model: model,
            parameter_initial_guess,
            epsilon,
            weights,
        } = builder;
        assert!(x.is_none());
        assert!(y.is_none());
        assert!(model.is_none());
        assert!(parameter_initial_guess.is_none());
        assert!(epsilon.is_none());
        assert_eq!(weights, Weights::Unit);
    }

    #[test]
    #[allow(clippy::float_cmp)] //clippy moans, but it's wrong (again!)
    #[allow(non_snake_case)]
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
        let builder = LevMarProblemBuilder::new()
            .x(x.clone())
            .y(y.clone())
            .initial_guess(initial_guess.as_slice())
            .model(&model);
        let problem = builder
            .clone()
            .build()
            .expect("Valid builder should not fail build");

        assert_eq!(problem.x, x);
        assert_eq!(problem.y_w, y);
        assert_eq!(problem.model_parameters, initial_guess);
        //assert!(problem.model.as_ref().as_ptr()== model.as_ptr()); // this don't work, boss
        assert_eq!(problem.svd_epsilon, f64::EPSILON); //clippy moans, but it's wrong (again!)
        assert!(
            problem.cached.as_ref().unwrap().current_svd.u.is_some(),
            "SVD U must have been calculated on initialization"
        );
        assert!(
            problem.cached.as_ref().unwrap().current_svd.v_t.is_some(),
            "SVD V^T must have been calculated on initialization"
        );

        // now check that the given epsilon is also passed correctly to the model
        // and also that the weights are correctly passed and used to weigh the original data
        let weights = 2. * &y;
        let W = DMatrix::from_diagonal(&weights);

        let problem = builder
            .epsilon(-1.337) // check that negative values are converted to absolutes
            .weights(weights.clone())
            .build()
            .expect("Valid builder should not fail");
        assert_eq!(problem.svd_epsilon, 1.337);
        assert_eq!(
            problem.y_w,
            &W * &y,
            "Data must be correctly weighted with weights"
        );
        if let Weights::Diagonal(diag) = problem.weights {
            assert_eq!(
                diag,
                DiagDMatrix::from(weights),
                "Diagonal weight matrix must be correctly passed on"
            );
        } else {
            panic!("Simple weights call must produce diagonal weight matrix!");
        }
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

        let _builder = LevMarProblemBuilder::new()
            .x(x.clone())
            .y(y.clone())
            .initial_guess(initial_guess.as_slice())
            .model(&model);

        assert!(matches!(
            LevMarProblemBuilder::new()
                .y(y.clone())
                .initial_guess(initial_guess.as_slice())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::XDataMissing)
        ));
        assert!(matches!(
            LevMarProblemBuilder::new()
                .x(x.clone())
                .initial_guess(initial_guess.as_slice())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::YDataMissing)
        ));
        assert!(matches!(
            LevMarProblemBuilder::new()
                .x(x.clone())
                .y(y.clone())
                .model(&model)
                .build(),
            Err(LevMarBuilderError::InitialGuessMissing)
        ));
        assert!(matches!(
            LevMarProblemBuilder::new()
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
                LevMarProblemBuilder::new()
                    .x(x.clone())
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
                LevMarProblemBuilder::new()
                    .x(DVector::from(vec! {1.,2.,3.}))
                    .y(y.clone())
                    .initial_guess(&[1., 2.])
                    .model(&model)
                    .build(),
                Err(LevMarBuilderError::InvalidLengthOfData { .. })
            ),
            "invalid parameter count must produce correct error"
        );

        assert!(
            matches!(
                LevMarProblemBuilder::new()
                    .x(DVector::from(Vec::<f64>::new()))
                    .y(DVector::from(Vec::<f64>::new()))
                    .initial_guess(&[1., 2.])
                    .model(&model)
                    .build(),
                Err(LevMarBuilderError::ZeroLengthVector)
            ),
            "zero parameter count must produce correct error"
        );

        assert!(
            matches!(
                LevMarProblemBuilder::new()
                    .x(x)
                    .y(y)
                    .initial_guess(&[1., 2.])
                    .model(&model)
                    .weights(vec! {1.,2.,3.})
                    .build(),
                Err(LevMarBuilderError::InvalidLengthOfWeights { .. })
            ),
            "invalid length of weights"
        );
    }
}
