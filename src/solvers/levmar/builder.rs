use crate::prelude::*;
use crate::solvers::levmar::SeparableProblem;
use crate::util::Weights;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, DMatrix, Dyn, OMatrix, OVector, Owned, Scalar};
use num_traits::{Float, Zero};
use std::ops::Mul;
use thiserror::Error as ThisError;

/// Errors pertaining to use errors of the [LeastSquaresProblemBuilder]
#[derive(Debug, Clone, ThisError, PartialEq, Eq)]
pub enum LevMarBuilderError {
    /// the data for y variable was not given to the builder
    #[error("Right hand side(s) not provided")]
    YDataMissing,

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
    #[error(
        "Initial guess vector must have same length as parameters. Model has {} parameters and {} initial guesses were provided.",
        model_count,
        provided_count
    )]
    InvalidParameterCount {
        model_count: usize,
        provided_count: usize,
    },

    /// y vector and weights have different lengths
    #[error("The weights must have the same length as the data y.")]
    InvalidLengthOfWeights,
}

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
/// # fn model(model : SeparableModel<f64>,y: DVector<f64>) {
///   let problem = LevMarProblemBuilder::new(model)
///                 .observations(y)
///                 .build()
///                 .unwrap();
/// # }
/// ```
///
/// # Building a Model
///
/// A new builder is constructed with the [new](LevMarProblemBuilder::new) constructor. It must be filled with
/// content using the methods described in the following. After all mandatory fields have been filled,
/// the [build](LevMarProblemBuilder::build) method can be called. This returns a [Result](std::result::Result)
/// type that contains the finished model iff all mandatory fields have been set with valid values. Otherwise
/// it contains an error variant.
///
/// ## Multiple Right Hand Sides
///
/// We can also construct a problem with multiple right hand sides, using the
/// [mrhs](LevMarProblemBuilder::mrhs) constructor, see [LevMarProblem] for
/// additional details.
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SeparableProblemBuilder<Model, const MRHS: bool>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    Model: SeparableNonlinearModel,
{
    /// Required: the data `$\vec{y}(\vec{x})$` that we want to fit
    Y: Option<DMatrix<Model::ScalarType>>,
    /// Required: the model to be fitted to the data
    separable_model: Model,
    /// Optional: set epsilon below which two singular values
    /// are considered zero.
    /// if this is not given, when building the builder,
    /// the this is set to machine epsilon.
    epsilon: Option<<Model::ScalarType as ComplexField>::RealField>,
    /// Optional: weights to be applied to the data for weightes least squares
    /// if no weights are given, the problem is unweighted, i.e. the same as if
    /// all weights were 1.
    /// Must have the same length as x and y.
    weights: Weights<Model::ScalarType, Dyn>,
}

impl<Model> SeparableProblemBuilder<Model, false>
where
    Model::ScalarType: Scalar + ComplexField + Zero + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
{
    /// Create a new builder based on the given model for a problem
    /// with a **single right hand side**. This is the standard use case,
    /// where the data is a vector that is fitted by the model.
    pub fn new(model: Model) -> Self {
        Self {
            Y: None,
            separable_model: model,
            epsilon: None,
            weights: Weights::default(),
        }
    }
}

impl<Model> SeparableProblemBuilder<Model, false>
where
    Model::ScalarType: Scalar + ComplexField + Zero + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
{
    /// **Mandatory**: Set the data which we want to fit: since this
    /// is called on a model builder for problems with **single right hand sides**,
    /// this is a column vector `$\vec{y}=\vec{y}(\vec{x})$` containing the
    /// values we want to fit with the model.
    ///
    /// The length of `$\vec{y}` and the output dimension of the model must be
    /// the same.
    pub fn observations(self, observed: OVector<Model::ScalarType, Dyn>) -> Self {
        let nrows = observed.nrows();
        Self {
            Y: Some(observed.reshape_generic(Dyn(nrows), Dyn(1))),
            ..self
        }
    }
}

impl<Model> SeparableProblemBuilder<Model, true>
where
    Model::ScalarType: Scalar + ComplexField + Zero + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
{
    /// Create a new builder based on the given model
    /// for a problem with **multiple right hand sides** and perform a global
    /// fit. This uses single threaded calculations.
    ///
    /// That means the observations are expected to be a matrix, where the
    /// columns correspond to the individual observations.
    ///
    /// For a set of observations `$\vec{y}_1,\dots,\vec{y}_S$` (column vectors) we
    /// now have to pass a _matrix_ `$Y$` of observations, rather than a single
    /// vector to the builder. As explained above, the resulting matrix would look
    /// like this.
    ///
    /// ```math
    /// \boldsymbol{Y}=\left(\begin{matrix}
    ///  \vert &  & \vert \\
    ///  \vec{y}_1 &  \dots & \vec{y}_S \\
    ///  \vert &  & \vert \\
    /// \end{matrix}\right)
    /// ```
    ///
    /// The nonlinear parameters will be optimized across all the observations
    /// globally, but the best linear coefficients are calculated for each observation
    /// individually. Hence, the latter also become a matrix `$C$`, where the columns
    /// correspond to the linear coefficients of the observation in the same column.
    ///
    /// ```math
    /// \boldsymbol{C}=\left(\begin{matrix}
    ///  \vert &  & \vert \\
    ///  \vec{c}_1 &  \dots & \vec{c}_S \\
    ///  \vert &  & \vert \\
    /// \end{matrix}\right)
    /// ```
    ///
    /// The (column) vector of linear coefficients `$\vec{c}_j$` is for the observation
    /// `$\vec{y}_j$` in the same column.
    pub fn mrhs(model: Model) -> Self {
        Self {
            Y: None,
            separable_model: model,
            epsilon: None,
            weights: Weights::default(),
        }
    }
}

impl<Model> SeparableProblemBuilder<Model, true>
where
    Model::ScalarType: Scalar + ComplexField + Zero + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
{
    /// **Mandatory**: Set the data which we want to fit: This is either a single vector
    /// `$\vec{y}=\vec{y}(\vec{x})$` or a matrix `$\boldsymbol{Y}$` of multiple
    /// vectors. In the former case this corresponds to fitting a single right hand side,
    /// in the latter case, this corresponds to global fitting of a problem with
    /// multiple right hand sides.
    /// The length of `$\vec{x}$` and the number of _rows_ in the data must
    /// be the same.
    pub fn observations(self, observed: OMatrix<Model::ScalarType, Dyn, Dyn>) -> Self {
        Self {
            Y: Some(observed),
            ..self
        }
    }
}

impl<Model, const MRHS: bool> SeparableProblemBuilder<Model, MRHS>
where
    Model::ScalarType: Scalar + ComplexField + Zero + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
{
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
    pub fn epsilon(self, eps: <Model::ScalarType as ComplexField>::RealField) -> Self {
        Self {
            epsilon: Some(<_ as Float>::abs(eps)),
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
    pub fn weights(self, weights: OVector<Model::ScalarType, Dyn>) -> Self {
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
    #[allow(non_snake_case)]
    pub fn build(self) -> Result<SeparableProblem<Model, MRHS>, LevMarBuilderError>
    //@note(geo) both the parallel and non parallel model implement the LeastSquaresProblem trait,
    // but the trait solver cannot figure that out without this extra hint.
    where
        SeparableProblem<Model, MRHS>: LeastSquaresProblem<
                Model::ScalarType,
                Dyn,
                Dyn,
                ParameterStorage = Owned<Model::ScalarType, Dyn>,
            >,
    {
        // and assign the defaults to the values we don't have
        let Y = self.Y.ok_or(LevMarBuilderError::YDataMissing)?;
        let model = self.separable_model;
        let epsilon = self.epsilon.unwrap_or_else(Float::epsilon);
        let weights = self.weights;

        // now do some sanity checks for the values and return
        // an error if they do not pass the test
        let x_len: usize = model.output_len();
        if x_len == 0 || Y.is_empty() {
            return Err(LevMarBuilderError::ZeroLengthVector);
        }

        if x_len != Y.nrows() {
            return Err(LevMarBuilderError::InvalidLengthOfData {
                x_length: x_len,
                y_length: Y.nrows(),
            });
        }

        if !weights.is_size_correct_for_data_length(Y.nrows()) {
            //check that weights have correct length if they were given
            return Err(LevMarBuilderError::InvalidLengthOfWeights);
        }

        //now that we have valid inputs, construct the levmar problem
        // 1) create weighted data
        #[allow(non_snake_case)]
        let Y_w = &weights * Y;

        let params = model.params();
        // 2) initialize the levmar problem. Some field values are dummy initialized
        // (like the SVD) because they are calculated in step 3 as part of set_params
        let mut problem = SeparableProblem {
            // these parameters all come from the builder
            Y_w,
            model,
            svd_epsilon: epsilon,
            cached: None,
            weights,
        };
        problem.set_params(&params);

        Ok(problem)
    }
}
// make available for testing and doc tests
#[cfg(any(test, doctest))]
mod test;
