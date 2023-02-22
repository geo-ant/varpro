use std::thread::available_parallelism;

use crate::model::errors::ModelError;
use crate::model::model_basis_function::ModelBasisFunction;
use nalgebra::base::Scalar;
use nalgebra::{DMatrix, DVector, Dyn};
use num_traits::Zero;

mod detail;
/// contains the error structure that belongs to a model
pub mod errors;

/// contains the builder code for model creation
pub mod builder;
mod model_basis_function;
#[cfg(test)]
pub mod test;

/// Represents an abstraction for a separable nonlinear model
///
/// # Introduction
/// A separable nonlinear model is
/// a nonlinear, vector valued function function `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` which depends on
/// * an independent variable `$\vec{x}$`, e.g. a location, time, etc...
/// * the *nonlinear* model parameters `$\vec{\alpha}$`.
/// * and a number of linear parameters `$\vec{c}$`
/// The number of elements in `$\vec{f}$` must be the same as in the independent variable `$\vec{x}$`.
///
/// *Separable* means that the nonlinear model function can be written as the
/// linear combination of `$N_{base}$` nonlinear base functions, i.e.
/// ```math
/// \vec{f}(\vec{x},\vec{\alpha},\vec{c}) = \sum_{j=1}^M c_j \cdot \vec{f}_j(\vec{x},S_j(\vec{\alpha})),
/// ```
/// where `$\vec{c}=(c_1,\dots,\c_M)$` are the coefficients of the model base functions and
/// `$S_j(\vec{\alpha})$` is a subset of the nonlinear model parameters that may be different
/// for each model function. The base functions should depend on the model parameters nonlinearly.
/// Linear parameters should be in the coefficient vector `$\vec{c}$` only.
///
/// ## Important Considerations for Basis Functions
///
/// We have already stated that the base functions should depend on their parameter subset
/// `$S_j(\vec{\alpha})$` in a non-linear manner. Linear dependencies should be rewritten in such a
/// way that we can stick them into the coefficient vector `$\vec{c}$`. It is not strictly necessary
/// to do that, but it will only make the fitting process slower and less robust. The great strength
/// of varpro comes from treating the linear and nonlinear parameters differently.
///
/// Another important thing is to ensure that the base functions are not linearly dependent. That is,
/// at least not for all possible choices of `$\vec{alpha}$`. It is OK if model functions become linearly
/// dependent for *some* combinations of model parameters. See also
/// [LevMarProblemBuilder::epsilon](crate::solver::levmar::builder::LevMarProblemBuilder::epsilon).
///
/// ## Basis functions
///
/// It perfectly fine for a base function to depend on all or none of the model parameters or any
/// subset of the model parameters.
///
/// ### Invariant Functions
///
/// There may be functions that depend only on the location parameter `$\vec{x}$` but
/// not on the nonlinear parameters. It's best to subsume all those functions under one
/// _invariant_ base function.
///
/// ### Base Functions
///
/// The varpro library expresses base function signatures as `$\vec{f}_j(\vec{x},p_1,...,p_{P_j}))$`, where
/// `$p_1,...,p_{P_J}$` are the paramters that the basefunction with index `$j$` actually depends on.
/// These are not given as a vector, but as a (variadic) list of arguments. The parameters must be
/// a subset of the model parameters. In order to add functions like this to a model, we must also
/// provide all partial derivatives (for parameters that the function explicitly depends on).
///
/// Refer to the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
/// to see how to construct a model with base functions.
///
/// # Usage
///
/// There is two ways to describe a separable nonlinear model. Firstly, you can
/// use the builder in this crate or you can implement this trait on your own
/// type.
///
/// ## Using the Builder
///
/// The simplest way to build an instance of a model us to use the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
/// to create a model from a set of base functions and their derivatives. Then
/// pass this to a solver for solving for both the linear coefficients as well
/// as the nonlinear parameters. This is recommended for prototyping and should
/// already run way faster and more stable than just using a least squares backend
/// directly. For maximum performance, you can also write your own type
/// that implements the [SeparableNonlinearModel] interface.
///
/// ## Implementing the Interface By Hand
///
/// A handrolled implementation allows us avoid some indirection that the model
/// builder introduces. For that, write your own type that implements this
/// trait and pay close attention to the documentation of the member functions
/// below.
pub trait SeparableNonlinearModel<ScalarType: Scalar> {
    /// the associated error type that can occur when the
    /// model or the derivative is evaluated.
    /// If this model does not need (or for performance reasons does not want)
    /// to return an error, it is possible to specify [`std::convert::Infallible`]
    /// as the associated `Error` type.
    type Error: std::error::Error;

    /// must return the number of *nonlinear* parameters that this model depends on.
    /// This does not include the number of linear *coefficients*.
    /// The parameters passed to `eval` and `eval_partial_deriv` must
    /// have this amount of elements.
    fn parameter_count(&self) -> usize;

    /// must return the number of base functions that this model depends on.
    /// This is equal to the number of *linear coefficients* of the model.
    /// This is also equal to the number of _columns_ of the matrices returned
    /// from the `eval` and `eval_partial_deriv` methods.
    fn base_function_count(&self) -> usize;

    fn set_params(&mut self, parameters : &[ScalarType]) -> Result<(),Self::Error>;

    /// Evaluate the base functions of the model at the given location `$\vec{x}$`
    /// and parameters `$\vec{\alpha}$` and return them in matrix form.
    /// The columns of this matrix are the evaluated base functions.
    /// See below for a detailed explanation.
    ///
    /// # Arguments
    ///
    /// * `location`: the value of the independent location parameter `$\vec{x}$`
    /// * `parameters`: the parameter vector `$\vec{\alpha}$`
    ///
    /// # Result
    ///
    /// As explained above, this method returns a matrix, whose columns are the
    /// base functions evaluated at the given parameters and location. More formally,
    /// if the model is written as a superposition of `$M$` base functions like so:
    ///
    /// ```math
    /// \vec{f}(\vec{x},\vec{\alpha}) = \sum_{j=1}^M c_j \cdot \vec{f}_j(\vec{x},S_j(\vec{\alpha})),
    /// ```
    ///  
    /// then the matrix must look like this:
    ///
    /// ```math
    ///   \mathbf{\Phi}(\vec{x},\vec{\alpha}) \coloneqq
    ///   \begin{pmatrix}
    ///   \vert & \dots & \vert \\
    ///   f_1(\vec{x},\vec{\alpha}) & \dots & f_M(\vec{x},\vec{\alpha}) \\
    ///   \vert & \dots & \vert \\
    ///   \end{pmatrix},
    /// ```
    ///
    /// The ordering of the function must not change between function calls
    /// and it must also be the same as for the evaluation of the derivatives.
    /// The j-th base function must be at the j-th column. The one thing to remember
    /// is that in computerland the matrix indices start at 0, so the first function
    /// is at column 0, and so on...
    ///
    /// ## Errors
    /// Errors can e.g. be returned if...
    /// * too few (or too many) parameters have been provided to evaluate the model
    /// * the basis functions do not produce a vector of the same length as the `location` argument `$\vec{x}$`,
    /// which indicates a programming error
    /// * ...
    fn eval(
        &self,
    ) -> Result<DMatrix<ScalarType>, Self::Error>;

    /// Evaluate the partial derivatives for the base function at for the
    /// given location and parameters and return them in matrix form.
    ///
    /// # Arguments
    ///
    /// * `location`: the value of the independent location parameter `$\vec{x}$`
    /// * `parameters`: the parameter vector `$\vec{\alpha}$`
    /// * `derivative_index`: The index of the nonlinear parameter with respect to which
    /// partial derivative should be evaluated. We use _zero based indexing_
    /// here! Put in more simple terms, say your model has three nonlinear parameters
    /// `a,b,c`, so your vector of nonlinear parameters is `$\vec{\alpha} = (a,b,c)$`.
    /// Then index 0 requests `$\partial/\partial_a$`, index 1 requests `$\partial/\partial_b$`
    /// and index 2 requests `$\partial/\partial_c$`.
    ///
    /// # Result
    ///
    /// Like the `eval` method, this method returns a matrix, whose columns are the
    /// correspond to the base functions evaluated at the given parameters and location.
    /// However, for this
    /// More formally,
    /// if the model is written as a superposition of `$M$` base functions like so:
    ///
    /// ```math
    /// \vec{f}(\vec{x},\vec{\alpha}) = \sum_{j=1}^M c_j \cdot \vec{f}_j(\vec{x},S_j(\vec{\alpha})),
    /// ```
    ///
    /// Further assume that our vector of nonlinear parameters looks like
    /// `$\vec{\alpha} = (\alpha_1,...,\alpha_N)$` and that the partial derivative
    /// with respect to `$\alpha_\ell$` (so the given `derivative_index` was
    /// `$\ell-1$`, since it is zero-based).
    ///
    /// then the matrix must look like this:
    ///
    /// ```math
    ///   \mathbf{\Phi}(\vec{x},\vec{\alpha}) \coloneqq
    ///   \begin{pmatrix}
    ///   \vert & \dots & \vert \\
    ///   \partial/\partial_{\alpha\ell} f_1(\vec{x},\vec{\alpha}) & \dots & \partial/\partial_{\alpha\ell} f_M(\vec{x},\vec{\alpha}) \\
    ///   \vert & \dots & \vert \\
    ///   \end{pmatrix},
    /// ```
    ///
    /// The order of the derivatives must be the same as in the model evaluation
    /// and must not change between method calls.
    ///
    /// ## Errors
    ///
    /// An error result is returned when
    /// * the parameters do not have the same length as the model parameters given when building the model
    /// * the basis functions do not produce a vector of the same length as the `location` argument `$\vec{x}$`
    /// * the given parameter index is out of bounds
    /// * ...
    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<DMatrix<ScalarType>, Self::Error>;
}

/// The type returned from building a model using the
/// [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
pub struct SeparableModel<ScalarType>
where
    ScalarType: Scalar,
{
    /// the parameter names of the model. This defines the order in which the
    /// parameters are expected when the methods for evaluating the function
    /// values and the jacobian are called.
    /// The list of parameter contains a nonzero number of names and these names
    /// are unique.
    parameter_names: Vec<String>,
    /// the set of base functions for the model. This already contains the base functions
    /// which are wrapped inside a lambda function so that they can take the whole
    /// parameter space of the model as an argument
    basefunctions: Vec<ModelBasisFunction<ScalarType>>,
    /// the independent variable `$\vec{x}$`
    x_vector : DVector<ScalarType>,
    /// the current parameters with which this model (and its derivatives)
    /// are evaluated
    current_parameters : Vec<ScalarType>,

}

impl<ScalarType> std::fmt::Debug for SeparableModel<ScalarType>
where ScalarType : Scalar{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeparableModel").field("parameter_names", &self.parameter_names).field("basefunctions", &"/* omitted */").field("x_vector", &self.x_vector).finish()
    }
}

impl<ScalarType> SeparableModel<ScalarType>
where
    ScalarType: Scalar,
{
    /// Get the parameters of the model
    pub fn parameters(&self) -> &[String] {
        &self.parameter_names
    }
}

impl<ScalarType> SeparableNonlinearModel<ScalarType> for SeparableModel<ScalarType>
where
    ScalarType: Scalar + Zero,
{
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }

    fn base_function_count(&self) -> usize {
        self.basefunctions.len()
    }

    fn eval(
        &self,
    ) -> Result<DMatrix<ScalarType>, ModelError> {
        let location = &self.x_vector;
        let parameters = &self.current_parameters;
        if parameters.len() != self.parameter_count() {
            return Err(ModelError::IncorrectParameterCount {
                required: self.parameter_count(),
                actual: parameters.len(),
            });
        }

        let nrows = location.len();
        let ncols = self.base_function_count();
        // this pattern is not great, but the trait bounds in copy_from still
        // prevent us from doing something better
        let mut function_value_matrix =
            unsafe { DMatrix::uninit(Dyn(nrows), Dyn(ncols)).assume_init() };

        for (basefunc, mut column) in self
            .basefunctions
            .iter()
            .zip(function_value_matrix.column_iter_mut())
        {
            let function_value =
                model_basis_function::evaluate_and_check(&basefunc.function, location, parameters)?;
            column.copy_from(&function_value);
        }
        Ok(function_value_matrix)
    }

    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<DMatrix<ScalarType>, Self::Error> {
        let location = &self.x_vector;
        let parameters = &self.current_parameters;
        if parameters.len() != self.parameter_names.len() {
            return Err(ModelError::IncorrectParameterCount {
                required: self.parameter_names.len(),
                actual: parameters.len(),
            });
        }

        if derivative_index >= self.parameter_names.len() {
            return Err(ModelError::DerivativeIndexOutOfBounds {
                index: derivative_index,
            });
        }

        let nrows = location.len();
        let ncols = self.basefunctions.len();
        let mut derivative_function_value_matrix =
            DMatrix::<ScalarType>::from_element(nrows, ncols, Zero::zero());

        for (basefunc, mut column) in self
            .basefunctions
            .iter()
            .zip(derivative_function_value_matrix.column_iter_mut())
        {
            if let Some(derivative) = basefunc.derivatives.get(&derivative_index) {
                let deriv_value =
                    model_basis_function::evaluate_and_check(derivative, location, parameters)?;
                column.copy_from(&deriv_value);
            }
        }
        Ok(derivative_function_value_matrix)
    }

    fn set_params(&mut self, parameters : &[ScalarType]) -> Result<(),Self::Error> {
        self.current_parameters = parameters.to_vec();
        Ok(())
    }
}
