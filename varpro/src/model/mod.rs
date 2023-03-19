use crate::model::errors::ModelError;
use crate::model::model_basis_function::ModelBasisFunction;
use nalgebra::base::Scalar;
use nalgebra::{DMatrix, DVector, Dyn, Dim, OVector, DefaultAllocator, OMatrix};
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
///
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
/// There is two ways to get a type that implements the separable nonlinear model trait.
/// Firstly, you can obviously creat your own type and make it implement the trait.
/// Secondly you can use the `SeparableModelBuilder` in this crate. 
///
/// ## Using the Builder
///
/// The simplest way to build an instance of a model us to use the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
/// to create a model from a set of base functions and their derivatives. Then
/// pass this to a solver for solving for both the linear coefficients as well
/// as the nonlinear parameters. This is definitely recommended for prototyping and it 
/// might already be enough for a production implementation. and should
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
pub trait SeparableNonlinearModel<ScalarType: Scalar>
    where DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, Self::ParameterDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, Self::OutputDim, Self::ModelDim>

{
    /// the associated error type that can occur when the
    /// model or the derivative is evaluated.
    /// If this model does not need (or for performance reasons does not want)
    /// to return an error, it is possible to specify [`std::convert::Infallible`]
    /// as the associated `Error` type.
    type Error: std::error::Error;
    
    /// the number of *nonlinear* parameters that this model depends on,
    /// expressed as a dimension type of the nalgebra crate. If the number of
    /// parameters is not known at compile time, then use the [nalgebra::Dynamic](nalgebra::Dynamic)
    /// type.
    type ParameterDim: Dim;
    
    /// the number of base functions that this model depends on,
    /// expressed as a dimension type of the nalgebra crate. If the number of
    /// model base functions is not known at compile time, then use the [nalgebra::Dynamic](nalgebra::Dynamic)
    /// type.
    type ModelDim: Dim;

    type OutputDim: Dim;

    /// return the number of *nonlinear* parameters that this model depends on.
    fn parameter_count(&self) -> Self::ParameterDim;

    /// return the number of base functions that this model depends on.
    fn base_function_count(&self) -> Self::ModelDim;
    
    fn output_len(&self) -> Self::OutputDim;

    fn set_params(&mut self, parameters : OVector<ScalarType,Self::ParameterDim>) -> Result<(),Self::Error>;

    fn params(&self) -> OVector<ScalarType,Self::ParameterDim>;

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
    ) -> Result<OMatrix<ScalarType,Self::OutputDim,Self::ModelDim>, Self::Error>;

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
    ) -> Result<OMatrix<ScalarType,Self::OutputDim,Self::ModelDim>, Self::Error>;
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
    current_parameters : DVector<ScalarType>,
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
    type ParameterDim = Dyn;
    type ModelDim = Dyn;
    type OutputDim = Dyn;

    fn parameter_count(&self) -> Dyn {
        Dyn(self.parameter_names.len())
    }

    fn base_function_count(&self) -> Self::ModelDim {
        Dyn(self.basefunctions.len())
    }

    fn set_params(&mut self, parameters : DVector<ScalarType>) -> Result<(),Self::Error> {
        if parameters.len() != self.parameter_count().value() {
            return Err(ModelError::IncorrectParameterCount {
                expected: self.parameter_count().value(),
                actual: parameters.len(),
            });
        }
        self.current_parameters = parameters;
        Ok(())
    }

    fn params(&self) -> DVector<ScalarType> {
        DVector::from(self.current_parameters.clone())
    }

    fn eval(
        &self,
    ) -> Result<DMatrix<ScalarType>, ModelError> {
        let location = &self.x_vector;
        let parameters = &self.current_parameters;
        if parameters.len() != self.parameter_count().value() {
            return Err(ModelError::IncorrectParameterCount {
                expected: self.parameter_count().value(),
                actual: parameters.len(),
            });
        }

        let nrows = location.len();
        let ncols = self.base_function_count();
        // this pattern is not great, but the trait bounds in copy_from still
        // prevent us from doing something better
        let mut function_value_matrix =
            unsafe { DMatrix::uninit(Dyn(nrows), ncols).assume_init() };

        for (basefunc, mut column) in self
            .basefunctions
            .iter()
            .zip(function_value_matrix.column_iter_mut())
        {
            let function_value =
                model_basis_function::evaluate_and_check(&basefunc.function, location, parameters.as_slice())?;
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
                expected: self.parameter_names.len(),
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
                    model_basis_function::evaluate_and_check(derivative, location, parameters.as_slice())?;
                column.copy_from(&deriv_value);
            }
        }
        Ok(derivative_function_value_matrix)
    }

    fn output_len(&self) -> Self::OutputDim {
       Self::OutputDim::from_usize(self.x_vector.len())
    }
}
