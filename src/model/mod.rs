use crate::model::errors::ModelError;
use crate::model::model_basis_function::ModelBasisFunction;
use nalgebra::base::Scalar;
use nalgebra::{DMatrix, DVector, DefaultAllocator, Dyn, OMatrix, OVector};
use num_traits::Zero;

mod detail;
/// contains the error structure that belongs to a model
pub mod errors;

/// contains the builder code for model creation
pub mod builder;
mod model_basis_function;
#[cfg(any(test, doctest))]
#[allow(missing_docs)]
pub mod test;

/// Represents an abstraction for a separable nonlinear model
///
/// # Introduction
///
/// A separable nonlinear model is
/// a nonlinear, vector valued function function `$\vec{f}(\vec{\alpha},\vec{c}) \in \mathbb{C}^n$` which depends on
/// * the *nonlinear* model parameters `$\vec{\alpha}$`.
/// * and a number of linear parameters `$\vec{c}$`
///
/// The model is a real valued or complex valued function of the model parameters. The
/// model is vector valued, i.e. it returns a vector of `$n$` values.
///
/// *Separable* means that the nonlinear model function can be written as the
/// linear combination of `$M$` nonlinear base functions , i.e.
///
/// ```math
/// \vec{f}(\vec{x},\vec{\alpha},\vec{c}) = \sum_{j=1}^M c_j \cdot \vec{f}_j(S_j(\vec{\alpha})),
/// ```
///
/// where `$\vec{c}=(c_1,\dots,c_M)$` are the coefficients of the model base functions `$f_j$` and
/// `$S_j(\vec{\alpha})$` is a subset of the nonlinear model parameters that may be different
/// for each model function. The base functions should depend on the model parameters nonlinearly.
/// Linear parameters should be in the coefficient vector `$\vec{c}$` only.
///
/// ## Important Considerations for Base Functions
///
/// We have already stated that the base functions should depend on their parameter subset
/// `$S_j(\vec{\alpha})$` in a *non-linear* manner. Linear dependencies should be rewritten in such a
/// way that we can stick them into the coefficient vector `$\vec{c}$`. It is not strictly necessary
/// to do that, but it will only make the fitting process slower and less robust. The great strength
/// of varpro comes from treating the linear and nonlinear parameters differently.
///
/// Another important thing is to ensure that the base functions are not linearly dependent,
/// at least not for all possible choices of `$\vec{alpha}$`. It is sometimes unavoidable that
/// that model functions become linearly
/// dependent for *some* combinations of model parameters. See also
/// [SeparableProblemBuilder::epsilon](crate::problem::SeparableProblemBuilder::epsilon).
///
/// It perfectly fine for a base function to depend on all or none of the model parameters or any
/// subset of the model parameters. There is also no restrictions on which base functions
/// can depend on which nonlinear parameters.
///
/// # Usage
///
/// There is two ways to get a type that implements the separable nonlinear model trait.
/// Firstly, you can obviously create your own type and make it implement this trait.
/// Secondly you can use the [`crate::model::builder::SeparableModelBuilder`] in this crate.
///
/// ## Using the Builder
///
/// The simplest way to build an instance of a model us to use the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
/// to create a model from a set of base functions and their derivatives. Then
/// pass this to a solver for solving for both the linear coefficients as well
/// as the nonlinear parameters. This is definitely recommended for prototyping and it
/// might already be enough for a production implementation as it should
/// already run way faster and more stable than just using a least squares backend
/// directly.
///
/// ## Manual Implementation
///
/// For maximum performance, you can also write your own type that implements this trait.
/// An immediate advantage of this is that you can cache the results of the base functions
/// and reuse them in the derivatives as is often possible. This can be a huge performance boost if the
/// base functions are expensive to compute. Also, you can avoid the indirection that the model builder
/// introduces, although this should matter to a lesser degree.
///
/// ### Manual Implementation Example
///
/// This example shows how to implement a double exponential
/// decay with constant offset from hand and takes advantage
/// of caching the intermediate calculations.
///
/// ```
/// use varpro::prelude::*;
/// use nalgebra::{DVector,OVector,Vector2,DMatrix,OMatrix,DefaultAllocator,U2,U3,Dyn};
/// /// A separable model for double exponential decay
/// /// with a constant offset
/// /// f_j = c1*exp(-x_j/tau1) + c2*exp(-x_j/tau2) + c3
/// /// this is a handcrafted model which uses caching for
/// /// maximum performance.
/// ///
/// /// This is an example of how to implement a separable model
/// /// using the trait directly without using the builder.
/// /// This allows us to use caching of the intermediate results
/// /// to calculate the derivatives more efficiently.
/// pub struct DoubleExpModelWithConstantOffsetSepModel {
///     /// the x vector associated with this model.
///     /// We make this configurable to enable models to
///     /// work with different x coordinates, but this is
///     /// not an inherent requirement. We don't have to have a
///     /// field like this. The only requirement given by the trait
///     /// is that the model can be queried about the length
///     /// of its output
///     x_vector : DVector<f64>,
///     /// current parameters [tau1,tau2] of the exponential
///     /// functions
///     params : OVector<f64,Dyn>,
///     /// cached evaluation of the model
///     /// the matrix columns contain the complete evaluation
///     /// of the model. That is the first column contains the
///     /// exponential exp(-x/tau), the second column contains
///     /// exp(-x/tau) both evaluated on the x vector. The third
///     /// column contains a column of straight ones for the constant
///     /// offset.
///     ///
///     /// This value is calculated in the set_params method, which is
///     /// the only method with mutable access to the model state.
///     eval : OMatrix<f64,Dyn,Dyn>,
/// }
///
/// impl DoubleExpModelWithConstantOffsetSepModel {
///     /// create a new model with the given x vector and initial guesses
///     /// for the exponential decay constants tau_1 and tau_2
///     pub fn new(x_vector : DVector<f64>,(tau1_guess,tau2_guess):(f64,f64)) -> Self {
///         let x_len = x_vector.len();
///         let mut ret = Self {
///             x_vector,
///             params : OVector::<f64,Dyn>::from_column_slice(&[tau1_guess,tau2_guess]),//<-- will be overwritten by set_params
///             eval : OMatrix::zeros_generic(Dyn(x_len), Dyn(3))
///         };
///         ret.set_params(OVector::<f64,Dyn>::from_column_slice(&[tau1_guess,tau2_guess])).unwrap();
///         ret
///     }
/// }
///
/// impl SeparableNonlinearModel for DoubleExpModelWithConstantOffsetSepModel {
///     /// we give a custom mddel error here, but we
///     /// could also have indicated that our calculations cannot
///     /// fail by using [`std::convert::Infallible`].
///     type Error = varpro::model::errors::ModelError;
///     /// the actual scalar type that our model uses for calculations
///     type ScalarType = f64;
///
///     #[inline]
///     fn parameter_count(&self) -> usize {
///         // regardless of the fact that we know at compile time
///         // that the length is 2, we still have to return it
///         2
///     }
///
///     #[inline]
///     fn base_function_count(&self) -> usize {
///         // same as above
///         3
///     }
///     
///     // we use this method not only to set the parameters inside the
///     // model but we also cache some calculations. The advantage is that
///     // we don't have to recalculate the exponential terms for either
///     // the evaluations or the derivatives for the same parameters.
///     fn set_params(&mut self, parameters : OVector<f64,Dyn>) -> Result<(),Self::Error> {
///         // even if it is not the only thing we do, we still
///         // have to update the internal parameters of the model
///         self.params = parameters;
///
///         // parameters expected in this order
///         // use unsafe to avoid bounds checks
///         let tau1 = unsafe { self.params.get_unchecked(0) };
///         let tau2 = unsafe { self.params.get_unchecked(1) };
///
///         // the exponential exp(-x/tau1)
///         let f1 = self.x_vector.map(|x| f64::exp(-x / tau1));
///         // the exponential exp(-x/tau2)
///         let f2 = self.x_vector.map(|x| f64::exp(-x / tau2));
///
///         self.eval.set_column(0, &f1);
///         self.eval.set_column(1, &f2);
///         self.eval.set_column(2, &DVector::from_element(self.x_vector.len(), 1.));
///         Ok(())
///     }
///
///     fn params(&self) -> OVector<f64, Dyn> {
///         self.params.clone()
///     }
///     
///     // since we cached the model evaluation, we can just return
///     // it here
///     fn eval(
///         &self,
///     ) -> Result<OMatrix<f64,Dyn,Dyn>, Self::Error> {
///         Ok(self.eval.clone())
///     }
///     
///     // here we take advantage of the cached calculations
///     // so that we do not have to recalculate the exponential
///     // to calculate the derivative.
///     fn eval_partial_deriv(
///         &self,
///         derivative_index: usize,
///     ) -> Result<nalgebra::OMatrix<f64,Dyn,Dyn>, Self::Error> {
///         let location = &self.x_vector;
///         let parameters = &self.params;
///         // derivative index can be either 0,1 (corresponding to the linear parameters
///         // tau1, tau2). Since only one of the basis functions depends on
///         // tau_i, we can simplify calculations here
///
///         let tau = parameters[derivative_index];
///         // the only nonzero derivative is the derivative of the exp(-x/tau) for
///         // the corresponding tau at derivative_index
///         // we can use the precalculated results so we don't have to use the
///         // exponential function again
///         let df = location.map(|x| x / (tau * tau)).component_mul(&self.eval.column(derivative_index));
///
///         // two of the columns are always zero when we differentiate
///         // with respect to tau_1 or tau_2. Remember the constant term
///         // also occupies one column and will always be zero when differentiated
///         // with respect to the nonlinear params of the model
///         let mut derivatives = OMatrix::zeros_generic(Dyn(location.len()), Dyn(3));
///
///         derivatives.set_column(derivative_index, &df);
///         Ok(derivatives)
///     }
///
///     fn output_len(&self) -> usize {
///         // this is how we give a length that is only known at runtime.
///         // We wrap it in a `Dyn` instance.
///         self.x_vector.len()
///     }
/// }
/// ```
#[allow(clippy::type_complexity)]
pub trait SeparableNonlinearModel
where
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn, Dyn>,
{
    /// the scalar number type for this model, which should be
    /// a real or complex number type, commonly either `f64` or `f32`.
    type ScalarType: Scalar;

    /// the associated error type that can occur when the
    /// model or the derivative is evaluated.
    /// If this model does not need (or for performance reasons does not want)
    /// to return an error, it is possible to specify [`std::convert::Infallible`]
    /// as the associated `Error` type.
    type Error: std::error::Error + Send;

    /// return the number of *nonlinear* parameters that this model depends on.
    fn parameter_count(&self) -> usize;

    /// return the number of base functions that this model depends on.
    fn base_function_count(&self) -> usize;

    /// return the dimension `$n$` of the output of the model `$\vec{f}(\vec{x},\vec{\alpha},\vec{c}) \in \mathbb{R}^n$`.
    /// This is also the dimension of every single base function.
    fn output_len(&self) -> usize;

    /// Set the nonlinear parameters `$\vec{\alpha}$` of the model to the given vector .
    fn set_params(&mut self, parameters: OVector<Self::ScalarType, Dyn>)
        -> Result<(), Self::Error>;

    /// Get the currently set nonlinear parameters of the model, i.e.
    /// the vector `$\vec{\alpha}$`.
    fn params(&self) -> OVector<Self::ScalarType, Dyn>;

    /// Evaluate the base functions of the model at the currently
    /// set parameters `$\vec{\alpha}$` and return them in matrix form.
    /// The columns of this matrix are the evaluated base functions.
    /// See below for a detailed explanation.
    ///
    /// # Result
    ///
    /// As explained above, this method returns a matrix, whose columns are the
    /// base functions evaluated at the given parameters. More formally,
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
    /// is that in Rust the matrix indices start at 0, so the first function
    /// is at column 0, and so on...
    ///
    /// ## Errors
    /// An error can be returned if the evaluation fails for  some reason.
    ///
    fn eval(&self) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error>;

    /// Evaluate the partial derivatives for the base function at for the
    /// currently set parameters and return them in matrix form.
    ///
    /// # Arguments
    ///
    /// * `derivative_index`: The index of the nonlinear parameter with respect to which
    ///   partial derivative should be evaluated. We use _zero based indexing_
    ///   here! Put in more simple terms, say your model has three nonlinear parameters
    ///   `a,b,c`, so your vector of nonlinear parameters is `$\vec{\alpha} = (a,b,c)$`.
    ///   Then index 0 requests `$\partial/\partial_a$`, index 1 requests `$\partial/\partial_b$`
    ///   and index 2 requests `$\partial/\partial_c$`.
    ///
    /// # Result
    ///
    /// Like the `eval` method, this method must return a matrix, whose columns are the
    /// correspond to the base functions evaluated at the given parameters and location.
    /// However, for this.
    ///
    /// More formally, if the model is written as a superposition of `$M$` base functions like so:
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
    ) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error>;
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
    x_vector: DVector<ScalarType>,
    /// the current parameters with which this model (and its derivatives)
    /// are evaluated
    current_parameters: DVector<ScalarType>,
}

impl<ScalarType> std::fmt::Debug for SeparableModel<ScalarType>
where
    ScalarType: Scalar,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeparableModel")
            .field("parameter_names", &self.parameter_names)
            .field("basefunctions", &"/* omitted */")
            .field("x_vector", &self.x_vector)
            .finish()
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

impl<ScalarType> SeparableNonlinearModel for SeparableModel<ScalarType>
where
    ScalarType: Scalar + Zero,
{
    type ScalarType = ScalarType;
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }

    fn base_function_count(&self) -> usize {
        self.basefunctions.len()
    }

    fn set_params(&mut self, parameters: DVector<ScalarType>) -> Result<(), Self::Error> {
        if parameters.len() != self.parameter_count() {
            return Err(ModelError::IncorrectParameterCount {
                expected: self.parameter_count(),
                actual: parameters.len(),
            });
        }
        self.current_parameters = parameters;
        Ok(())
    }

    fn params(&self) -> DVector<ScalarType> {
        self.current_parameters.clone()
    }

    fn eval(&self) -> Result<DMatrix<ScalarType>, ModelError> {
        let location = &self.x_vector;
        let parameters = &self.current_parameters;
        if parameters.len() != self.parameter_count() {
            return Err(ModelError::IncorrectParameterCount {
                expected: self.parameter_count(),
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
            let function_value = model_basis_function::evaluate_and_check(
                &basefunc.function,
                location,
                parameters.as_slice(),
            )?;
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
                let deriv_value = model_basis_function::evaluate_and_check(
                    derivative,
                    location,
                    parameters.as_slice(),
                )?;
                column.copy_from(&deriv_value);
            }
        }
        Ok(derivative_function_value_matrix)
    }

    fn output_len(&self) -> usize {
        self.x_vector.len()
    }
}
