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


/// TODO TODO TODO DOCUMENT
pub trait SeparableNonlinearModel<ScalarType: Scalar> {
    /// the associated error type that can possibly produce when the
    /// model or the derivative is evaluated.
    /// If this model does not need (or for performance reasons does not want)
    /// to return an error, it is possible to specify [`std::convert::Infallible`]
    /// as the associated `Error` type.
    type Error : std::error::Error;
    /// must return the number of *nonlinear* parameters that this model depends on.
    /// This does not include the number of linear *coefficients*.
    fn parameter_count(&self) -> usize;
    
    /// must return the number of basis functions that this model depends on.
    /// This is equal to the number of *linear coefficients* of the model.
    fn basis_function_count(&self) -> usize;
    
    ///TODO DOCUMENT
    fn eval(&self, location : &DVector<ScalarType>, parameters : &[ScalarType])-> Result<DMatrix<ScalarType>, Self::Error>; 
    
    /// TODO DOCUMENT
    fn eval_partial_deriv(&self, location: &DVector<ScalarType>, parameters : &[ScalarType],derivative_index : usize) -> Result<DMatrix<ScalarType>, Self::Error>;
}

/// This structure represents a separable nonlinear model.
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
/// f(\vec{x},\vec{\alpha}) = \sum_{j=1}^M c_j \cdot \vec{f}_j(\vec{x},S_j(\vec{\alpha})),
/// ```
/// where `$\vec{c}=(c_1,\dots,\c_M)$` are the coefficients of the model basis functions and
/// `$S_j(\vec{\alpha})$` is a subset of the nonlinear model parameters that may be different
/// for each model function. The basis functions should depend on the model parameters nonlinearly.
/// Linear parameters should be in the coefficient vector `$\vec{c}$` only.
///
/// ## Important Considerations for Basis Functions
/// We have already stated that the basis functions should depend on their parameter subset
/// `$S_j(\vec{\alpha})$` in a non-linear manner. Linear dependencies should be rewritten in such a
/// way that we can stick them into the coefficient vector `$\vec{c}$`. It is not strictly necessary
/// to do that, but it will only make the fitting process slower and less robust. The great strength
/// of varpro comes from treating the linear and nonlinear parameters differently.
///
/// Another important thing is to ensure that the basis functions are not linearly dependent. That is,
/// at least not for all possible choices of `$\vec{alpha}$`. It is OK if model functions become linearly
/// dependent for *some* combinations of model parameters. See also
/// [LevMarProblemBuilder::epsilon](crate::solver::levmar::builder::LevMarProblemBuilder::epsilon).
///
/// ## Base Functions
/// It perfectly fine for a base function to depend on all or none of the model parameters or any
/// subset of the model parameters.
///
/// ### Invariant Functions
/// We refer to functions `$\vec{f}_j(\vec{x})$` that do not depend on the model parameters as *invariant
/// functions*. We offer special methods to add invariant functions during the model building process.
///
/// ### Base Functions
/// The varpro library expresses base function signatures as `$\vec{f}_j(\vec{x},p_1,...,p_{P_j}))$`, where
/// `$p_1,...,p_{P_J}$` are the paramters that the basefunction with index `$j$` actually depends on.
/// These are not given as a vector, but as a (variadic) list of arguments. The parameters must be
/// a subset of the model parameters. In order to add functions like this to a model, we must also
/// provide all partial derivatives (for parameters that the function explicitly depends on).
///
/// Refer to the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
/// to see how to construct a model with basis functions.
///
/// # Usage
/// The is no reason to interface with the separable model directly. First, construct it using a
/// [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder) and then pass the model
/// to one of the problem builders (e.g. [LevMarProblemBuilder](crate::solvers::levmar::builder::LevMarProblemBuilder))
/// to use it for nonlinear least squares fitting.
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

impl<ScalarType> SeparableModel<ScalarType>
where
    ScalarType: Scalar + Zero,
{
    /// # Arguments
    /// * `location`: the value of the independent location parameter `$\vec{x}$`
    /// * `parameters`: the parameter vector `$\vec{\alpha}$`
    /// # Result
    /// Evaluates the model in matrix form for the given nonlinear parameters. This produces
    /// a matrix where the columns of the matrix are given by the basis functions, evaluated in
    /// the order that they were added to the model. Assume the model consists of `$f_1(\vec{x},\vec{\alpha})$`,
    /// `$f_2(\vec{x},\vec{\alpha})$`, and `$f_3(\vec{x},\vec{\alpha})$` and the functions where
    /// added to the model builder in this particular order. Then
    /// the matrix is given as
    /// ```math
    ///   \mathbf{\Phi}(\vec{x},\vec{\alpha}) \coloneqq
    ///   \begin{pmatrix}
    ///   \vert & \vert & \vert \\
    ///   f_1(\vec{x},\vec{\alpha}) & f_2(\vec{x},\vec{\alpha}) & f_3(\vec{x},\vec{\alpha}) \\
    ///   \vert & \vert & \vert \\
    ///   \end{pmatrix},
    /// ```
    /// where, again, the function `$f_j$` gives the column values for colum `$j$` of `$\mathbf{\Phi}(\vec{x},\vec{\alpha})$`.
    /// Since model function is a linear combination of the functions `$f_j$`, the value of the modelfunction
    /// at these parameters can be obtained as the matrix vector product `$\mathbf{\Phi}(\vec{x},\vec{\alpha}) \, \vec{c}$`,
    /// where `$\vec{c}$` is a vector of the linear coefficients.
    /// ## Errors
    /// An error result is returned when
    /// * the parameters do not have the same length as the model parameters given when building the model
    /// * the basis functions do not produce a vector of the same length as the `location` argument `$\vec{x}$`
    pub fn eval(
        &self,
        location: &DVector<ScalarType>,
        parameters: &[ScalarType],
    ) -> Result<DMatrix<ScalarType>, ModelError> {
        if parameters.len() != self.parameter_count() {
            return Err(ModelError::IncorrectParameterCount {
                required: self.parameter_count(),
                actual: parameters.len(),
            });
        }

        let nrows = location.len();
        let ncols = self.basis_function_count();
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

    /// # Arguments
    /// * `location`: the value of the independent location parameter `$\vec{x}$`
    /// * `parameters`: the parameter vector `$\vec{\alpha}$`
    /// # Usage
    /// This function returns a proxy for syntactic sugar. Use it directly to call get the derivative
    /// matrix of model as `model.eval_deriv(&x,parameters).at(0)`. We can also access the derivative
    /// by name for convenience as `model.eval_deriv(&x,parameters).at_param_name("tau")`, which will
    /// produce the same result of the index based call if `"tau"` is the name of the first model
    /// parameter.
    /// **NOTE**: In code, the derivatives are indexed with index 0. The index is given by the order that the model
    /// parameters where given when building a model. Say our model was given model parameters
    /// `&["tau","omega"]`, then parameter `"tau"` corresponds to index `0` and parameter `"omega"`
    /// to index `1`. This means that the derivative `$\partial/\partial\tau$` has index `0` and
    /// `$\partial/\partial\omega$` has index 1.
    /// # Result
    /// The function returns a matrix where the derivatives of the model functions are
    /// forming the columns of the matrix. Assume the model consists of `$f_1(\vec{x},\vec{\alpha})$`,
    /// `$f_2(\vec{x},\vec{\alpha})$`, and `$f_3(\vec{x},\vec{\alpha})$` and the functions where
    /// added to the model builder in this particular order. Let the model parameters be denoted by
    /// `$\vec{\alpha}=(\alpha_1,\alpha_2,...,\alpha_N)$`, then this function returns the matrix
    /// ```math
    ///   \mathbf{D}_j(\vec{x},\vec{\alpha}) \coloneqq
    ///   \begin{pmatrix}
    ///   \vert & \vert & \vert \\
    ///   \frac{\partial f_1(\vec{x},\vec{\alpha})}{\partial \alpha_j} & \frac{\partial f_2(\vec{x},\vec{\alpha})}{\partial \alpha_j} & \frac{\partial f_3(\vec{x},\vec{\alpha})}{\partial \alpha_j} \\
    ///   \vert & \vert & \vert \\
    ///   \end{pmatrix},
    /// ```
    /// where in the code the index `j` of the derivative begins with `0` and goes to `N-1`.
    /// ## Errors
    /// An error result is returned when
    /// * the parameters do not have the same length as the model parameters given when building the model
    /// * the basis functions do not produce a vector of the same length as the `location` argument `$\vec{x}$`
    /// * the given parameter index is out of bounds
    /// * the given parameter name is not a parameter of the model.
    pub fn eval_deriv<'a, 'b, 'c, 'd>(
        &'a self,
        location: &'b DVector<ScalarType>,
        parameters: &'c [ScalarType],
    ) -> DerivativeProxy<'d, ScalarType>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
    {
        DerivativeProxy {
            basefunctions: &self.basefunctions,
            location,
            parameters,
            model_parameter_names: &self.parameter_names,
        }
    }
}

impl<ScalarType> SeparableNonlinearModel<ScalarType> for SeparableModel<ScalarType> 
    where ScalarType : Scalar + Zero {
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }

    fn basis_function_count(&self) -> usize {
        self.basefunctions.len()
    }

    fn eval(&self, location : &DVector<ScalarType>, parameters : &[ScalarType])-> Result<DMatrix<ScalarType>, Self::Error> {
        self.eval(location,parameters)
    }

    fn eval_partial_deriv(&self, location: &DVector<ScalarType>, parameters : &[ScalarType],derivative_index : usize) -> Result<DMatrix<ScalarType>, Self::Error> {
        self.eval_deriv(location,parameters).at(derivative_index)
    }

}

/// A helper proxy that is used in conjuntion with the method to evalue the derivative of a
/// separable model. This structure serves no purpose other than making the function call to
/// calculate the derivative a little more readable
#[must_use = "Derivative Proxy should be used immediately to evaluate a derivative matrix"]
pub struct DerivativeProxy<'a, ScalarType: Scalar> {
    basefunctions: &'a [ModelBasisFunction<ScalarType>],
    location: &'a DVector<ScalarType>,
    parameters: &'a [ScalarType],
    model_parameter_names: &'a [String],
}

impl<'a, ScalarType: Scalar + Zero> DerivativeProxy<'a, ScalarType> {
    /// This function is used in conjunction with evaluating the derivative matrix of the
    /// separable model. It is documented as part of the [SeparableModel] interface.
    #[inline]
    pub fn at(&self, param_index: usize) -> Result<DMatrix<ScalarType>, ModelError> {
        if self.parameters.len() != self.model_parameter_names.len() {
            return Err(ModelError::IncorrectParameterCount {
                required: self.model_parameter_names.len(),
                actual: self.parameters.len(),
            });
        }

        if param_index >= self.model_parameter_names.len() {
            return Err(ModelError::DerivativeIndexOutOfBounds { index: param_index });
        }

        let nrows = self.location.len();
        let ncols = self.basefunctions.len();
        let mut derivative_function_value_matrix =
            DMatrix::<ScalarType>::from_element(nrows, ncols, Zero::zero());

        for (basefunc, mut column) in self
            .basefunctions
            .iter()
            .zip(derivative_function_value_matrix.column_iter_mut())
        {
            if let Some(derivative) = basefunc.derivatives.get(&param_index) {
                let deriv_value = model_basis_function::evaluate_and_check(
                    derivative,
                    self.location,
                    self.parameters,
                )?;
                column.copy_from(&deriv_value);
            }
        }
        Ok(derivative_function_value_matrix)
    }

    /// This function is used in conjunction with evaluating the derivative matrix of the
    /// separable model. It is documented as part of the [SeparableModel] interface.
    /// Allows to access the derivative by parameter name instead of index.
    /// # Returns
    /// If the parameter is in the model parameters, returns the same result as calculating
    /// the derivative at the same parameter index. Otherwise returns an error indicating
    /// the parameter is not in the model parameters.
    #[inline]
    pub fn at_param_name<StrType: AsRef<str>>(
        &self,
        param_name: StrType,
    ) -> Result<DMatrix<ScalarType>, ModelError> {
        let index = self
            .model_parameter_names
            .iter()
            .position(|p| p == param_name.as_ref())
            .ok_or(ModelError::ParameterNotInModel {
                parameter: param_name.as_ref().into(),
            })?;
        self.at(index)
    }
}
