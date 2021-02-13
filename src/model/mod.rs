use crate::model::modelfunction::ModelFunction;
use nalgebra::base::Scalar;
use nalgebra::{DMatrix, DVector};
use num_traits::Zero;

mod detail;
pub mod errors;

pub mod builder;
pub mod modelfunction;
#[cfg(test)]
mod test;

//TODO Document
//modelfunction f(x,alpha), where x is the independent variable, alpha: (potentially) nonlinear params
pub type BaseFuncType<ScalarType> =
    Box<dyn Fn(&DVector<ScalarType>, &DVector<ScalarType>) -> DVector<ScalarType>>;

/// # A Separable Nonlinear Model
/// TODO Document:
/// A separable nonlinear model is a (nonlinear) function `$f(\vec{x},\vec{\alpha})$` which depends on
/// * the independent variable `$\vec{x}$`, e.g. a location, time, etc...
/// * the actual model parameters `$\vec{\alpha}$`.
///
/// *Separable* means that the nonlinear model function can be written as the
/// linear combination of nonlinear base functions, i.e.
/// ```math
/// f(\vec{x},\vec{\alpha}) = \sum_j f_j(\vec{x},\vec{\alpha})
/// ```
/// The base functions `$f_j$` typically depend on individual subsets of the model parameters `$\vec{\alpha}$`.
///
/// ## Base Functions
/// It perfectly fine for a base function to depend on all or none of the model parameters or any
/// subset of the model parameters.
/// ### Invariant Functions
/// We refer to functions `$f_j(\vec{x})$` that depend on none of the model parameters as *invariant
/// functions*. We offer special methods to add invariant functions during the model building process.
/// ### Base Functions
/// TODO FURTHER: can depend on subset, must have derivatives, should be nonlinear and should be
/// linearly independent
/// TODO FURTHER
///
///
/// A separable nonlinear model is the linear combination of a set of nonlinear basefunctions.
/// The basefunctions depend on a vector `alpha` of parameters. They basefunctions are commonly
/// nonlinear in the parameters `alpha` (but they don't have to be). Each individual base function
/// may also depend only on a subset of the model parameters `alpha`.
/// Further, the modelfunctions should be linearly independent to make the fit more numerically
/// robust.
/// Fitting a separable nonlinear model consists of finding the best combination of the parameters
/// for the linear combination of the functions and the nonlinear parameters.
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
    /// the set of model. This already contains the model
    /// which are wrapped inside a lambda function so that they can take the
    /// parameter space of the model function set as an argument
    modelfunctions: Vec<ModelFunction<ScalarType>>,
}

impl<ScalarType> SeparableModel<ScalarType>
where
    ScalarType: Scalar,
{
    /// Get the parameters of the model
    pub fn parameters(&self) -> &Vec<String> {
        &self.parameter_names
    }

    /// Get the number of nonlinear parameters of the model
    pub fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }

    /// Get the model functions that comprise the model
    pub fn functions(&self) -> &[ModelFunction<ScalarType>] {
        self.modelfunctions.as_slice()
    }
}

//TODO: find out if this will really work!!!!!!
impl<ScalarType> SeparableModel<ScalarType>
where
    ScalarType: Scalar + Zero,
{
    pub fn eval(
        &self,
        location: &DVector<ScalarType>,
        parameters: &DVector<ScalarType>,
    ) -> DMatrix<ScalarType> {
        // helpful links:
        //https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi6-v2O1eLuAhVqDWMBHU-tBqMQFjAAegQIARAC&url=https%3A%2F%2Fdiscourse.nphysics.org%2Ft%2Fusing-nalgebra-in-generics%2F90&usg=AOvVaw1rBwzIclum71sfw-6Ejvlh
        //https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi6-v2O1eLuAhVqDWMBHU-tBqMQFjABegQIBBAC&url=https%3A%2F%2Fdiscourse.nphysics.org%2Ft%2Fhelp-understanding-trait-bounds-required-allocators%2F440&usg=AOvVaw2mbvY28cSFLzTzRnX9rWAN

        //todo include panics or errors if functions don't produce correct length. Maybe better error!

        let nrows = location.len();
        let ncols = self.modelfunctions.len();
        //todo: optimize this by using unsafe initialization (https://docs.rs/nalgebra/0.24.1/nalgebra/base/struct.Matrix.html#method.new_uninitialized_generic)
        // https://docs.rs/nalgebra/0.24.1/nalgebra/base/struct.Matrix.html#generic-constructors
        // phew, this was hard to understand how to make that work (so far)
        let mut function_value_matrix =
            DMatrix::<ScalarType>::from_element(nrows, ncols, Zero::zero());

        for (basefunc, mut column) in self
            .modelfunctions
            .iter()
            .zip(function_value_matrix.column_iter_mut())
        {
            let function_value = (basefunc.function)(location, parameters);
            column.copy_from(&function_value);
        }
        function_value_matrix
    }
}
