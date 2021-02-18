use crate::model::model_basis_function::ModelBasisFunction;
use nalgebra::base::Scalar;

mod detail;
pub mod errors;

pub mod builder;
mod model_basis_function;
#[cfg(test)]
mod test;

/// This structure represents a separable nonlinear model.A separable nonlinear model is
/// a (nonlinear) function `$f(\vec{x},\vec{\alpha})$` which depends on
/// * the independent variable `$\vec{x}$`, e.g. a location, time, etc...
/// * the actual model parameters `$\vec{\alpha}$`.
///
/// *Separable* means that the nonlinear model function can be written as the
/// linear combination of `$M$` nonlinear base functions, i.e.
/// ```math
/// f(\vec{x},\vec{\alpha}) = \sum_{j=1}^M c_j \cdot f_j(\vec{x},\vec{\alpha}),
/// ```
/// where `\vec{c}=(c_1,\dots,\c_M)` are the coefficients of the model basis functions.
///
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
    pub fn parameters(&self) -> &Vec<String> {
        &self.parameter_names
    }

    /// Get the number of nonlinear parameters of the model
    pub fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }
}
//
// //TODO: find out if this will really work!!!!!!
// impl<ScalarType> SeparableModel<ScalarType>
// where
//     ScalarType: Scalar + Zero,
// {
//     /// TODO DOCUMENT
//     pub fn eval(
//         &self,
//         location: &DVector<ScalarType>,
//         parameters: &[ScalarType],
//     ) -> Result<DMatrix<ScalarType>, ModelError> {
//         //todo: check that parameters.len() is indeed the expected length
//         let nrows = location.len();
//         let ncols = self.basefunctions.len();
//         let mut function_value_matrix =
//             unsafe { DMatrix::<ScalarType>::new_uninitialized(nrows, ncols) };
//
//         for (basefunc, mut column) in self
//             .basefunctions
//             .iter()
//             .zip(function_value_matrix.column_iter_mut())
//         {
//             let function_value =
//                 model_basis_function::evaluate_and_check(&basefunc.function, location, parameters)?;
//             column.copy_from(&function_value);
//         }
//         Ok(function_value_matrix)
//     }
//
//     /// TODO DOCUMENT
//     pub fn deriv<'a, 'b, 'c, 'd>(
//         &'a self,
//         location: &'b DVector<ScalarType>,
//         parameters: &'c [ScalarType],
//     ) -> DerivativeProxy<'d, ScalarType>
//     where
//         'a: 'd,
//         'b: 'd,
//         'c: 'd,
//     {
//         DerivativeProxy {
//             basefunctions: &self.basefunctions,
//             location,
//             parameters,
//             model_parameter_names: &self.parameter_names,
//         }
//     }
// }
// /// A helper proxy that is used in conjuntion with the method to evalue the derivative of a
// /// separable model. This structure serves no purpose other than making the function call to
// /// calculate the derivative a little more readable
// pub struct DerivativeProxy<'a, ScalarType: Scalar> {
//     basefunctions: &'a [ModelBasisFunction<ScalarType>],
//     location: &'a DVector<ScalarType>,
//     parameters: &'a [ScalarType],
//     model_parameter_names: &'a [String],
// }
//
// impl<'a, ScalarType: Scalar + Zero> DerivativeProxy<'a, ScalarType> {
//     /// TODO DOCUMENT
//     #[inline]
//     pub fn eval_at_param_index(&self, index: usize) -> Result<DMatrix<ScalarType>, ModelError> {
//         if index >= self.model_parameter_names.len() {
//             return Err(ModelError::DerivativeIndexOutOfBounds { index });
//         }
//
//         let nrows = self.location.len();
//         let ncols = self.basefunctions.len();
//         let mut derivative_function_value_matrix =
//             DMatrix::<ScalarType>::from_element(nrows, ncols, Zero::zero());
//
//         for (basefunc, mut column) in self
//             .basefunctions
//             .iter()
//             .zip(derivative_function_value_matrix.column_iter_mut())
//         {
//             if let Some(derivative) = basefunc.derivatives.get(&index) {
//                 let deriv_value = model_basis_function::evaluate_and_check(
//                     derivative,
//                     self.location,
//                     self.parameters,
//                 )?;
//                 column.copy_from(&deriv_value);
//             }
//         }
//         Ok(derivative_function_value_matrix)
//     }
//
//     /// Convenience method that allows to calculate the derivative of the function value matrix
//     /// by giving the parameter name.
//     /// # Returns
//     /// If the parameter is in the model parameters, returns the same result as calculating
//     /// the derivative at the same parameter index. Otherwise returns an error indicating
//     /// the parameter is not in the model parameters.
//     #[inline]
//     pub fn eval_at_param_name<StrType: AsRef<str>>(
//         &self,
//         param_name: StrType,
//     ) -> Result<DMatrix<ScalarType>, ModelError> {
//         let index = self
//             .model_parameter_names
//             .iter()
//             .position(|p| p == param_name.as_ref())
//             .ok_or(ModelError::ParameterNotInModel {
//                 parameter: param_name.as_ref().into(),
//             })?;
//         self.eval_at_param_index(index)
//     }
// }
