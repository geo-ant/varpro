pub mod errors;
mod detail;
mod basefunction;

#[cfg(test)]
mod test;
pub mod builder;

use nalgebra::base::{Scalar, Dim};
use nalgebra::{U1, Dynamic};
use nalgebra::Vector;
use nalgebra::base::storage::Owned;

use basefunction::*;


/// typedef for a vector that owns its data
pub type OwnedVector<ScalarType, Rows> = Vector<ScalarType, Rows, Owned<ScalarType, Rows, U1>>;

//TODO Document
//modelfunction f(x,alpha), where x is the independent variable, alpha: (potentially) nonlinear params
pub type BaseFuncType<ScalarType, NData> = Box<dyn Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, Dynamic>) -> OwnedVector<ScalarType, NData>>;


/// # Separable (Nonlinear) Model
/// A separable nonlinear model is the linear combination of a set of nonlinear basefunctions.
/// The basefunctions depend on a vector `alpha` of parameters. They basefunctions are commonly
/// nonlinear in the parameters `alpha` (but they don't have to be). Each individual base function
/// may also depend only on a subset of the model parameters `alpha`.
/// Further, the modelfunctions should be linearly independent to make the fit more numerically
/// robust.
/// Fitting a separable nonlinear model consists of finding the best combination of the parameters
/// for the linear combination of the functions and the nonlinear parameters.
pub struct SeparableModel<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the parameter names of the model. This defines the order in which the
    /// parameters are expected when the methods for evaluating the function
    /// values and the jacobian are called.
    /// The list of parameter contains a nonzero number of names and these names
    /// are unique.
    pub parameter_names: Vec<String>,
    /// the set of model. This already contains the model
    /// which are wrapped inside a lambda function so that they can take the
    /// parameter space of the model function set as an argument
    pub modelfunctions: Vec<ModelFunction<ScalarType, NData>>,
}

impl<ScalarType, NData> SeparableModel<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>
{
    /// Get the parameters of the model
    pub fn parameters(&self) -> &Vec<String> {
        &self.parameter_names
    }

    /// Get the number of nonlinear parameters of the model
    pub fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }

    //TODO
    // make methods to push a function which is dependent on some parameters and also
    // a method that pushes a function which is independent of parameters to the set
    // those could be different spatially varying backgrounds
    // so functions f(x, alpha) with derivatives with respect to alpha but also
    // a function type f(x) which does not depend on the nonlinear params alpha
}


