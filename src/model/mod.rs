pub mod errors;
mod detail;
mod basefunction;

#[cfg(test)]
mod test;

use nalgebra::base::{Scalar, Dim};
use nalgebra::{Matrix, U1, Dynamic, DimName, DimMax, DimMul};
use nalgebra::Vector;

use nalgebra::base::storage::Owned;

/// typedef for a vector that owns its data
pub type OwnedVector<ScalarType: Scalar, Rows: Dim> = Vector<ScalarType, Rows, Owned<ScalarType, Rows, U1>>;

//TODO Document
//modelfunction f(x,alpha), where x is the independent variable, alpha: (potentially) nonlinear params
pub type BaseFuncType<ScalarType: Scalar, NData: Dim> = Box<dyn Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, Dynamic>) -> OwnedVector<ScalarType, NData>>;

use errors::ModelfunctionError;
use std::hash::Hash;
use std::collections::{HashSet, BTreeMap};

use detail::*;

use basefunction::*;


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
          NData: Dim ,
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
    pub(in self) modelfunctions: Vec<Basefunction<ScalarType, NData>>,
}

impl<ScalarType, NData> SeparableModel<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>
{
    pub fn new<StrType>(param_names: Vec<StrType>) -> Result<Self, ModelfunctionError>
    where StrType : Into<String> {
        let parameter_names: Vec<String> = param_names.into_iter().map(|param| param.into()).collect();
        check_parameter_names(parameter_names.as_slice())?;
        Ok(Self {
            parameter_names,
            modelfunctions: Vec::default(),
        })
    }

    /// Get the parameters of the model
    pub fn parameters(&self) -> &Vec<String> {
        &self.parameter_names
    }

    /// Get the number of nonlinear parameters of the model
    pub fn parameter_count(&self) -> usize {
        self.parameter_names.len()
    }


    /// TODO Document and say to use it in conjunction with push
    pub fn independent_function<FuncType>(&mut self, function: FuncType) -> ParameterIndepententModelFunctionProxy<ScalarType, NData, FuncType>
        where FuncType: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> +'static {
        ParameterIndepententModelFunctionProxy::new(self, function)
    }

    //TODO
    // make methods to push a function which is dependent on some parameters and also
    // a method that pushes a function which is independent of parameters to the set
    // those could be different spatially varying backgrounds
    // so functions f(x, alpha) with derivatives with respect to alpha but also
    // a function type f(x) which does not depend on the nonlinear params alpha
}


