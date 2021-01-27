pub mod errors;
mod detail;

use nalgebra::base::{Scalar, Dim};
use nalgebra::{Matrix, U1, Dynamic, DimName, DimMax, DimMul};
use nalgebra::Vector;

use crate::types::*;
use nalgebra::base::storage::Owned;

/// internal type that indicates a vector that owns its data
type OwnedVector<ScalarType: Scalar, Rows: Dim> = Vector<ScalarType, Rows, Owned<ScalarType, Rows, U1>>;

//TODO Document
//modelfunction f(x,alpha), where x is the independent variable, alpha: (potentially) nonlinear params
type BaseFuncType<ScalarType: Scalar, NData: Dim> = Box<dyn Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, Dynamic>) -> OwnedVector<ScalarType, NData>>;

use errors::ModelfunctionError;
use std::hash::Hash;
use std::collections::HashSet;

use detail::*;

struct Basefunction<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim + DimName,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    function : BaseFuncType<ScalarType,NData>,
    derivatives : HashSet<usize,BaseFuncType<ScalarType,NData>>,
}

/// A set of modelfunctions that can be used to fit a separable problem.
/// The modelfunctions are typically nonlinear in their parameters, but they don't have to be.
/// Further, the modelfunctions should be linearly independent to make the fit more numerically
/// robust.
pub struct SeparableModel<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim + DimName,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
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
    modelfunctions: Vec<Basefunction<ScalarType, NData>>,
}

impl<S, N> SeparableModel<S, N>
    where S: Scalar,
          N: Dim + DimName,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<S, N>
{
    pub fn new<StrType: Into<String>>(param_names: Vec<StrType>) -> Result<Self, ModelfunctionError> {
        let parameter_names: Vec<String> = param_names.into_iter().map(|param| param.into()).collect();
        check_parameter_names(parameter_names.as_slice())?;
        Ok(Self {
            parameter_names,
            modelfunctions: Vec::default(),
        })
    }

    /// Get the parameters of the
    pub fn parameter_names(&self) -> &Vec<String> {
        &self.parameter_names
    }

    //TODO
    // make methods to push a function which is dependent on some parameters and also
    // a method that pushes a function which is independent of parameters to the set
    // those could be different spatially varying backgrounds
    // so functions f(x, alpha) with derivatives with respect to alpha but also
    // a function type f(x) which does not depend on the nonlinear params alpha
}


