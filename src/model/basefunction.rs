use nalgebra::base::{Scalar, Dim};
use nalgebra::{Dynamic, DimName};
use nalgebra::Vector;

use super::BaseFuncType;
use super::SeparableModel;
use std::collections::{HashSet, HashMap};


use super::OwnedVector;
use crate::model::errors::ModelfunctionError;

pub struct Basefunction<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the function. Takes the full model parameters alpha.
    function: BaseFuncType<ScalarType, NData>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    derivatives: HashMap<usize, BaseFuncType<ScalarType, NData>>,
}

impl<ScalarType, NData> Basefunction<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    pub fn parameter_independent<FuncType>(function: FuncType) -> Self
        where FuncType: Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, Dynamic>) -> OwnedVector<ScalarType, NData> +'static {
        Self {
            function : Box::new(function),
            derivatives : HashMap::default(),
        }
    }
}

/// Internal type that is not useful in and of itself and only gets generated as part of
/// pushing a function that does not depend on the model parameter to a separable model.
#[must_use="This alone will not push the function to the set. To do that, append a call to 'push()'."]
pub struct ParameterIndepententModelFunctionProxy<'a, ScalarType, NData, FuncType>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>,
          FuncType: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static
{
    model: &'a mut SeparableModel<ScalarType, NData>,
    function: FuncType,
}

impl<'a, ScalarType, NData, FuncType> ParameterIndepententModelFunctionProxy<'a, ScalarType, NData, FuncType>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>,
          FuncType: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> +'static {

    pub(in crate::model) fn new(model : &'a mut SeparableModel<ScalarType, NData>, function : FuncType) -> Self {
        Self {
            model,
            function
        }
    }

    /// push the function to the set of model functions. This function is
    /// used after the call to [new_independent_function] of [SeparableModel] to push the function
    /// into the model. This function always returns with an Ok-variant but the interface is
    /// the same as for parameter dependent functions.
    pub fn push(self)->Result<(),ModelfunctionError> {
        let Self{ model, function } = self;
        model.modelfunctions.push(Basefunction::parameter_independent(move |x , _model_params| (function)(x)));
        Ok(())
    }
}


