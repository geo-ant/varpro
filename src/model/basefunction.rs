use nalgebra::base::{Scalar, Dim};
use nalgebra::{Dynamic};


use super::BaseFuncType;
use super::SeparableModel;
use std::collections::{HashMap};


use super::OwnedVector;

pub struct Basefunction<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim ,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the function. Takes the full model parameters alpha.
    pub (in crate) function: BaseFuncType<ScalarType, NData>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    pub (in crate) derivatives: HashMap<usize, BaseFuncType<ScalarType, NData>>,
}

// TODO document
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
