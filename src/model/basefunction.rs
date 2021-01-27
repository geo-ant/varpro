
use nalgebra::base::{Scalar, Dim};
use nalgebra::{Matrix, U1, Dynamic, DimName, DimMax, DimMul};
use nalgebra::Vector;

use super::BaseFuncType;
use std::collections::HashSet;


pub struct Basefunction<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim + DimName,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the function. Takes the full model parameters alpha.
    function: BaseFuncType<ScalarType, NData>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    derivatives: HashSet<usize, BaseFuncType<ScalarType, NData>>,
}