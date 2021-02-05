use nalgebra::base::{Dim, Scalar};
use nalgebra::Dynamic;

use crate::model::BaseFuncType;
use crate::model::SeparableModel;
use std::collections::HashMap;

use crate::model::OwnedVector;
use crate::model::detail::{check_parameter_names, create_wrapper_function};
use crate::model::errors::ModelError;
use std::hash::Hash;

// todo comment
pub struct ModelFunction<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the function. Takes the full model parameters alpha.
    pub function: BaseFuncType<ScalarType, NData>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    pub derivatives: HashMap<usize, BaseFuncType<ScalarType, NData>>,
}

// TODO document
impl<ScalarType, NData> ModelFunction<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// Create a function that does not depend on any model parameters and just
    /// takes a location parameter as its argument.
    pub fn parameter_independent<FuncType>(function: FuncType) -> Self
    where
        FuncType: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
    {
        Self {
            function: Box::new(move |x, _params| (function)(x)),
            derivatives: HashMap::default(),
        }
    }
}
