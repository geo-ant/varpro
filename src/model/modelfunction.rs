use nalgebra::base::Scalar;

use crate::model::BaseFuncType;

use nalgebra::DVector;
use std::collections::HashMap;

/// An internal type that is used to store basefunctions whose interface has been wrapped in
/// such a way that they can accept the location and the complete model parameters as arguments
pub struct ModelBaseFunction<ScalarType>
where
    ScalarType: Scalar,
{
    /// the function. Takes the full model parameters alpha.
    pub function: BaseFuncType<ScalarType>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    pub derivatives: HashMap<usize, BaseFuncType<ScalarType>>,
}

// TODO document
impl<ScalarType> ModelBaseFunction<ScalarType>
where
    ScalarType: Scalar,
{
    /// Create a function that does not depend on any model parameters and just
    /// takes a location parameter as its argument.
    pub fn parameter_independent<FuncType>(function: FuncType) -> Self
    where
        FuncType: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        Self {
            function: Box::new(move |x, _params| (function)(x)),
            derivatives: HashMap::default(),
        }
    }
}
