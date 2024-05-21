use std::collections::HashMap;

use crate::model::errors::ModelError;
use nalgebra::base::Scalar;
use nalgebra::DVector;

/// Helper type that is a typedef for a function `$f(\vec{x},\vec{\alpha})$`, where
/// the first argument is a location argument, and the second argument is the
/// (nonlinear) parameters. This is the most low level representation of how our
/// wrapped functions are actually stored inside the model functions
type BaseFuncType<ScalarType> =
    Box<dyn Fn(&DVector<ScalarType>, &[ScalarType]) -> DVector<ScalarType>>;

/// An internal type that is used to store basefunctions whose interface has been wrapped in
/// such a way that they can accept the location and the *complete model parameters as arguments*.
#[doc(hidden)]
pub struct ModelBasisFunction<ScalarType>
where
    ScalarType: Scalar,
{
    /// the function. Takes the full model parameters alpha.
    pub function: BaseFuncType<ScalarType>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    /// If a derivative with respect to a given model parameter index does not exist, it
    /// means this function does not depend on that model parameter
    pub derivatives: HashMap<usize, BaseFuncType<ScalarType>>,
}

impl<ScalarType: Scalar> std::fmt::Debug for ModelBasisFunction<ScalarType> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBasisFunction")
            .field("function", &"/* omitted */")
            .field("derivatives", &"/* omitted */")
            .finish()
    }
}

impl<ScalarType> ModelBasisFunction<ScalarType>
where
    ScalarType: Scalar,
{
    /// Create a function that does not depend on any model parameters and just
    /// takes a location parameter as its argument.
    /// To create parameter dependent model basis functions use the [ModelBasisFunctionBuilder].
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

#[inline]
/// Helper function to evaluate the given function with the location and parameters
/// and make sure that the output vector size of the function has the same length as
/// the location vector. Otherwise returns an error.
pub fn evaluate_and_check<ScalarType: Scalar>(
    func: &BaseFuncType<ScalarType>,
    location: &DVector<ScalarType>,
    parameters: &[ScalarType],
) -> Result<DVector<ScalarType>, ModelError> {
    let result = (func)(location, parameters);
    if result.len() == location.len() {
        Ok(result)
    } else {
        Err(ModelError::UnexpectedFunctionOutput {
            expected_length: location.len(),
            actual_length: result.len(),
        })
    }
}
