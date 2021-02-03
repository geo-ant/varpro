use nalgebra::base::{Dim, Scalar};
use nalgebra::Dynamic;

use super::BaseFuncType;
use super::SeparableModel;
use std::collections::HashMap;

use super::OwnedVector;
use crate::model::detail::{create_wrapper_function, check_parameter_names};
use crate::model::errors::Error;
use std::hash::Hash;

pub struct ModelFunction<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the function. Takes the full model parameters alpha.
    pub(self) function: BaseFuncType<ScalarType, NData>,
    /// the derivatives of the function by index (also taking the full parameters alpha).
    /// The index is based on the index of the parameters in the model function set.
    pub(self) derivatives: HashMap<usize, BaseFuncType<ScalarType, NData>>,
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

/// The modelfunction builder allows to create model functions that depend on
/// a subset or the whole model parameters. Functions that depend on model parameters
/// need to have partial derivatives provided for each parameter they depend on.
pub struct ModelfunctionBuilder<'a, ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// a reference to the model parameters for the model given in the constructor of this builder
    model_paramters: &'a Vec<String>,
    function_parameters : Vec<String>,
    model_function_result: Result<ModelFunction<ScalarType, NData>, Error>,
}

impl<'a, ScalarType, NData> ModelfunctionBuilder<'a, ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// begin constructing a modelfunction for a specific model. The modelfunction must take
    /// a subset of the model parameters. This is the first step in creating a function, because
    /// the modelfunction must have partial derivatives specified for each parameter it takes.
    /// # Arguments
    /// * `model`: the model for which this function should be generated. This is important
    /// so the builder understands how the parameters of the function relate to the parameters of the model
    /// * `function_parameters`: the parameters that the function takes. Those must be in the order
    /// of the parameter vector. The paramters must not be empty, nor may they contain duplicates
    /// * `function`: the actual function.
    /// # Result
    /// A model builder that can be used to add derivatives.
    pub fn with_function<StrType, FuncType>(
        model: &'a SeparableModel<ScalarType, NData>,
        function_parameters: &[StrType],
        function: FuncType,
    ) -> Self
    where
        StrType: Into<String> + Clone + Hash + Eq,
        String: PartialEq<StrType>,
        FuncType: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
    {
        // check that the function parameter list is valid, otherwise continue with an
        // internal error state
        if let Err(err) = check_parameter_names(function_parameters) {
            return Self {
                model_function_result: Err(err),
                model_paramters: model.parameters(),
                function_parameters : function_parameters.iter().cloned().map(|s|s.into()).collect(),
            };
        }

        let model_function_result = create_wrapper_function(model, function_parameters, function)
            .map(|function| ModelFunction {
                function,
                derivatives: Default::default()
            });

        Self {
            model_function_result,
            model_paramters : model.parameters(),
            function_parameters : function_parameters.iter().cloned().map(|s|s.into()).collect(),
        }
    }

    /// Add a derivative for the function with respect to the given parameter.
    /// # Arguments
    /// * `parameter`: the parameter with respect to which the derivative is taken.
    /// The parameter must be inside the set of model parameters. Furthermore the
    /// * `derivative`: the partial derivative of the function with which the
    /// builder was created.
    pub fn partial_deriv<FuncType>(self, parameter : &str, derivative : FuncType) -> Self
    where          FuncType: Fn(
        &OwnedVector<ScalarType, NData>,
        &OwnedVector<ScalarType, Dynamic>,
    ) -> OwnedVector<ScalarType, NData>
    + 'static,
    {
        //check to see if parameter is in list
        if let Some(index) = self.function_parameters.iter().position(|function_param|function_param==parameter) {

            //todo CONTINUE HERE: see if internal model function is an ok variant and if so
            // check that the hashmap does not already contain that value. If so, then
            // also return a value.

            // todo: this pattern of returning self from the builder instead of mutating the internal state
            // is much better. Also use that for the other builders!!

            if self.

        } else {
            return Self {
                model_function_result: Err(Error::InvalidDerivative { parameter: parameter.into(), function_parameters: self.function_parameters.clone()}),
                ..self
            }
        }
        unimplemented!()
    }

}
