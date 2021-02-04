use nalgebra::base::{Dim, Scalar};
use nalgebra::Dynamic;

use crate::model::BaseFuncType;
use crate::model::SeparableModel;
use std::collections::HashMap;

use crate::model::OwnedVector;
use crate::model::detail::{check_parameter_names, create_wrapper_function};
use crate::model::errors::Error;
use std::hash::Hash;
use crate::model::modelfunction::ModelFunction;


/// The modelfunction builder allows to create model functions that depend on
/// a subset or the whole model parameters. Functions that depend on model parameters
/// need to have partial derivatives provided for each parameter they depend on.
pub struct ModelFunctionBuilder<ScalarType, NData>
    where
        ScalarType: Scalar,
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the model parameters for the model this function belongs to
    model_parameters: Vec<String>,
    /// the parameters that the function depends on. Must be a subset of the model parameters
    function_parameters: Vec<String>,
    /// the current result of the building process of the model function
    model_function_result: Result<ModelFunction<ScalarType, NData>, Error>,
}

impl<'a, ScalarType, NData> ModelFunctionBuilder<ScalarType, NData>
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
    pub fn new<FuncType>(
        model_parameters : Vec<String>,
        function_parameters: &[String],
        function: FuncType,
    ) -> Self
        where
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
                model_parameters,
                function_parameters: function_parameters
                    .iter()
                    .cloned()
                    .map(|s| s.into())
                    .collect(),
            };
        }

        let model_function_result = create_wrapper_function(&model_parameters, function_parameters, function)
            .map(|function| ModelFunction {
                function,
                derivatives: Default::default(),
            });

        Self {
            model_function_result,
            model_parameters ,
            function_parameters: function_parameters
                .iter()
                .cloned()
                .map(|s| s.into())
                .collect(),
        }
    }

    /// Add a derivative for the function with respect to the given parameter.
    /// # Arguments
    /// * `parameter`: the parameter with respect to which the derivative is taken.
    /// The parameter must be inside the set of model parameters. Furthermore the
    /// * `derivative`: the partial derivative of the function with which the
    /// builder was created.
    pub fn partial_deriv<FuncType>(mut self, parameter: &str, derivative: FuncType) -> Self
        where
            FuncType: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
    {
        //check to see if parameter is in list
        if let Some(deriv_index) = self
            .function_parameters
            .iter()
            .position(|function_param| function_param == parameter)
        {
            if let Ok(model_function) = self.model_function_result.as_mut() {
                match create_wrapper_function(&self.model_parameters, &self.function_parameters, derivative) {
                    Ok(deriv) => {
                        // push derivative and check that the derivative was not already in the set
                        if model_function
                            .derivatives
                            .insert(deriv_index, deriv)
                            .is_some()
                        {
                            self.model_function_result = Err(Error::DuplicateDerivative {
                                parameter: parameter.into(),
                            });
                        }
                    }
                    Err(error) => {
                        self.model_function_result = Err(error);
                    }
                }
            }
            self
        } else {
            Self {
                model_function_result: Err(Error::InvalidDerivative {
                    parameter: parameter.into(),
                    function_parameters: self.function_parameters.clone(),
                }),
                ..self
            }
        }
    }

    /// Build a modelfunction with derivatives from the contents of this builder
    /// # Result
    /// A modelfunction containing the given function and derivatives or an error
    /// variant if an error occurred during the building process
    fn build(self) -> Result<ModelFunction<ScalarType,NData>,Error>{
        self.model_function_result
    }
}
