#[cfg(test)]
mod test;

use nalgebra::base::{Dim, Scalar};
use nalgebra::Dynamic;

use crate::model::detail::{check_parameter_names, create_index_mapping, create_wrapper_function};
use crate::model::errors::ModelError;
use crate::model::modelfunction::ModelFunction;
use crate::model::OwnedVector;

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
    model_function_result: Result<ModelFunction<ScalarType, NData>, ModelError>,
}

impl<ScalarType, NData> ModelFunctionBuilder<ScalarType, NData>
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
        model_parameters: Vec<String>,
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
                function_parameters: function_parameters.to_vec(),
            };
        }

        let model_function_result =
            create_wrapper_function(&model_parameters, function_parameters, function).map(
                |function| ModelFunction {
                    function,
                    derivatives: Default::default(),
                },
            );

        Self {
            model_function_result,
            model_parameters,
            function_parameters: function_parameters.to_vec(),
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
        //this makes sure that the index of the derivative is calculated with respect to the
        //model parameter list while also making sure that the given derivative exists in the function
        //parameters
        if let Some((deriv_index_in_model, _)) = self
            .model_parameters
            .iter()
            .enumerate()
            .filter(|(_idx, model_param)| self.function_parameters.contains(model_param))
            .find(|(_idx, model_param)| model_param == &parameter)
        {
            if let Ok(model_function) = self.model_function_result.as_mut() {
                match create_wrapper_function(
                    &self.model_parameters,
                    &self.function_parameters,
                    derivative,
                ) {
                    Ok(deriv) => {
                        // push derivative and check that the derivative was not already in the set
                        if model_function
                            .derivatives
                            .insert(deriv_index_in_model, deriv)
                            .is_some()
                        {
                            self.model_function_result = Err(ModelError::DuplicateDerivative {
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
                model_function_result: Err(ModelError::InvalidDerivative {
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
    pub fn build(self) -> Result<ModelFunction<ScalarType, NData>, ModelError> {
        if self.model_function_result.is_ok() {
            self.check_completion()?;
        }
        self.model_function_result
    }

    /// If the modelfunction builder is carrying an Error result, this function returns Ok(()).
    /// If it carries an ok variant, then this function checks that the modelfunction
    /// has all necessary derivatives provided and if so returns Ok(()), otherwise it returns
    /// an error indicating that there are missing derivatives
    fn check_completion(&self) -> Result<(), ModelError> {
        match self.model_function_result.as_ref() {
            Ok(modelfunction) => {
                // this should not go wrong, but to be safe
                check_parameter_names(self.model_parameters.as_slice())?;
                check_parameter_names(self.function_parameters.as_slice())?;
                // create the index mapping
                let index_mapping = create_index_mapping(
                    self.model_parameters.as_slice(),
                    self.function_parameters.as_slice(),
                )?;
                // now make sure that the derivatives are provided for all indices of that mapping
                for (index, parameter) in index_mapping.iter().zip(self.function_parameters.iter())
                {
                    if !modelfunction.derivatives.contains_key(index) {
                        return Err(ModelError::MissingDerivative {
                            missing_parameter: parameter.clone(),
                            function_parameters: self.function_parameters.clone(),
                        });
                    }
                }
                // this is a sanity check. if this came this far, there should not be an error here
                if index_mapping.len() != modelfunction.derivatives.len() {
                    // this also should never be the case and indicates a programming error in the library
                    return Err(ModelError::Fatal {
                        message: "Too many or too few derivatives in set!".to_string(),
                    });
                }
                // otherwise
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }
}

