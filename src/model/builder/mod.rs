use std::hash::Hash;

use nalgebra::{Scalar, DVector};

use crate::model::builder::modelfunction_builder::ModelFunctionBuilder;
use crate::model::detail::check_parameter_names;
use crate::model::errors::ModelError;
use crate::model::modelfunction::ModelFunction;
use crate::model::SeparableModel;

mod modelfunction_builder;

#[cfg(test)]
mod test;

// //todo document
pub struct SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    model_result: Result<SeparableModel<ScalarType>, ModelError>,
}

/// This trait can be used to extend an existing model with more functions.
impl<ScalarType> From<SeparableModel<ScalarType>>
    for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(model: SeparableModel<ScalarType>) -> Self {
        Self {
            model_result: Ok(model),
        }
    }
}

impl<ScalarType> From<ModelError> for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelError) -> Self {
        Self {
            model_result: Err(err),
        }
    }
}

impl<ScalarType> SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    //todo document
    pub fn new<StrType>(parameter_names: &[StrType]) -> Self
    where
        StrType: Clone + Into<String> + Eq + Hash,
        String: From<StrType>,
    {
        if let Err(parameter_error) = check_parameter_names(&parameter_names) {
            Self {
                model_result: Err(parameter_error),
            }
        } else {
            let parameter_names = parameter_names
                .iter()
                .map(|name| name.clone().into())
                .collect();
            let model_result = Ok(SeparableModel {
                parameter_names,
                modelfunctions: Vec::default(),
            });
            Self { model_result }
        }
    }

    //todo document
    pub fn invariant_function<F>(mut self, function: F) -> Self
    where
        F: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        if let Ok(model) = self.model_result.as_mut() {
            model
                .modelfunctions
                .push(ModelFunction::parameter_independent(function));
        }
        self
    }

    //todo document
    pub fn function<F>(
        self,
        function_params: &[String],
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: Fn(
                &DVector<ScalarType>,
                &DVector<ScalarType>,
            ) -> DVector<ScalarType>
            + 'static,
    {
        SeparableModelBuilderProxyWithDerivatives::new(self.model_result, function_params, function)
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelError> {
        self.model_result.and_then(check_validity)
    }
}

/// a helper function that checks a model for validity.
/// # Returns
/// If the model is valid, then the model is returned as an ok variant, otherwise an error variant
/// A model is valid, when
///  * the model has at least one modelfunction, and
///  * for each model parameter we have at least one function that depends on this
///    parameter.
fn check_validity<ScalarType>(
    model: SeparableModel<ScalarType>,
) -> Result<SeparableModel<ScalarType>, ModelError>
where
    ScalarType: Scalar,
{
    if model.modelfunctions.is_empty() {
        Err(ModelError::EmptyModel)
    }
    else {
        // now check that all model parameters are referenced in at least one parameter of one
        // of the given model functions. We do this by checking the indices of the derivatives
        // because each model parameter must occur at least once as a key in at least one of the
        // modelfunctions derivatives
        for (param_index,parameter_name) in model.parameter_names.iter().enumerate() {
            if !model.modelfunctions.iter().any(|function|function.derivatives.contains_key(&param_index)) {
                return Err(ModelError::UnusedParameter {parameter: parameter_name.clone()})
            }
        }
        // otherwise return this model
        Ok(model)
    }
}

/// helper struct that contains a seperable model as well as a model function builder
/// used inside the SeparableModelBuilderProxyWithDerivatives.
struct ModelAndFunctionbuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    model: SeparableModel<ScalarType>,
    builder: ModelFunctionBuilder<ScalarType>,
}

impl<ScalarType> ModelAndFunctionbuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    fn new(
        model: SeparableModel<ScalarType>,
        builder: ModelFunctionBuilder<ScalarType>,
    ) -> Self {
        Self { model, builder }
    }
}

pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    current_result: Result<ModelAndFunctionbuilderPair<ScalarType>, ModelError>,
}

impl<ScalarType> From<ModelError>
    for SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelError) -> Self {
        Self {
            current_result: Err(err),
        }
    }
}

impl<ScalarType> SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    //todo document
    fn new<F>(
        model_result: Result<SeparableModel<ScalarType>, ModelError>,
        function_parameters: &[String],
        function: F,
    ) -> Self
    where
        F: Fn(
                &DVector<ScalarType>,
                &DVector<ScalarType>,
            ) -> DVector<ScalarType>
            + 'static,
    {
        match model_result {
            Ok(model) => {
                let model_parameters = model.parameters().clone();
                Self {
                    current_result: Ok(ModelAndFunctionbuilderPair::new(
                        model,
                        ModelFunctionBuilder::new(model_parameters, function_parameters, function),
                    )),
                }
            }
            Err(err) => Self {
                current_result: Err(err),
            },
        }
    }

    //todo document
    pub fn partial_deriv<StrType: AsRef<str>, F>(self, parameter: StrType, derivative: F) -> Self
    where
        F: Fn(
                &DVector<ScalarType>,
                &DVector<ScalarType>,
            ) -> DVector<ScalarType>
            + 'static,
    {
        match self.current_result {
            Ok(result) => Self {
                current_result: Ok(ModelAndFunctionbuilderPair {
                    model: result.model,
                    builder: result.builder.partial_deriv(parameter.as_ref(), derivative),
                }),
            },
            Err(err) => Self::from(err),
        }
    }

    //todo document
    pub fn invariant_function<F>(self, function: F) -> SeparableModelBuilder<ScalarType>
    where
        F: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        match self.current_result {
            Ok(result) => {
                let ModelAndFunctionbuilderPair { mut model, builder } = result;
                if let Err(err) = builder.build().map(|func| model.modelfunctions.push(func)) {
                    SeparableModelBuilder::from(err)
                } else {
                    SeparableModelBuilder::from(model).invariant_function(function)
                }
            }
            Err(err) => SeparableModelBuilder::from(err),
        }
    }

    //todo document
    pub fn function<F>(
        self,
        function_params: &[String],
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: Fn(
                &DVector<ScalarType>,
                &DVector<ScalarType>,
            ) -> DVector<ScalarType>
            + 'static,
    {
        match self.current_result {
            Ok(result) => {
                let ModelAndFunctionbuilderPair { mut model, builder } = result;
                if let Err(err) = builder.build().map(|func| model.modelfunctions.push(func)) {
                    SeparableModelBuilderProxyWithDerivatives::from(err)
                } else {
                    SeparableModelBuilderProxyWithDerivatives::new(
                        Ok(model),
                        function_params,
                        function,
                    )
                }
            }
            Err(err) => SeparableModelBuilderProxyWithDerivatives::from(err),
        }
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelError> {
        // this method converts the internal results into a separable model and uses its
        // facilities to check for completion and the like
        match self.current_result {
            Ok(result) => {
                let ModelAndFunctionbuilderPair { mut model, builder } = result;
                if let Err(err) = builder.build().map(|func| model.modelfunctions.push(func)) {
                    SeparableModelBuilder::from(err).build()
                } else {
                    SeparableModelBuilder::from(model).build()
                }
            }
            Err(err) => SeparableModelBuilder::from(err).build(),
        }
    }
}
