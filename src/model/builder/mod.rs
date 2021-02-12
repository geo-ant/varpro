use std::hash::Hash;

use nalgebra::{Dim, Dynamic, Scalar};

use crate::model::builder::modelfunction_builder::ModelFunctionBuilder;
use crate::model::detail::check_parameter_names;
use crate::model::errors::ModelError;
use crate::model::modelfunction::ModelFunction;
use crate::model::{OwnedVector, SeparableModel};

mod modelfunction_builder;

#[cfg(test)]
mod test;

// //todo document
pub struct SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    model_result: Result<SeparableModel<ScalarType, NData>, ModelError>,
}

/// This trait can be used to extend an existing model with more functions.
impl<ScalarType, NData> From<SeparableModel<ScalarType, NData>>
    for SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn from(model: SeparableModel<ScalarType, NData>) -> Self {
        Self {
            model_result: Ok(model),
        }
    }
}

impl<ScalarType, NData> From<ModelError> for SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn from(err: ModelError) -> Self {
        Self {
            model_result: Err(err),
        }
    }
}

impl<ScalarType, NData> SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
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
        F: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
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
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        F: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
    {
        SeparableModelBuilderProxyWithDerivatives::new(self.model_result, function_params, function)
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType, NData>, ModelError> {
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
fn check_validity<ScalarType, NData>(
    model: SeparableModel<ScalarType, NData>,
) -> Result<SeparableModel<ScalarType, NData>, ModelError>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
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
struct ModelAndFunctionbuilderPair<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    model: SeparableModel<ScalarType, NData>,
    builder: ModelFunctionBuilder<ScalarType, NData>,
}

impl<ScalarType, NData> ModelAndFunctionbuilderPair<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn new(
        model: SeparableModel<ScalarType, NData>,
        builder: ModelFunctionBuilder<ScalarType, NData>,
    ) -> Self {
        Self { model, builder }
    }
}

pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    current_result: Result<ModelAndFunctionbuilderPair<ScalarType, NData>, ModelError>,
}

impl<ScalarType, NData> From<ModelError>
    for SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn from(err: ModelError) -> Self {
        Self {
            current_result: Err(err),
        }
    }
}

impl<ScalarType, NData> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    //todo document
    fn new<F>(
        model_result: Result<SeparableModel<ScalarType, NData>, ModelError>,
        function_parameters: &[String],
        function: F,
    ) -> Self
    where
        F: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
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
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
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
    pub fn invariant_function<F>(self, function: F) -> SeparableModelBuilder<ScalarType, NData>
    where
        F: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
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
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        F: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
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
    pub fn build(self) -> Result<SeparableModel<ScalarType, NData>, ModelError> {
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
