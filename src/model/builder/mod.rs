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
    pub(self) model_result: Result<SeparableModel<ScalarType, NData>, ModelError>,
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

impl<ScalarType, NData> From<ModelError>
for SeparableModelBuilder<ScalarType, NData>
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
    pub fn with_parameters<StrType>(parameter_names: &[StrType]) -> Self
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
    pub fn push_invariant_function<F>(mut self, function: F) -> Self
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
    pub fn push_function<F>(
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
        //TODO: add a check for completeness function that checks
        // 1. that the model has at least one modelfunction
        // 2. each modelparameter has at least one modelfunction that depends on it
        todo!()
        //self.model_result
    }
}
// helper struct that contains a seperable model as well as a model function builder
// used inside the SeparableModelBuilderProxyWithDerivatives.
struct ModelAndFunctionBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    model: SeparableModel<ScalarType, NData>,
    builder: ModelFunctionBuilder<ScalarType, NData>,
}

impl<ScalarType, NData> ModelAndFunctionBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    pub fn new(
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
    current_result: Result<ModelAndFunctionBuilder<ScalarType, NData>, ModelError>,
}

impl<ScalarType,NData> From<ModelError> for SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        ScalarType: Scalar,
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn from(err: ModelError) -> Self {
        Self {
            current_result: Err(err)
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
    pub(self) fn new<F>(
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
                    current_result: Ok(ModelAndFunctionBuilder::new(
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
                current_result: Ok(ModelAndFunctionBuilder {
                    model: result.model,
                    builder: result.builder.partial_deriv(parameter.as_ref(), derivative),
                }),
            },
            Err(err) => Self::from(err),
        }
    }

    //todo document
    pub fn push_invariant_function<F>(self, function: F) -> SeparableModelBuilder<ScalarType,NData>
        where
            F: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
    {
        match self.current_result {
            Ok(result) => {
                let ModelAndFunctionBuilder {
                    mut model  ,
                    builder,
                } = result;
                if let Err(err) =  builder.build().map(|func|{model.modelfunctions.push(func)}) {
                    SeparableModelBuilder::from(err)
                } else {
                    SeparableModelBuilder::from(model).push_invariant_function(function)
                }
            }
            Err(err) => {
                SeparableModelBuilder::from(err)
            }
        }
    }

    //todo document
    pub fn push_function<F>(self, function_params : &[String], function: F) -> SeparableModelBuilderProxyWithDerivatives<ScalarType,NData>
        where
            F: Fn(&OwnedVector<ScalarType, NData>,&OwnedVector<ScalarType,Dynamic>) -> OwnedVector<ScalarType, NData> + 'static,
    {
        match self.current_result {
            Ok(result) => {
                let ModelAndFunctionBuilder {
                    mut model  ,
                    builder,
                } = result;
                if let Err(err) =  builder.build().map(|func|{model.modelfunctions.push(func)}) {
                    SeparableModelBuilderProxyWithDerivatives::from(err)
                } else {
                    SeparableModelBuilderProxyWithDerivatives::new(Ok(model), function_params, function)
                }
            }
            Err(err) => {
                SeparableModelBuilderProxyWithDerivatives::from(err)
            }
        }
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType, NData>, ModelError> {
        // TODO implement this by converting this into a separable model and then using the
        // build method on it to avoid code duplication when checking for completeness

        todo!()
    }

}
