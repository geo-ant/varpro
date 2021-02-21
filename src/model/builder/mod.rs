use nalgebra::{DVector, Scalar};

use crate::basis_function::BasisFunction;
use crate::model::builder::modelfunction_builder::ModelBasisFunctionBuilder;
use crate::model::detail::check_parameter_names;
use crate::model::model_basis_function::ModelBasisFunction;
use crate::model::SeparableModel;
use error::ModelBuildError;

mod modelfunction_builder;

#[cfg(test)]
mod test;

pub mod error;

///! This is a builder for a [SeparableModel].
pub struct SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    model_result: Result<SeparableModel<ScalarType>, ModelBuildError>,
}

/// create a SeparableModelBuilder which contains an error variant
impl<ScalarType> From<ModelBuildError> for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelBuildError) -> Self {
        Self {
            model_result: Err(err),
        }
    }
}

/// create a SeparableModelBuilder with the given result variant
impl<ScalarType> From<Result<SeparableModel<ScalarType>, ModelBuildError>>
    for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(model_result: Result<SeparableModel<ScalarType>, ModelBuildError>) -> Self {
        Self { model_result }
    }
}

impl<ScalarType> SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    //todo document
    pub fn new<StrCollection>(parameter_names: StrCollection) -> Self
    where
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        let parameter_names: Vec<String> = parameter_names
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        if let Err(parameter_error) = check_parameter_names(&parameter_names) {
            Self {
                model_result: Err(parameter_error),
            }
        } else {
            let model_result = Ok(SeparableModel {
                parameter_names,
                basefunctions: Vec::default(),
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
                .basefunctions
                .push(ModelBasisFunction::parameter_independent(function));
        }
        self
    }

    //todo document
    pub fn function<F, StrCollection, ArgList>(
        self,
        function_params: StrCollection,
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        SeparableModelBuilderProxyWithDerivatives::new(self.model_result, function_params, function)
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelBuildError> {
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
) -> Result<SeparableModel<ScalarType>, ModelBuildError>
where
    ScalarType: Scalar,
{
    if model.basefunctions.is_empty() {
        Err(ModelBuildError::EmptyModel)
    } else {
        // now check that all model parameters are referenced in at least one parameter of one
        // of the given model functions. We do this by checking the indices of the derivatives
        // because each model parameter must occur at least once as a key in at least one of the
        // modelfunctions derivatives
        for (param_index, parameter_name) in model.parameter_names.iter().enumerate() {
            if !model
                .basefunctions
                .iter()
                .any(|function| function.derivatives.contains_key(&param_index))
            {
                return Err(ModelBuildError::UnusedParameter {
                    parameter: parameter_name.clone(),
                });
            }
        }
        // otherwise return this model
        Ok(model)
    }
}

/// helper struct that contains a seperable model as well as a model function builder
/// used inside the SeparableModelBuilderProxyWithDerivatives.
struct ModelAndModelBasisFunctionBuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    model: SeparableModel<ScalarType>,
    builder: ModelBasisFunctionBuilder<ScalarType>,
}

impl<ScalarType> ModelAndModelBasisFunctionBuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    fn new(
        model: SeparableModel<ScalarType>,
        builder: ModelBasisFunctionBuilder<ScalarType>,
    ) -> Self {
        Self { model, builder }
    }
}

pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    current_result: Result<ModelAndModelBasisFunctionBuilderPair<ScalarType>, ModelBuildError>,
}

impl<ScalarType> From<ModelBuildError> for SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelBuildError) -> Self {
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
    fn new<F, StrCollection, ArgList>(
        model_result: Result<SeparableModel<ScalarType>, ModelBuildError>,
        function_parameters: StrCollection,
        function: F,
    ) -> Self
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        match model_result {
            Ok(model) => {
                let model_parameters = model.parameters().clone();
                Self {
                    current_result: Ok(ModelAndModelBasisFunctionBuilderPair::new(
                        model,
                        ModelBasisFunctionBuilder::new(
                            model_parameters,
                            function_parameters,
                            function,
                        ),
                    )),
                }
            }
            Err(err) => Self {
                current_result: Err(err),
            },
        }
    }

    //todo document
    //BIG TODO: document that the argument list of the partial derivative must be the same
    //as the argument list of the parent function. This is a limitation of the way I pass functions
    //because I assume the same argument list. But it actually makes sense because paramters would
    //only vanish in the derivatives when the are linear, which I do not want to encourage anyways
    //TODO ALSO: document that partial derivative must take the arguments in the same order as
    //the base function for which they are the derivative
    pub fn partial_deriv<StrType: AsRef<str>, F, ArgList>(
        self,
        parameter: StrType,
        derivative: F,
    ) -> Self
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
    {
        match self.current_result {
            Ok(result) => Self {
                current_result: Ok(ModelAndModelBasisFunctionBuilderPair {
                    model: result.model,
                    builder: result.builder.partial_deriv(parameter.as_ref(), derivative),
                }),
            },
            Err(err) => Self::from(err),
        }
    }

    // since this proxy variant was started by adding a function to the [SeparableModelBuilder]
    // we know here that we must first finalize the function in the builder, add it to the model
    // (if the model is ok) and then add the invariant function that is pushed here. We add the
    // invariant function by bassing the model to a
    pub fn invariant_function<F>(self, function: F) -> SeparableModelBuilder<ScalarType>
    where
        F: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        match self.current_result {
            Ok(ModelAndModelBasisFunctionBuilderPair { mut model, builder }) => {
                let intermediate_result = builder.build().map(|func| {
                    model.basefunctions.push(func);
                    model
                });
                SeparableModelBuilder::from(intermediate_result).invariant_function(function)
            }
            Err(err) => SeparableModelBuilder::from(err),
        }
    }

    //todo document
    pub fn function<F, StrCollection, ArgList>(
        self,
        function_params: StrCollection,
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        match self.current_result {
            Ok(result) => {
                let ModelAndModelBasisFunctionBuilderPair { mut model, builder } = result;
                if let Err(err) = builder.build().map(|func| model.basefunctions.push(func)) {
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
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelBuildError> {
        // this method converts the internal results into a separable model and uses its
        // facilities to check for completion and the like
        match self.current_result {
            Ok(result) => {
                let ModelAndModelBasisFunctionBuilderPair { mut model, builder } = result;
                let intermediate_result = builder.build().map(|func| {
                    model.basefunctions.push(func);
                    model
                });
                SeparableModelBuilder::from(intermediate_result).build()
            }
            Err(err) => SeparableModelBuilder::from(err).build(),
        }
    }
}
