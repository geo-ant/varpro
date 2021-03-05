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

///! A builder that allows us to construct a valid [SeparableModel](crate::model::SeparableModel).
///
/// The builder allows us to provide basis functions for a separable model as a step by step process.
/// TODO DOCUMENT
//     //     //FOR PARTIAL DERIVS: mention that the partial deriv function must have the same number
//              of args as the original function. (mention: this should be the case anyways because
//              otherwise it indicates that the parameter was linear in the deriv
//     //     //as the argument list of the parent function. This is a limitation of the way I pass functions
//     //     //because I assume the same argument list. But it actually makes sense because paramters would
//     //     //only vanish in the derivatives when the are linear, which I do not want to encourage anyways
//     //     //!!!ALSO: document that partial derivative must take the arguments in the same order as!!!!
//     //     //the base function for which they are the derivative
#[must_use="The builder should be transformed into a model using the build() method"]
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
    /// Create a new builder for a model that depends on this list of paramters
    /// The model parameters indices correspond to the order of appearance in here.
    /// Model parameter indices start at 0.
    /// # Arguments
    /// * `parameter_names` A collection containing all the nonlinear model parameters
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

    /// Add a function `$\vec{f}(\vec{x})$` to the model. In the `varpro` library this is called
    /// an *invariant* function because the model function is independent of the model parameters
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
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

    /// Add a function `$\vec{f}(\vec{x},\alpha_1,...,\alpha_n)$` to the model that depends on the
    /// location parameter `\vec{x}` and nonlinear model parameters.
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
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

    /// Build a separable model from the contents of this builder.
    /// # Result
    /// A valid separable model or an error indicating why a valid model could not be constructed.
    /// See also the documentation for [SeparableModelBuilder](self::SeparableModelBuilder)
    /// fo what constitutes a valid model.
    ///
    /// **Note** The order of basis functions in the model is order in which the basis functions
    /// where provided during the builder stage. That means the first basis functions gets index `0` in
    /// the model, the second gets index `1` and so on.
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

/// This is just a proxy that does need to be used directly. For constructing a model from
/// a builder see the documentation for [SeparableModelBuilder](self::SeparableModelBuilder).
/// **Sidenote** This structure will hopefully be made more elegant using some metaprogramming techniques
/// in the future. Right now this exists to make sure that partial derivatives cannot accidentally
/// be added to invariant functions. The compiler simply will forbid it. In future the library aims
/// to make more such checks at compile time and reduce the need for runtime errors caused by invalid model
/// construction.
#[must_use="This is meant as a transient expression proxy. Use build() to build a model."]
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
    /// Construct an instance. This is invoked when adding a function that depends on model
    /// parameters to the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// # Arguments
    /// * `model_result`: the current model or an error
    /// * `function_parameters`: the given list of nonlinear function parameters
    /// * `function`: the function
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
                let model_parameters = model.parameters().to_vec();
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

    /// Add a partial derivative to a function. This function is documented as part of the
    /// [SeparableModelBuilder](self::SeparableModelBuilder) documentation.
    /// *Info*: the reason it appears in this proxy only is to make sure that partial derivatives
    /// can only be added to functions that depend on model parameters.
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

    /// Add a function `\vec{f}(\vec{x})` that does not depend on the model parameters. This is documented
    /// as part of the [SeparableModelBuilder](self::SeparableModelBuilder) documentation.
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

    /// Add a function `\vec{f}(\vec{x},\alpha_j,...,\alpha_k)` that depends on a subset of the
    /// model parameters. This functionality is documented as part of the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// documentation.
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

    /// Finalized the building process and build a separable model.
    /// This functionality is documented as part of the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// documentation.
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
