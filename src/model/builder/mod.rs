use std::hash::Hash;

use nalgebra::{Dim, Scalar};

use crate::model::builder::modelfunction_builder::ModelFunctionBuilder;
use crate::model::detail::check_parameter_names;
use crate::model::errors::Error;
use crate::model::modelfunction::ModelFunction;
use crate::model::{OwnedVector, SeparableModel};

mod modelfunction_builder;
#[cfg(test)]
mod test;

/// Helper trait that provides common **unchecked** implementations to push
/// functions and derivatives to a Result<SeparableModel, ModelBuilderError> that is internally carried.
/// It mimics inheritance by providing access to the underlying result types with the two methods
/// `current_model_result` and `current_model_result_mut`. The other functions allow pushing
/// fnuctions and derivatives. The functions to push functions and derivatives take care of wrapping
/// the given functions with the correct parameters BUT they do not check if it is allowed at this
/// stage of the building process to perform the operation. The callers have to make sure it is
/// indeed valid and leaves the invariants of the model building process intact.
trait ModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>,
{
    /// expose the internal model result as mutable
    fn current_model_result_mut(
        &mut self,
    ) -> Result<&mut SeparableModel<ScalarType, NData>, &mut Error>;

    //TODO implement unchecked methods here
}

// //todo document
pub struct SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    pub(self) model_result: Result<SeparableModel<ScalarType, NData>, Error>,
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
    {
        if let Err(parameter_error) = check_parameter_names(&parameter_names) {
            Self {
                model_result: Err(parameter_error),
            }
        } else {
            let parameter_names = parameter_names
                .into_iter()
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
    pub fn push_function<F, StrType>(
        self,
        function_params: &[StrType],
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        F: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, NData>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
        StrType: Into<String> + Clone,
    {
        // use some common function here that can also be used from the push function in the model builder with derivatives
        // make some common function that wraps the modelfunction into a lambda that distributes the model params to the funciton
        // params. This can then be used for the other functions.
        //
        // IDEA: when pushing derivatives(in the other builder) we can always track if the last() element of the vector
        // has the derivatives it needs and such
        unimplemented!()
    }

    //todo document
    pub fn build(self) -> Result<SeparableModel<ScalarType, NData>, Error> {
        self.model_result
    }
}

pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    model_result: Result<SeparableModel<ScalarType, NData>, Error>,
    current_function_builder: Option<ModelFunctionBuilder<ScalarType, NData>>,
}

impl<ScalarType, NData> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        ScalarType: Scalar,
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    pub fn new<F>(previous : SeparableModelBuilder<ScalarType,NData>, function : F) -> Self
    where F: Fn(&OwnedVector<ScalarType, NData>,
    &OwnedVector<ScalarType, NData>,
    ) -> OwnedVector<ScalarType, NData>
    + 'static,
    {
        todo!()
    }
}