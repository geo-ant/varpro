pub mod proxies;

use crate::model::basefunction::Basefunction;
use crate::model::builder::proxies::FunctionBuilderProxy;
use crate::model::errors::{ModelBuilderError, ModelfunctionError};
use crate::model::{OwnedVector, SeparableModel};
use nalgebra::{Dim, Dynamic, Scalar};

//TODO MAYBE RETHINK THIS AND MAYBE GO BACK TO PULLING THE
// GO BACK FROM TRAITS AND HAVE TWO STRUCTS THAT SeparableModelBuilder and FunctionBuilderProxy which both have internal
// implementations to push independent functions. This should rely on an internal method of the model (just refactor and reuse
// the one I have, to not require push anymore). The Proxy that deals with derivatives exposes the with_derivatives method also and
// it has to check in all methods it exposes (also the one for independent functions) that the invariants of the model are respected,
// i.e. that the complete set of derivatives has been provided before building or pushing another function.
// BOTH structures carry the model or error internally !!!!
//todo implement the trait on SeparableModelBuilder, Result<SeparableModelBuilder,...>, and Result<FunctionBuilderProxy,...>
// pub trait GeneralModelBuilder<ScalarType, NData> : Sized
//     where ScalarType: Scalar,
//           NData: Dim,
//           nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
// {
//     fn current_model(self) -> Result<SeparableModel<ScalarType, NData>, ModelBuilderError>;
//
//     //todo: this thing can just be absorbed into the model
//     fn push_invariant_function<F>(mut self, function : F) -> Result<SeparableModelBuilder<ScalarType, NData>, ModelBuilderError>
//         where F: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static {
//         let mut model_result = self.current_model();
//         match model_result.as_mut() {
//             Ok(model) => {
//                 model.modelfunctions.push(Basefunction::parameter_independent(move |x, _model_params| (function)(x)));
//             }
//             Err(err) => {}
//         }
//         model_result.map(|model| SeparableModelBuilder::from(model))
//     }
//
//     fn push_function<F,StrType>(mut self, function_params: Vec<StrType> , function : F) -> Result<FunctionBuilderProxy<ScalarType, NData>, ModelBuilderError>
//         where F: Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
//         StrType : Into<String>
//     {
//         //todo make default implementation
//         unimplemented!()
//     }
// }
//
// //todo implement this trait for Result<FunctionBuilderProxy, Error>
// pub trait DerivativeModelBuilder<ScalarType, NData>: GeneralModelBuilder<ScalarType, NData>
//     where ScalarType: Scalar,
//           NData: Dim,
//           nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
// {
//     fn and_partial_deriv<F,StrType>(derivative_param: StrType , function : F) -> Result<FunctionBuilderProxy<ScalarType, NData>, ModelBuilderError>
//         where F: Fn(&OwnedVector<ScalarType, NData>, &OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
//               StrType : Into<String>;
// }
//
// impl<ScalarType,NData> GeneralModelBuilder<ScalarType, NData> for Result<SeparableModelBuilder<ScalarType,NData>, ModelBuilderError>
//     where ScalarType: Scalar,
//           NData: Dim,
//           nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
// {
//     fn current_model(self) -> Result<SeparableModel<ScalarType, NData>, ModelBuilderError> {
//         unimplemented!()
//     }
// }
// //todo document
pub struct SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    pub(self) model_result: Result<SeparableModel<ScalarType, NData>, ModelBuilderError>,
}

impl<ScalarType, NData> From<SeparableModel<ScalarType, NData>>
    for SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    fn from(_: SeparableModel<ScalarType, NData>) -> Self {
        unimplemented!()
    }
}

impl<ScalarType, NData> SeparableModelBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    //todo document
    pub fn with_parameters<StrType>(parameter_names: Vec<StrType>) -> Self
    where
        StrType: Into<String>,
    {
        let model_result =
            SeparableModel::new(parameter_names).map_err(|err| ModelBuilderError::from(err));
        Self { model_result }
    }

    //todo document
    pub fn push_invariant_function<F>(mut self, function: F) -> Self
    where
        F: Fn(&OwnedVector<ScalarType, NData>) -> OwnedVector<ScalarType, NData> + 'static,
    {
        match self.model_result.as_mut() {
            Ok(model) => {
                model
                    .modelfunctions
                    .push(Basefunction::parameter_independent(
                        move |x, _model_params| (function)(x),
                    ));
            }
            Err(err) => {}
        }
        self
    }

    //todo document
    pub fn push_function<F, StrType>(
        mut self,
        function_params: Vec<StrType>,
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
    where
        F: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, NData>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
        StrType: Into<String>,
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
    pub fn build(self) -> Result<SeparableModel<ScalarType, NData>, ModelBuilderError> {
        self.model_result
    }
}

pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    pub(self) model_result: Result<SeparableModel<ScalarType, NData>, ModelBuilderError>,
}
