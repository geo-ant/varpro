use nalgebra::{Scalar, Dim};
use crate::model::SeparableModel;
use crate::model::errors::{ModelfunctionError, ModelBuilderError};


pub struct FunctionBuilderProxy<ScalarType, NData>
    where ScalarType: Scalar,
          NData: Dim,
          nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>  //see https://github.com/dimforge/nalgebra/issues/580
{
    model_result: Result<SeparableModel<ScalarType, NData>, ModelBuilderError>,
}
