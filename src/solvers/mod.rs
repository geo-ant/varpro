
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Scalar, Dynamic, ComplexField, Vector, Matrix, U3, Vector3, VecStorage, U1, DefaultAllocator, U2, Matrix2, Vector2, VectorN, DVector};
use nalgebra::storage::Owned;
use std::sync::Arc;
use nalgebra::allocator::Allocator;

pub struct LevMarSolver<ScalarType>
where ScalarType : Scalar{
    parameters : Vec<ScalarType>
}


// We implement a trait for every problem we want to solve
impl<ScalarType> LeastSquaresProblem<ScalarType, Dynamic, Dynamic> for LevMarSolver<ScalarType>
where ScalarType : Scalar+ComplexField{
    type ResidualStorage = Owned<ScalarType, Dynamic>;
    type JacobianStorage = Owned<ScalarType, Dynamic, Dynamic>;
    type ParameterStorage = Owned<ScalarType, Dynamic>;

    fn set_params(&mut self, params: &Vector<ScalarType, Dynamic, Self::ParameterStorage>) {
        self.parameters = params.iter().cloned().collect();
    }
    fn params(&self) -> Vector<ScalarType, Dynamic, Self::ParameterStorage> {
        DVector::from(self.parameters.clone())
    }
    fn residuals(&self) -> Option<Vector<ScalarType, Dynamic, Self::ResidualStorage>> {
        unimplemented!()
    }
    fn jacobian(&self) -> Option<Matrix<ScalarType, Dynamic, Dynamic, Self::JacobianStorage>> {
        unimplemented!()
    }
}
