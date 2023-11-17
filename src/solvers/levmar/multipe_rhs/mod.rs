//! This module contains vapro solvers for global fitting with multiple right hand sides.

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, DMatrix, Dyn, OMatrix, Owned, Scalar, SVD};

use crate::{prelude::SeparableNonlinearModel, util::Weights};

struct CachedCalculationsMrhs<ScalarType>
where
    ScalarType: Scalar + nalgebra::ComplexField,
{
    current_residuals: OMatrix<ScalarType, Dyn, Dyn>,
    current_svd: SVD<ScalarType, Dyn, Dyn>,
    linear_coeff_matrix: OMatrix<ScalarType, Dyn, Dyn>,
}

#[allow(non_snake_case)]
pub struct LevMarProblemMrhs<Model>
where
    Model: SeparableNonlinearModel<OutputDim = Dyn, ParameterDim = Dyn, ModelDim = Dyn>,
    Model::ScalarType: Scalar + ComplexField,
{
    Y_w: OMatrix<Model::ScalarType, Dyn, Dyn>,
    model: Model,
    weights: Weights<Model::ScalarType, Dyn>,
}

impl<Model> LevMarProblemMrhs<Model>
where
    Model: SeparableNonlinearModel<OutputDim = Dyn, ParameterDim = Dyn, ModelDim = Dyn>,
    Model::ScalarType: Scalar + ComplexField,
{
    #[deprecated]
    pub fn new(
        model: Model,
        data: DMatrix<Model::ScalarType>,
        weights: Weights<Model::ScalarType, Dyn>,
    ) -> Self {
        Self {
            Y_w: data,
            model,
            weights,
        }
    }
}

impl<Model> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn> for LevMarProblemMrhs<Model>
where
    Model: SeparableNonlinearModel<OutputDim = Dyn, ParameterDim = Dyn, ModelDim = Dyn>,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    type ResidualStorage = Owned<Model::ScalarType, Dyn>;

    type JacobianStorage = Owned<Model::ScalarType, Dyn, Dyn>;

    type ParameterStorage = Owned<Model::ScalarType, Dyn>;

    fn set_params(&mut self, x: &nalgebra::Vector<Model::ScalarType, Dyn, Self::ParameterStorage>) {
        todo!()
    }

    fn params(&self) -> nalgebra::Vector<Model::ScalarType, Dyn, Self::ParameterStorage> {
        todo!()
    }

    fn residuals(&self) -> Option<nalgebra::Vector<Model::ScalarType, Dyn, Self::ResidualStorage>> {
        todo!()
    }

    fn jacobian(
        &self,
    ) -> Option<nalgebra::Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
        todo!()
    }
}
