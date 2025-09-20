use crate::{
    model::SeparableNonlinearModel,
    problem::{RhsType, SeparableProblem},
};
use nalgebra::{ComplexField, DMatrix, Dyn, Scalar, SVD};
mod svd;

// TODO only on lapack feature
mod colpiv_qr;

use colpiv_qr::ColPivQrLinearSolver;

#[allow(type_alias_bounds)]
/// levmar problem where the linear part is solved via column pivoted QR
/// decomposition.
//TODO expose only on lapack feature
pub type LevMarProblemCpQr<Model: SeparableNonlinearModel, Rhs> =
    LevMarProblem<Model, Rhs, ColPivQrLinearSolver<Model::ScalarType>>;

#[derive(Debug)]
pub struct LevMarProblem<Model, Rhs, Solver>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    Solver: LinearSolver<ScalarType = Model::ScalarType>,
    Rhs: RhsType,
    Model: SeparableNonlinearModel,
{
    pub(crate) separable_problem: SeparableProblem<Model, Rhs>,
    pub(crate) cached: Option<Solver>,
}

impl<Model, Rhs, Solver> From<SeparableProblem<Model, Rhs>> for LevMarProblem<Model, Rhs, Solver>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    Solver: LinearSolver<ScalarType = Model::ScalarType>,
    Rhs: RhsType,
    Model: SeparableNonlinearModel,
{
    fn from(problem: SeparableProblem<Model, Rhs>) -> Self {
        Self {
            separable_problem: problem,
            cached: None,
        }
    }
}

/// helper trait to abstract over the linear solvers (as of now ColPivQr
/// and SVD) used in the LevMarProblem. We don't actually abstract much
/// here, other than giving a method for getting the linear coefficients,
/// becaue the actual implementation of the LeastSquaresProblem
/// trait is done for specializations on the solvers for concrete types.
pub trait LinearSolver {
    type ScalarType: Scalar;
    /// get the linear coefficients in matrix form. For single RHS
    /// this is a matrix with just one column.
    fn linear_coefficients_matrix(&self) -> DMatrix<Self::ScalarType>;
}
