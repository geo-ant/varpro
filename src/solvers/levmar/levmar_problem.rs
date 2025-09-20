use nalgebra::{ComplexField, DMatrix, Dyn, Scalar};

use crate::{
    model::SeparableNonlinearModel,
    problem::{RhsType, SeparableProblem},
};

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

pub trait LinearSolver {
    type ScalarType: Scalar;
    fn linear_coefficients_matrix(&self) -> DMatrix<Self::ScalarType>;
}

pub struct ColPivQrLinearSolver<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    pub(crate) decomposition: nalgebra_lapack::ColPivQR<ScalarType, Dyn, Dyn>,
    /// the linear coefficients `$\boldsymbol C$` providing the current best fit
    pub(crate) linear_coefficients: DMatrix<ScalarType>,
}

impl<ScalarType> LinearSolver for ColPivQrLinearSolver<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    type ScalarType = ScalarType;

    fn linear_coefficients_matrix(&self) -> DMatrix<Self::ScalarType> {
        self.linear_coefficients.clone()
    }
}
