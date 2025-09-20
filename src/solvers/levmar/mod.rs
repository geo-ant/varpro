use crate::fit::FitResult;
use crate::prelude::*;
use crate::problem::{RhsType, SeparableProblem, SingleRhs};
use crate::statistics::FitStatistics;
use crate::util::to_vector;
use levenberg_marquardt::LeastSquaresProblem;
/// Type alias for the solver of the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate.
/// This provides the Levenberg-Marquardt nonlinear least squares optimization algorithm.
// pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use levenberg_marquardt::LevenbergMarquardt;
use levmar_problem::{LevMarProblem, LevMarProblemCpQr, LevMarProblemSvd, LinearSolver};
use nalgebra::{ComplexField, DMatrix, Dyn, Matrix, RawStorageMut, RealField, Scalar, U1};

use nalgebra_lapack::colpiv_qr::{ColPivQrReal, ColPivQrScalar};
use num_traits::float::TotalOrder;
use num_traits::{Float, FromPrimitive};

#[cfg(any(test, doctest))]
mod test;

// I got 99 problems but a module ain't one...
// Maybe we'll make this module public, but for now I feel this would make
// the API more complicated.
mod levmar_problem;

/// A thin wrapper around the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// solver from the `levenberg_marquardt` crate. The core benefit of this
/// wrapper is that we can also use it to calculate statistics.
#[derive(Debug)]
pub struct LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    solver: LevenbergMarquardt<Model::ScalarType>,
}

impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    /// construct a new solver using the given underlying solver. This allows
    /// us to configure the underlying solver with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self { solver }
    }

    #[allow(clippy::result_large_err)]
    fn solve<Rhs: RhsType, Solver: LinearSolver<ScalarType = Model::ScalarType>>(
        &self,
        problem: LevMarProblem<Model, Rhs, Solver>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
        LevMarProblem<Model, Rhs, Solver>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
    {
        let (problem, report) = self.solver.minimize(problem);

        let LevMarProblem {
            separable_problem,
            cached,
        } = problem;

        let linear_coefficients = cached.map(|cached| cached.linear_coefficients_matrix());

        let result = FitResult::new(separable_problem, linear_coefficients, report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn solve_with_cpqr<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: ColPivQrReal
            + ColPivQrScalar
            + Scalar
            + ComplexField
            + RealField
            + Float
            + FromPrimitive
            + TotalOrder,
    {
        let levmar_problem = LevMarProblemCpQr::from(problem);
        self.solve(levmar_problem)
    }

    #[allow(clippy::result_large_err)]
    pub fn solve_with_svd<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        let levmar_problem = LevMarProblemSvd::from(problem);
        self.solve(levmar_problem)
    }

    /// Try to solve the given varpro minimization problem. The parameters of
    /// the problem which are set when this function is called are used as the initial guess
    ///
    /// # Returns
    ///
    /// On success, returns an Ok value containing the fit result, which contains
    /// the final state of the problem as well as some convenience functions that
    /// allow to query the optimal parameters. Note that success of failure is
    /// determined from the minimization report. A successful result might still
    /// correspond to a failed minimization in some cases.
    /// On failure (when the minimization was not deemeed successful), returns
    /// an error with the same information as in the success case.
    #[allow(clippy::result_large_err)]
    #[deprecated(since = "0.14.0", note = "use the solve method instead")]
    pub fn fit<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
        Model::ScalarType: ColPivQrReal + ColPivQrScalar + Float + RealField + TotalOrder,
    {
        todo!()
    }
}

// TODO SVD
// impl<Model> LevMarSolver<Model>
// where
//     Model: SeparableNonlinearModel,
// {
//     /// creata a new solver using the given underlying solver. This allows
//     /// us to configure the underlying solver with non-default parameters
//     pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
//         Self { solver }
//     }

//     /// Try to solve the given varpro minimization problem. The parameters of
//     /// the problem which are set when this function is called are used as the initial guess
//     ///
//     /// # Returns
//     ///
//     /// On success, returns an Ok value containing the fit result, which contains
//     /// the final state of the problem as well as some convenience functions that
//     /// allow to query the optimal parameters. Note that success of failure is
//     /// determined from the minimization report. A successful result might still
//     /// correspond to a failed minimization in some cases.
//     /// On failure (when the minimization was not deemeed successful), returns
//     /// an error with the same information as in the success case.
//     #[allow(clippy::result_large_err)]
//     pub fn fit<Rhs: RhsType>(
//         &self,
//         problem: SeparableProblem<Model, Rhs>,
//     ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
//     where
//         Model: SeparableNonlinearModel,
//         Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
//     {
//         #[allow(deprecated)]
//         let (problem, report) = self.solver.minimize(problem);
//         let result = FitResult::new(problem, report);
//         if result.was_successful() {
//             Ok(result)
//         } else {
//             Err(result)
//         }
//     }
// }
impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    /// Same as the [`LevMarSolver::fit`](LevMarSolver::fit) function but also calculates some additional
    /// statistical information about the fit, if the fit was successful.
    ///
    /// # Returns
    ///
    /// See also the [`LevMarSolver::fit`](LevMarSolver::fit) function, but on success also returns statistical
    /// information about the fit in the form of a [`FitStatistics`] object.
    ///
    /// ## Problems with Multiple Right Hand Sides
    ///
    /// **Note** For now, fitting with statistics is only supported for problems
    /// with a single right hand side. If this function is invoked on a problem
    /// with multiple right hand sides, an error is returned.
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    pub fn fit_with_statistics(
        &self,
        problem: SeparableProblem<Model, SingleRhs>,
    ) -> Result<(FitResult<Model, SingleRhs>, FitStatistics<Model>), FitResult<Model, SingleRhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float,
        // TODO QR
        Model::ScalarType: ColPivQrReal + ColPivQrScalar + Float + RealField + TotalOrder,
    {
        let FitResult {
            problem,
            minimization_report,
            linear_coefficients,
        } = self.fit(problem)?;
        if !minimization_report.termination.was_successful() {
            return Err(FitResult::new(
                problem,
                linear_coefficients,
                minimization_report,
            ));
        }

        let Some(coefficients) = linear_coefficients else {
            return Err(FitResult::new(
                problem,
                linear_coefficients,
                minimization_report,
            ));
        };

        match FitStatistics::try_calculate(
            problem.model(),
            problem.weighted_data(),
            problem.weights(),
            coefficients.as_view(),
        ) {
            Ok(statistics) => Ok((
                FitResult::new(problem, Some(coefficients), minimization_report),
                statistics,
            )),
            Err(_) => Err(FitResult::new(
                problem,
                Some(coefficients),
                minimization_report,
            )),
        }
    }
}

impl<Model> Default for LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Float,
{
    fn default() -> Self {
        Self::with_solver(Default::default())
    }
}

/// copy the given matrix to a matrix column by stacking its column on top of each other
fn copy_matrix_to_column<T: Scalar + std::fmt::Display + Clone, S: RawStorageMut<T, Dyn>>(
    source: DMatrix<T>,
    target: &mut Matrix<T, Dyn, U1, S>,
) {
    //@todo make this more efficient...
    //@todo inefficient
    target.copy_from(&to_vector(source));
}
