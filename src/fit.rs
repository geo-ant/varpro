use crate::{
    model::SeparableNonlinearModel,
    problem::{MultiRhs, RhsType, SeparableProblem, SingleRhs},
};
use levenberg_marquardt::MinimizationReport;
use nalgebra::{
    DMatrix, DMatrixView, DVector, DVectorView, Dyn, MatrixView, OMatrix, OVector, RealField,
    Scalar, VectorView,
};
use num_traits::Float;

/// A helper type that contains the fitting problem after the
/// minimization, as well as a report and some convenience functions.
///
/// This structure is returned by the [`LevMarSolver::fit`](crate::solvers::levmar::LevMarSolver::fit)
/// and [`LevMarSolver::fit_with_statistics`](crate::solvers::levmar::LevMarSolver::fit_with_statistics) methods.
#[derive(Debug)]
pub struct FitResult<Model, Rhs: RhsType>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
{
    /// The final state of the fitting problem after the
    /// minimization finished (regardless of whether fitting was successful or not).
    pub problem: SeparableProblem<Model, Rhs>,

    /// the linear coefficient matrix (in case of single rhs: matrix has
    /// only one column) at the solution.
    pub(crate) linear_coefficients: Option<DMatrix<Model::ScalarType>>,

    /// The minimization report of the underlying solver.
    /// It contains information about the minimization process
    /// and should be queried to see whether the minimization
    /// was considered successful.
    pub minimization_report: MinimizationReport<Model::ScalarType>,
}

impl<Model> FitResult<Model, MultiRhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
{
    /// **Note** This implementation is for fitting problems with multiple right hand sides.
    ///
    /// Convenience function to get the linear coefficients after the fit has
    /// finished. Will return None if there was an error during fitting.
    ///
    /// The coefficients vectors for the individual
    /// members of the datasets are the colums of the returned matrix. That means
    /// one coefficient vector for each right hand side.
    pub fn linear_coefficients(&self) -> Option<DMatrixView<Model::ScalarType>> {
        self.linear_coefficients.as_ref().map(|c| c.as_view())
    }

    /// **Note** This implementation is for fitting problems with multiple right hand sides.
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for multiple right hand sides, the output is a matrix
    /// where each column corresponds to one right hand side.
    pub fn best_fit(&self) -> Option<OMatrix<Model::ScalarType, Dyn, Dyn>> {
        let coeff = self.linear_coefficients()?;
        let eval = self.problem.model().eval().ok()?;
        Some(eval * coeff)
    }
}

impl<Model> FitResult<Model, SingleRhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
{
    /// **Note** This implementation is for fitting problems with a single right hand side.
    ///
    /// Convenience function to get the linear coefficients after the fit has
    /// finished. Will return None if there was an error during fitting.
    /// The coefficients are given as a single vector.
    pub fn linear_coefficients(&self) -> Option<DVectorView<Model::ScalarType>> {
        let coeff = self.linear_coefficients.as_ref()?;
        debug_assert_eq!(
            coeff.ncols(),
            1,
            "Coefficient matrix must have exactly one colum for problem with single right hand side. This indicates a programming error inside this library!"
        );
        Some(coeff.column(0))
    }
    /// **Note** This implementation is for fitting problems with multiple right hand sides
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for a single right hand side, the output is a vector.
    pub fn best_fit(&self) -> Option<OVector<Model::ScalarType, Dyn>> {
        let coeff = self.linear_coefficients()?;
        let eval = self.problem.model().eval().ok()?;
        Some(eval * coeff)
    }
}

impl<Model, Rhs: RhsType> FitResult<Model, Rhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
{
    /// internal helper for constructing an instance
    pub(crate) fn new(
        problem: SeparableProblem<Model, Rhs>,
        linear_coefficients: Option<DMatrix<Model::ScalarType>>,
        minimization_report: MinimizationReport<Model::ScalarType>,
    ) -> Self {
        Self {
            problem,
            minimization_report,
            linear_coefficients,
        }
    }

    /// convenience function to get the nonlinear parameters of the model after
    /// the fitting process has finished.
    pub fn nonlinear_parameters(&self) -> OVector<Model::ScalarType, Dyn> {
        self.problem.model().params()
    }

    /// whether the fit was deemeed successful. The fit might still be not
    /// be optimal for numerical reasons, but the minimization process
    /// terminated successfully.
    pub fn was_successful(&self) -> bool {
        self.minimization_report.termination.was_successful()
    }
}
