use crate::statistics::FitStatistics;
use crate::{model, prelude::*};
use levenberg_marquardt::{LeastSquaresProblem, MinimizationReport};
use nalgebra::allocator::{Allocator, Reallocator};
use nalgebra::storage::Owned;
use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimAdd, DimMax, DimMaximum, DimMin, DimSub, Matrix,
    OVector, RawStorageMut, RealField, Scalar, Storage, UninitMatrix, Vector, SVD,
};

mod builder;
#[cfg(any(test, doctest))]
mod test;

use crate::util::Weights;
pub use builder::LevMarProblemBuilder;
/// type alias for the solver of the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate
// pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use levenberg_marquardt::LevenbergMarquardt;
use num_traits::{Float, FromPrimitive};
use std::ops::Mul;

/// A thin wrapper around the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// solver from the `levenberg_marquardt` crate. The core benefit of this
/// wrapper is that we can also use it to calculate statistics.
pub struct LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
{
    solver: LevenbergMarquardt<Model::ScalarType>,
}

/// A helper type that contains the fitting problem after the
/// minimization, as well as a report and some convenience functions
#[derive(Debug)]
pub struct FitResult<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::ParameterDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
    <Model as model::SeparableNonlinearModel>::OutputDim:
        nalgebra::DimMin<<Model as model::SeparableNonlinearModel>::ModelDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
{
    /// the final state of the fitting problem after the
    /// minimization finished (regardless of whether fitting was successful or not)
    pub problem: LevMarProblem<Model>,

    /// the minimization report of the underlying solver.
    /// It contains information about the minimization process
    /// and should be queried to see whether the minimization
    /// was considered successful
    pub minimization_report: MinimizationReport<Model::ScalarType>,
}

impl<Model> FitResult<Model>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::ParameterDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
    <Model as model::SeparableNonlinearModel>::OutputDim:
        nalgebra::DimMin<<Model as model::SeparableNonlinearModel>::ModelDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <<Model as model::SeparableNonlinearModel>::OutputDim as nalgebra::DimMin<
            <Model as model::SeparableNonlinearModel>::ModelDim,
        >>::Output,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::OutputDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as model::SeparableNonlinearModel>::ScalarType,
        <Model as model::SeparableNonlinearModel>::ModelDim,
    >,
{
    /// internal helper for constructing an instance
    fn new(
        problem: LevMarProblem<Model>,
        minimization_report: MinimizationReport<Model::ScalarType>,
    ) -> Self {
        Self {
            problem,
            minimization_report,
        }
    }

    /// convenience function to get the nonlinear parameters of the model after
    /// the fitting process has finished.
    pub fn nonlinear_parameters(&self) -> OVector<Model::ScalarType, Model::ParameterDim> {
        self.problem.model().params()
    }

    /// convenience function to get the linear coefficients after the fit has
    /// finished
    pub fn linear_coefficients(&self) -> Option<&OVector<Model::ScalarType, Model::ModelDim>> {
        self.problem.linear_coefficients()
    }

    /// whether the fit was deemeed successful. The fit might still be not
    /// be optimal for numerical reasons, but the minimization process
    /// terminated successfully.
    pub fn was_successful(&self) -> bool {
        self.minimization_report.termination.was_successful()
    }
}

impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
{
    /// create a new solver with default parameters. Uses the underlying
    /// solver of the `levenberg_marquardt` crate
    pub fn new() -> Self
    where
        Model::ScalarType: RealField + Float,
    {
        Self {
            solver: LevenbergMarquardt::new(),
        }
    }

    /// creata a new solver using the given underlying solver. This allows
    /// us to configure the underlying with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self { solver }
    }

    /// Solve the fitting problem. This method is deprecated, use the fit method instead.
    #[deprecated(since = "0.7.0", note = "use the fit(...) method instead")]
    pub fn minimize(&self,problem : LevMarProblem<Model>) -> (LevMarProblem<Model>,MinimizationReport<Model::ScalarType>)
    where Model:SeparableNonlinearModel,
        LevMarProblem<Model>:LeastSquaresProblem<Model::ScalarType,Model::OutputDim, Model::ParameterDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim> + Reallocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim,DimMaximum<Model::OutputDim,Model::ParameterDim>,Model::ParameterDim> + Allocator<usize,Model::ParameterDim>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float + FromPrimitive,
        <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField: Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
        Model: SeparableNonlinearModel,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
        <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: Storage<Model::ScalarType, Model::OutputDim>,
            <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: RawStorageMut<Model::ScalarType, Model::OutputDim>,
        Model::OutputDim: DimMin<Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField>::RealField, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMax<<Model as model::SeparableNonlinearModel>::ParameterDim>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMin<<Model as model::SeparableNonlinearModel>::ParameterDim>
    {
        let result = match self.fit(problem) {
            Ok(result) => result,
            Err(result) => result,
        };
        (result.problem, result.minimization_report)
    }

    /// Try to solve the given varpro minimization problem. The parameters of
    /// the problem which are set when this function is called are used as the initial guess
    /// # Returns
    /// On success, returns an Ok value containing the fit result, which contains
    /// the final state of the problem as well as some convenience functions that
    /// allow to query the optimal parameters. Note that success of failure is
    /// determined from the minimization report. A successful result might still
    /// correspond to a failed minimization in some cases.
    /// On failure (when the minimization was not deemeed successful), returns
    /// an error with the same information as in the success case.
    pub fn fit(&self,problem : LevMarProblem<Model>) -> Result<FitResult<Model>,FitResult<Model>>
    where Model:SeparableNonlinearModel,
        LevMarProblem<Model>:LeastSquaresProblem<Model::ScalarType,Model::OutputDim, Model::ParameterDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim> + Reallocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim,DimMaximum<Model::OutputDim,Model::ParameterDim>,Model::ParameterDim> + Allocator<usize,Model::ParameterDim>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float + FromPrimitive,
        <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField: Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
        Model: SeparableNonlinearModel,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
        <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: Storage<Model::ScalarType, Model::OutputDim>,
            <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: RawStorageMut<Model::ScalarType, Model::OutputDim>,
        Model::OutputDim: DimMin<Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField>::RealField, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMax<<Model as model::SeparableNonlinearModel>::ParameterDim>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMin<<Model as model::SeparableNonlinearModel>::ParameterDim>
    {
        let (problem, report) = self.solver.minimize(problem);
        let result = FitResult::new(problem, report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }

    pub fn fit_with_statistics(&self,problem : LevMarProblem<Model>) -> Result<(FitResult<Model>,FitStatistics<Model>),FitResult<Model>>
    where Model:SeparableNonlinearModel,
        LevMarProblem<Model>:LeastSquaresProblem<Model::ScalarType,Model::OutputDim, Model::ParameterDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim> + Reallocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim,DimMaximum<Model::OutputDim,Model::ParameterDim>,Model::ParameterDim> + Allocator<usize,Model::ParameterDim>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float,
        <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField: Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
        Model: SeparableNonlinearModel,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
        <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: Storage<Model::ScalarType, Model::OutputDim>,
            <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: RawStorageMut<Model::ScalarType, Model::OutputDim>,
        Model::OutputDim: DimMin<Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField>::RealField, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMax<<Model as model::SeparableNonlinearModel>::ParameterDim>,
        <Model as model::SeparableNonlinearModel>::OutputDim: DimMin<<Model as model::SeparableNonlinearModel>::ParameterDim>,
        <Model as model::SeparableNonlinearModel>::ModelDim: nalgebra::DimAdd<<Model as model::SeparableNonlinearModel>::ParameterDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model as model::SeparableNonlinearModel>::ScalarType, <<Model as model::SeparableNonlinearModel>::ModelDim as DimAdd<<Model as model::SeparableNonlinearModel>::ParameterDim>>::Output, <<Model as model::SeparableNonlinearModel>::ModelDim as DimAdd<<Model as model::SeparableNonlinearModel>::ParameterDim>>::Output>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model as model::SeparableNonlinearModel>::ScalarType, Model::OutputDim, <<Model as model::SeparableNonlinearModel>::ModelDim as DimAdd<<Model as model::SeparableNonlinearModel>::ParameterDim>>::Output>,
        DefaultAllocator: nalgebra::allocator::Allocator<<Model as model::SeparableNonlinearModel>::ScalarType,  <<Model as model::SeparableNonlinearModel>::ModelDim as DimAdd<<Model as model::SeparableNonlinearModel>::ParameterDim>>::Output,Model::OutputDim>,
    {
        let (problem, report) = self.solver.minimize(problem);
        if !report.termination.was_successful() {
            return Err(FitResult::new(problem, report));
        }

        if let Ok(statistics) = FitStatistics::try_calculate(
            problem.model(),
            problem.data(),
            problem.weights(),
            problem.linear_coefficients().unwrap(),
        ) {
            Ok((FitResult::new(problem, report), statistics))
        } else {
            Err(FitResult::new(problem, report))
        }
    }
}

impl<Model> Default for LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Float,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// helper structure that stores the cached calculations,
/// which are carried out by the LevMarProblem on setting the parameters
#[derive(Debug, Clone)]
struct CachedCalculations<ScalarType, ModelDim, OutputDim>
where
    ScalarType: Scalar + ComplexField,
    ModelDim: Dim,
    OutputDim: Dim + nalgebra::DimMin<ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<
        ScalarType,
        <OutputDim as DimMin<ModelDim>>::Output,
        ModelDim,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        ScalarType,
        OutputDim,
        <OutputDim as DimMin<ModelDim>>::Output,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        <ScalarType as ComplexField>::RealField,
        <OutputDim as DimMin<ModelDim>>::Output,
    >,
{
    /// The current residual of model function values belonging to the current parameters
    current_residuals: OVector<ScalarType, OutputDim>,
    /// Singular value decomposition of the current function value matrix
    current_svd: SVD<ScalarType, OutputDim, ModelDim>,
    /// the linear coefficients `$\vec{c}$` providing the current best fit
    linear_coefficients: OVector<ScalarType, ModelDim>,
}

/// This is a the problem of fitting the separable model to data in a form that the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate can use it to
/// perform the least squares fit.
/// # Construction
/// Use the [LevMarProblemBuilder](self::builder::LevMarProblemBuilder) to create an instance of a
/// levmar problem.
/// # Usage
/// After obtaining an instance of `LevMarProblem` we can pass it to the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// structure of the levenberg_marquardt crate for minimization. Refer to the documentation of the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) for an overview. A usage example
/// is provided in this crate documentation as well. The [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// solver is reexported by this module as [LevMarSolver](self::LevMarSolver) for naming consistency.
#[derive(Clone)]
pub struct LevMarProblem<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
    Model::OutputDim: DimMin<Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
        Model::ModelDim,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        Model::OutputDim,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        <Model::ScalarType as ComplexField>::RealField,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
{
    /// the *weighted* data vector to which to fit the model `$\vec{y}_w$`
    /// **Attention** the data vector is weighted with the weights if some weights
    /// where provided (otherwise it is unweighted)
    y_w: OVector<Model::ScalarType, Model::OutputDim>,
    /// a reference to the separable model we are trying to fit to the data
    model: Model,
    /// truncation epsilon for SVD below which all singular values are assumed zero
    svd_epsilon: <Model::ScalarType as ComplexField>::RealField,
    /// the weights of the data. If none are given, the data is not weighted
    /// If weights were provided, the builder has checked that the weights have the
    /// correct dimension for the data
    weights: Weights<Model::ScalarType, Model::OutputDim>,
    /// the currently cached calculations belonging to the currently set model parameters
    /// those are updated on set_params. If this is None, then it indicates some error that
    /// is propagated on to the levenberg-marquardt crate by also returning None results
    /// by residuals() and/or jacobian()
    cached: Option<CachedCalculations<Model::ScalarType, Model::ModelDim, Model::OutputDim>>,
}

impl<Model> std::fmt::Debug for LevMarProblem<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
    Model::OutputDim: DimMin<Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
        Model::ModelDim,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        Model::OutputDim,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        <Model::ScalarType as ComplexField>::RealField,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevMarProblem")
            .field("y_w", &self.y_w)
            .field("model", &"/* omitted */")
            .field("svd_epsilon", &self.svd_epsilon)
            .field("weights", &self.weights)
            .field("cached", &self.cached)
            .finish()
    }
}

impl<Model> LevMarProblem<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
    Model::OutputDim: DimMin<Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
        Model::ModelDim,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        Model::OutputDim,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
    DefaultAllocator: nalgebra::allocator::Allocator<
        <Model::ScalarType as ComplexField>::RealField,
        <Model::OutputDim as DimMin<Model::ModelDim>>::Output,
    >,
{
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    /// # Returns
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    pub fn linear_coefficients(&self) -> Option<&OVector<Model::ScalarType, Model::ModelDim>> {
        self.cached.as_ref().map(|cache| &cache.linear_coefficients)
    }

    /// access the contained model immutably
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// get the weights of the data for the fitting problem
    pub fn weights(&self) -> &Weights<Model::ScalarType, Model::OutputDim> {
        &self.weights
    }

    /// the data that we are trying to fit
    pub fn data(&self) -> &OVector<Model::ScalarType, Model::OutputDim> {
        &self.y_w
    }
}

impl<Model> LeastSquaresProblem<Model::ScalarType,Model::OutputDim, Model::ParameterDim>
    for LevMarProblem<Model>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField: Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim,Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
    <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: Storage<Model::ScalarType, Model::OutputDim>,
    <DefaultAllocator as nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>>::Buffer: RawStorageMut<Model::ScalarType, Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField> ::RealField, <<Model::OutputDim as DimMin<Model::ModelDim>>::Output as DimSub<Const<1>>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <<Model::OutputDim as DimMin<Model::ModelDim>>::Output as DimSub<Const<1>>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<(<Model::ScalarType as ComplexField>::RealField, usize), <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    <Model::OutputDim as DimMin<Model::ModelDim>>::Output: DimSub<nalgebra::dimension::Const<1>> ,
    Model::OutputDim: DimMin<Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField>::RealField, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,

{
    type ResidualStorage = Owned<Model::ScalarType, Model::OutputDim>;
    type JacobianStorage = Owned<Model::ScalarType, Model::OutputDim, Model::ParameterDim>;
    type ParameterStorage = Owned<Model::ScalarType, Model::ParameterDim>;

    #[allow(non_snake_case)]
    /// Set the (nonlinear) model parameters `$\vec{\alpha}$` and update the internal state of the
    /// problem accordingly. The parameters are expected in the same order that the parameter
    /// names were provided in at model creation. So if we gave `&["tau","beta"]` as parameters at
    /// model creation, the function expects the layout of the parameter vector to be `$\vec{\alpha}=(\tau,\beta)^T$`.
    fn set_params(&mut self, params: &Vector<Model::ScalarType, Model::ParameterDim, Self::ParameterStorage>) {
        if self.model.set_params(params.clone()).is_err() {
            self.cached = None;
        }
        // matrix of weighted model function values
        let Phi_w = self
            .model
            .eval()
            .ok()
            .map(|Phi| &self.weights * Phi);

        // calculate the svd
        let svd_epsilon = self.svd_epsilon;
        let current_svd = Phi_w.as_ref().map(|Phi_w| Phi_w.clone().svd(true, true));
        let linear_coefficients = current_svd
            .as_ref()
            .and_then(|svd| svd.solve(&self.y_w, svd_epsilon).ok());

        // calculate the residuals
        let current_residuals = Phi_w
            .zip(linear_coefficients.as_ref())
            .map(|(Phi_w, coeff)| &self.y_w - &Phi_w * coeff);

        // if everything was successful, update the cached calculations, otherwise set the cache to none
        if let (Some(current_residuals), Some(current_svd), Some(linear_coefficients)) =
            (current_residuals, current_svd, linear_coefficients)
        {
            self.cached = Some(CachedCalculations {
                current_residuals,
                current_svd,
                linear_coefficients,
            })
        } else {
            self.cached = None;
        }
    }

    /// Retrieve the (nonlinear) model parameters as a vector `$\vec{\alpha}$`.
    /// The order of the parameters in the vector is the same as the order of the parameter
    /// names given on model creation. E.g. if the parameters at model creation where given as
    /// `&["tau","beta"]`, then the returned vector is `$\vec{\alpha} = (\tau,\beta)^T$`, i.e.
    /// the value of parameter `$\tau$` is at index `0` and the value of `$\beta$` at index `1`.
    fn params(&self) -> Vector<Model::ScalarType, Model::ParameterDim, Self::ParameterStorage> {
        self.model.params()
    }

    /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
    /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
    /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
    /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
    /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
    /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn residuals(&self) -> Option<Vector<Model::ScalarType, Model::OutputDim, Self::ResidualStorage>> {
        self.cached
            .as_ref()
            .map(|cached| cached.current_residuals.clone())
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Model::OutputDim, Model::ParameterDim, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing
        if let Some(CachedCalculations {
            current_residuals: _,
            current_svd,
            linear_coefficients,
        }) = self.cached.as_ref()
        {

            // this is not a great pattern, but the trait bounds on copy_from
            // as of now prevent us from doing something more idiomatic
            let mut jacobian_matrix = unsafe {
                UninitMatrix::uninit(self.model.output_len(), self.model.parameter_count())
                    .assume_init()
            };

            let U = current_svd.u.as_ref()?; // will return None if this was not calculated
            let U_t = U.transpose();

            //let Sigma_inverse : DMatrix<Model::ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));
            //let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

            for (k, mut jacobian_col) in jacobian_matrix.column_iter_mut().enumerate() {
                // weighted derivative matrix
                let Dk = &self.weights
                    * self
                        .model
                        .eval_partial_deriv(k)
                        .ok()?; // will return none if this could not be calculated
                let Dk_c = Dk * linear_coefficients;
                let minus_ak = U * (&U_t * (&Dk_c)) - Dk_c;

                //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
                //let Dk_t_rw : DVector<Model::ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
                //let _minus_bk : DVector<Model::ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));
                jacobian_col.copy_from(&(minus_ak));
            }
            Some(jacobian_matrix)
        } else {
            None
        }
    }
}
