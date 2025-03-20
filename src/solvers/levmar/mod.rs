use crate::prelude::*;
use crate::statistics::FitStatistics;
use crate::util::{to_vector, Weights};
pub use builder::LevMarProblemBuilder;
/// type alias for the solver of the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate
// pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use levenberg_marquardt::LevenbergMarquardt;
use levenberg_marquardt::{LeastSquaresProblem, MinimizationReport};
use nalgebra::storage::Owned;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dim, DimMin, Dyn, Matrix, MatrixView, OMatrix,
    OVector, RawStorageMut, RealField, Scalar, UninitMatrix, Vector, VectorView, SVD, U1,
};
use num_traits::{Float, FromPrimitive};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::Mul;

mod builder;
#[cfg(any(test, doctest))]
mod test;

pub trait RhsType: std::fmt::Debug + Copy + Clone + PartialEq + Eq + Sync + Send {}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SingleRhs;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MultiRhs;

impl RhsType for SingleRhs {}
impl RhsType for MultiRhs {}

pub trait Parallelism: std::fmt::Debug + Copy + Clone + PartialEq + Eq + Sync + Send {}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Parallel {}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Sequential {}

impl Parallelism for Parallel {}
impl Parallelism for Sequential {}

/// contains levmar solvers based on QR decomposition
pub mod qr;

/// A thin wrapper around the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// solver from the `levenberg_marquardt` crate. The core benefit of this
/// wrapper is that we can also use it to calculate statistics.
pub struct LevMarSolver<Model, Rhs: RhsType>
where
    Model: SeparableNonlinearModel,
{
    solver: LevenbergMarquardt<Model::ScalarType>,
    phantom: PhantomData<Rhs>,
}

/// A helper type that contains the fitting problem after the
/// minimization, as well as a report and some convenience functions
#[derive(Debug)]
pub struct FitResult<Model, Rhs: RhsType>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Scalar + Float,
{
    /// the final state of the fitting problem after the
    /// minimization finished (regardless of whether fitting was successful or not)
    pub problem: LevMarProblem<Model, Rhs, Sequential>,

    /// the minimization report of the underlying solver.
    /// It contains information about the minimization process
    /// and should be queried to see whether the minimization
    /// was considered successful
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
    pub fn linear_coefficients(&self) -> Option<MatrixView<Model::ScalarType, Dyn, Dyn>> {
        Some(self.problem.cached.as_ref()?.linear_coefficients.as_view())
    }

    /// **Note** This implementation is for fitting problems with a single right hand side.
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for single right hand sides, the output is
    /// a column vector.
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
    pub fn linear_coefficients(&self) -> Option<VectorView<Model::ScalarType, Dyn>> {
        let coeff = &self.problem.cached.as_ref()?.linear_coefficients;
        debug_assert_eq!(coeff.ncols(),1,
            "Coefficient matrix must have exactly one colum for problem with single right hand side. This indicates a programming error inside this library!");
        Some(self.problem.cached.as_ref()?.linear_coefficients.column(0))
    }
    /// **Note** This implementation is for fitting problems with multiple right hand sides
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for single right hand sides, the output is
    /// a matrix containing composed of column vectors for the right hand sides.
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
    fn new(
        problem: LevMarProblem<Model, Rhs, Sequential>,
        minimization_report: MinimizationReport<Model::ScalarType>,
    ) -> Self {
        Self {
            problem,
            minimization_report,
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

impl<Model, Rhs: RhsType> LevMarSolver<Model, Rhs>
where
    Model: SeparableNonlinearModel,
{
    /// creata a new solver using the given underlying solver. This allows
    /// us to configure the underlying solver with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self {
            solver,
            phantom: PhantomData,
        }
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
    pub fn fit<Par: Parallelism>(
        &self,
        problem: LevMarProblem<Model, Rhs, Par>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        LevMarProblem<Model, Rhs, Par>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        #[allow(deprecated)]
        let (problem, report) = self.solver.minimize(problem);
        let result = FitResult::new(problem.into_sequential(), report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }
}

impl<Model> LevMarSolver<Model, SingleRhs>
where
    Model: SeparableNonlinearModel,
{
    /// Same as the [LevMarSolver::fit] function but also calculates some additional
    /// statistical information about the fit, if the fit was successful.
    ///
    /// # Returns
    ///
    /// See also the [LevMarSolver::fit] function, but on success also returns statistical
    /// information about the fit.
    ///
    /// ## Problems with Multiple Right Hand Sides
    ///
    /// **Note** For now, fitting with statistics is only supported for problems
    /// with a single right hand side. If this function is invoked on a problem
    /// with multiple right hand sides, an error is returned.
    #[allow(clippy::result_large_err)]
    pub fn fit_with_statistics<Par: Parallelism>(
        &self,
        problem: LevMarProblem<Model, SingleRhs, Par>,
    ) -> Result<(FitResult<Model, SingleRhs>, FitStatistics<Model>), FitResult<Model, SingleRhs>>
    where
        Model: SeparableNonlinearModel,
        LevMarProblem<Model, SingleRhs, Par>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
        Model::ScalarType: Scalar + ComplexField + RealField + Float,
    {
        let FitResult {
            problem,
            minimization_report,
        } = self.fit(problem)?;
        if !minimization_report.termination.was_successful() {
            return Err(FitResult::new(problem, minimization_report));
        }

        let Some(coefficients) = problem.linear_coefficients() else {
            return Err(FitResult::new(problem, minimization_report));
        };

        if let Ok(statistics) = FitStatistics::try_calculate(
            problem.model(),
            problem.weighted_data(),
            problem.weights(),
            coefficients,
        ) {
            Ok((FitResult::new(problem, minimization_report), statistics))
        } else {
            Err(FitResult::new(problem, minimization_report))
        }
    }
}

impl<Model, Rhs: RhsType> Default for LevMarSolver<Model, Rhs>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Float,
{
    fn default() -> Self {
        Self {
            solver: LevenbergMarquardt::default(),
            phantom: PhantomData,
        }
    }
}

/// helper structure that stores the cached calculations,
/// which are carried out by the LevMarProblem on setting the parameters
#[derive(Debug, Clone)]
struct CachedCalculationsSvd<ScalarType, ModelDim, OutputDim>
where
    ScalarType: Scalar + ComplexField,
    ModelDim: Dim,
    OutputDim: Dim + nalgebra::DimMin<ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<OutputDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<<OutputDim as DimMin<ModelDim>>::Output, ModelDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<OutputDim, <OutputDim as DimMin<ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<<OutputDim as DimMin<ModelDim>>::Output>,
{
    /// The current residual matrix of model function values belonging to the current parameters
    current_residuals: DMatrix<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    current_svd: SVD<ScalarType, OutputDim, ModelDim>,
    /// the linear coefficients `$\boldsymbol C$` providing the current best fit
    linear_coefficients: DMatrix<ScalarType>,
}

/// This is a the problem of fitting the separable model to data in a form that the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate can use it to
/// perform the least squares fit.
///
/// # Construction
///
/// Use the [LevMarProblemBuilder](self::builder::LevMarProblemBuilder) to create an instance of a
/// levmar problem.
///
/// # Usage
///
/// After obtaining an instance of `LevMarProblem` we can pass it to the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// structure of the levenberg_marquardt crate for minimization. Refer to the documentation of the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) for an overview. A usage example
/// is provided in this crate documentation as well. The [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// solver is reexported by this module as [LevMarSolver](self::LevMarSolver) for naming consistency.
///
/// # `MRHS`: Multiple Right Hand Sides
///
/// The problem generic on the boolean `MRHS` which indicates whether the
/// problem fits a single (`MRHS == false`) or multiple (`MRHS == true`) right
/// hand sides. This is decided during the building process. The underlying
/// math does not change, but the interface changes to use vectors for coefficients
/// and data in case of a single right hand side. For multiple right hand sides,
/// the coefficients and the data are matrices corresponding to columns of
/// coefficient vectors and data vectors respectively.
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct LevMarProblem<Model, Rhs: RhsType, Par: Parallelism>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
    Rhs: RhsType,
{
    /// the *weighted* data matrix to which to fit the model `$\boldsymbol{Y}_w$`.
    /// It is a matrix so it can accomodate multiple right hand sides. If
    /// the problem has only a single right hand side (MRHS = false), this is just
    /// a matrix with one column. The underlying math does not change in either case.
    /// **Attention** the data matrix is weighted with the weights if some weights
    /// where provided (otherwise it is unweighted)
    Y_w: DMatrix<Model::ScalarType>,
    /// a reference to the separable model we are trying to fit to the data
    model: Model,
    /// truncation epsilon for SVD below which all singular values are assumed zero
    svd_epsilon: <Model::ScalarType as ComplexField>::RealField,
    /// the weights of the data. If none are given, the data is not weighted
    /// If weights were provided, the builder has checked that the weights have the
    /// correct dimension for the data
    weights: Weights<Model::ScalarType, Dyn>,
    /// the currently cached calculations belonging to the currently set model parameters
    /// those are updated on set_params. If this is None, then it indicates some error that
    /// is propagated on to the levenberg-marquardt crate by also returning None results
    /// by residuals() and/or jacobian()
    cached: Option<CachedCalculationsSvd<Model::ScalarType, Dyn, Dyn>>,
    /// phantom data for the right hand sidedness of the problem
    phantom: PhantomData<(Rhs, Par)>,
}

impl<Model, Rhs: RhsType, Par: Parallelism> LevMarProblem<Model, Rhs, Par>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// convert from parallel problem to sequential one. Useful for funnelling the results
    /// of the parallel calculations to downstream tasks that only take sequential models
    /// for simplicity.
    pub fn into_sequential(self) -> LevMarProblem<Model, Rhs, Sequential> {
        let LevMarProblem {
            Y_w,
            model,
            svd_epsilon,
            weights,
            cached,
            phantom: _,
        } = self;
        LevMarProblem {
            Y_w,
            model,
            svd_epsilon,
            weights,
            cached,
            phantom: PhantomData,
        }
    }

    /// convert from sequential problem to a parallel one
    pub fn into_parallel(self) -> LevMarProblem<Model, MultiRhs, Sequential> {
        let LevMarProblem {
            Y_w,
            model,
            svd_epsilon,
            weights,
            cached,
            phantom: _,
        } = self;
        LevMarProblem {
            Y_w,
            model,
            svd_epsilon,
            weights,
            cached,
            phantom: PhantomData,
        }
    }
}

#[allow(unused)]
pub(crate) const PARALLEL_YES: bool = true;
pub(crate) const PARALLEL_NO: bool = false;

impl<Model, Rhs: RhsType, Par: Parallelism> std::fmt::Debug for LevMarProblem<Model, Rhs, Par>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevMarProblem")
            .field("y_w", &self.Y_w)
            .field("model", &"/* omitted */")
            .field("svd_epsilon", &self.svd_epsilon)
            .field("weights", &self.weights)
            .field("cached", &self.cached)
            .finish()
    }
}

impl<Model, Par: Parallelism> LevMarProblem<Model, MultiRhs, Par>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    ///
    /// # Returns
    ///
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    ///
    /// Since this method is for fitting a single right hand side, the coefficients
    /// are a single column vector.
    pub fn linear_coefficients(&self) -> Option<MatrixView<Model::ScalarType, Dyn, Dyn>> {
        self.cached
            .as_ref()
            .map(|cache| cache.linear_coefficients.as_view())
    }

    /// the weighted data matrix`$\boldsymbol{Y}_w$` to which to fit the model. Note
    /// that the weights are already applied to the data matrix and this
    /// is not the original data vector.
    ///
    /// This method is for fitting a multiple right hand sides, hence the data
    /// matrix is a matrix that contains the right hand sides as columns.
    pub fn weighted_data(&self) -> MatrixView<Model::ScalarType, Dyn, Dyn> {
        self.Y_w.as_view()
    }
}

impl<Model, Par: Parallelism> LevMarProblem<Model, SingleRhs, Par>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    /// # Returns
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    ///
    /// The linear coefficients are column vectors that are ordered
    /// into a matrix, where the column at index $$s$$ are the best linear
    /// coefficients for the member at index $$s$$ of the dataset.
    pub fn linear_coefficients(&self) -> Option<VectorView<Model::ScalarType, Dyn>> {
        self.cached
            .as_ref()
            .map(|cache|{
                debug_assert_eq!(cache.linear_coefficients.ncols(),1,
                    "coefficient matrix must have exactly one column for single right hand side. This indicates a programming error in the library.");
                cache.linear_coefficients.as_view()
            })
    }

    /// the weighted data vector `$\vec{y}_w$` to which to fit the model. Note
    /// that the weights are already applied to the data vector and this
    /// is not the original data vector.
    ///
    /// This method is for fitting a single right hand side, hence the data
    /// is a single column vector.
    pub fn weighted_data(&self) -> VectorView<Model::ScalarType, Dyn> {
        debug_assert_eq!(self.Y_w.ncols(),1,
                    "data matrix must have exactly one column for single right hand side. This indicates a programming error in the library.");
        self.Y_w.as_view()
    }
}

impl<Model, Rhs: RhsType, Par: Parallelism> LevMarProblem<Model, Rhs, Par>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// access the contained model immutably
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// get the weights of the data for the fitting problem
    pub fn weights(&self) -> &Weights<Model::ScalarType, Dyn> {
        &self.weights
    }
}

impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, Sequential>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
{
    type ResidualStorage = Owned<Model::ScalarType, Dyn>;
    type JacobianStorage = Owned<Model::ScalarType, Dyn, Dyn>;
    type ParameterStorage = Owned<Model::ScalarType, Dyn>;

    #[allow(non_snake_case)]
    /// Set the (nonlinear) model parameters `$\vec{\alpha}$` and update the internal state of the
    /// problem accordingly. The parameters are expected in the same order that the parameter
    /// names were provided in at model creation. So if we gave `&["tau","beta"]` as parameters at
    /// model creation, the function expects the layout of the parameter vector to be `$\vec{\alpha}=(\tau,\beta)^T$`.
    fn set_params(&mut self, params: &Vector<Model::ScalarType, Dyn, Self::ParameterStorage>) {
        if self.model.set_params(params.clone()).is_err() {
            self.cached = None;
        }
        // matrix of weighted model function values
        let Phi_w = self.model.eval().ok().map(|Phi| &self.weights * Phi);

        // calculate the svd
        let svd_epsilon = self.svd_epsilon;
        let current_svd = Phi_w.as_ref().map(|Phi_w| Phi_w.clone().svd(true, true));
        let linear_coefficients = current_svd
            .as_ref()
            .and_then(|svd| svd.solve(&self.Y_w, svd_epsilon).ok());

        // calculate the residuals
        let current_residuals = Phi_w
            .zip(linear_coefficients.as_ref())
            .map(|(Phi_w, coeff)| &self.Y_w - &Phi_w * coeff);

        // if everything was successful, update the cached calculations, otherwise set the cache to none
        if let (Some(current_residuals), Some(current_svd), Some(linear_coefficients)) =
            (current_residuals, current_svd, linear_coefficients)
        {
            self.cached = Some(CachedCalculationsSvd {
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
    fn params(&self) -> Vector<Model::ScalarType, Dyn, Self::ParameterStorage> {
        self.model.params()
    }

    /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
    /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
    /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
    /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
    /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
    /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn residuals(&self) -> Option<Vector<Model::ScalarType, Dyn, Self::ResidualStorage>> {
        self.cached
            .as_ref()
            .map(|cached| to_vector(cached.current_residuals.clone()))
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing
        if let Some(CachedCalculationsSvd {
            current_residuals: _,
            current_svd,
            linear_coefficients,
        }) = self.cached.as_ref()
        {
            // this is not a great pattern, but the trait bounds on copy_from
            // as of now prevent us from doing something more idiomatic
            let mut jacobian_matrix = unsafe {
                UninitMatrix::uninit(
                    Dyn(self.model.output_len() * self.Y_w.ncols()),
                    Dyn(self.model.parameter_count()),
                )
                .assume_init()
            };

            let U = current_svd.u.as_ref()?; // will return None if this was not calculated
            let U_t = U.transpose();

            //let Sigma_inverse : DMatrix<Model::ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));
            //let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

            // we use a functional style calculation here that is more easy to
            // parallelize with rayon later on. The only disadvantage is that
            // we don't short circuit anymore if there is an error in calculation,
            // but since that is the sad path anyways, we don't care about a
            // performance hit in the sad path.
            let result: Result<Vec<()>, Model::Error> = jacobian_matrix
                .column_iter_mut()
                .enumerate()
                .map(|(k, mut jacobian_col)| {
                    // weighted derivative matrix
                    let Dk = &self.weights * self.model.eval_partial_deriv(k)?; // will return none if this could not be calculated
                    let Dk_C = Dk * linear_coefficients;
                    let minus_ak = U * (&U_t * (&Dk_C)) - Dk_C;

                    //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
                    //let Dk_t_rw : DVector<Model::ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
                    //let _minus_bk : DVector<Model::ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));

                    //@todo CAUTION this relies on the fact that the
                    //elements are ordered in column major order but it avoids a copy
                    copy_matrix_to_column(minus_ak, &mut jacobian_col);
                    Ok(())
                })
                .collect::<Result<_, _>>();

            // we need this check to make sure the jacobian is returned
            // as None on error.
            result.ok()?;

            Some(jacobian_matrix)
        } else {
            None
        }
    }
}

#[cfg(feature = "parallel")]
impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, Parallel>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel + std::marker::Sync,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
{
    type ResidualStorage = Owned<Model::ScalarType, Dyn>;
    type JacobianStorage = Owned<Model::ScalarType, Dyn, Dyn>;
    type ParameterStorage = Owned<Model::ScalarType, Dyn>;

    #[allow(non_snake_case)]
    /// Set the (nonlinear) model parameters `$\vec{\alpha}$` and update the internal state of the
    /// problem accordingly. The parameters are expected in the same order that the parameter
    /// names were provided in at model creation. So if we gave `&["tau","beta"]` as parameters at
    /// model creation, the function expects the layout of the parameter vector to be `$\vec{\alpha}=(\tau,\beta)^T$`.
    fn set_params(&mut self, params: &Vector<Model::ScalarType, Dyn, Self::ParameterStorage>) {
        if self.model.set_params(params.clone()).is_err() {
            self.cached = None;
        }
        // matrix of weighted model function values
        let Phi_w = self.model.eval().ok().map(|Phi| &self.weights * Phi);

        // calculate the svd
        let svd_epsilon = self.svd_epsilon;
        let current_svd = Phi_w.as_ref().map(|Phi_w| Phi_w.clone().svd(true, true));
        let linear_coefficients = current_svd
            .as_ref()
            .and_then(|svd| svd.solve(&self.Y_w, svd_epsilon).ok());

        // calculate the residuals
        let current_residuals = Phi_w
            .zip(linear_coefficients.as_ref())
            .map(|(Phi_w, coeff)| &self.Y_w - &Phi_w * coeff);

        // if everything was successful, update the cached calculations, otherwise set the cache to none
        if let (Some(current_residuals), Some(current_svd), Some(linear_coefficients)) =
            (current_residuals, current_svd, linear_coefficients)
        {
            self.cached = Some(CachedCalculationsSvd {
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
    fn params(&self) -> Vector<Model::ScalarType, Dyn, Self::ParameterStorage> {
        self.model.params()
    }

    /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
    /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
    /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
    /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
    /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
    /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn residuals(&self) -> Option<Vector<Model::ScalarType, Dyn, Self::ResidualStorage>> {
        self.cached
            .as_ref()
            .map(|cached| to_vector(cached.current_residuals.clone()))
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing
        if let Some(CachedCalculationsSvd {
            current_residuals: _,
            current_svd,
            linear_coefficients,
        }) = self.cached.as_ref()
        {
            // this is not a great pattern, but the trait bounds on copy_from
            // as of now prevent us from doing something more idiomatic
            let mut jacobian_matrix = unsafe {
                UninitMatrix::uninit(
                    Dyn(self.model.output_len() * self.Y_w.ncols()),
                    Dyn(self.model.parameter_count()),
                )
                .assume_init()
            };

            let U = current_svd.u.as_ref()?; // will return None if this was not calculated
            let U_t = U.transpose();

            //let Sigma_inverse : DMatrix<Model::ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));
            //let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

            // we use a functional style calculation here that is more easy to
            // parallelize with rayon later on. The only disadvantage is that
            // we don't short circuit anymore if there is an error in calculation,
            // but since that is the sad path anyways, we don't care about a
            // performance hit in the sad path.
            let result: Result<Vec<()>, Model::Error> = jacobian_matrix
                .par_column_iter_mut()
                .enumerate()
                .map(|(k, mut jacobian_col)| {
                    // weighted derivative matrix
                    let Dk = &self.weights * self.model.eval_partial_deriv(k)?; // will return none if this could not be calculated
                    let Dk_C = Dk * linear_coefficients;
                    let minus_ak = U * (&U_t * (&Dk_C)) - Dk_C;

                    //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
                    //let Dk_t_rw : DVector<Model::ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
                    //let _minus_bk : DVector<Model::ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));

                    //@todo CAUTION this relies on the fact that the
                    //elements are ordered in column major order but it avoids a copy
                    copy_matrix_to_column(minus_ak, &mut jacobian_col);
                    Ok(())
                })
                .collect::<Result<_, _>>();

            // we need this check to make sure the jacobian is returned
            // as None on error.
            result.ok()?;

            Some(jacobian_matrix)
        } else {
            None
        }
    }
}

/// copy the
fn copy_matrix_to_column<T: Scalar + std::fmt::Display + Clone, S: RawStorageMut<T, Dyn>>(
    source: DMatrix<T>,
    target: &mut Matrix<T, Dyn, U1, S>,
) {
    //@todo make this more efficient...
    //@todo inefficient
    target.copy_from(&to_vector(source));
}
