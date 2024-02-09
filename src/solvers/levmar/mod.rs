use crate::prelude::*;
use crate::statistics::FitStatistics;
use levenberg_marquardt::{LeastSquaresProblem, MinimizationReport};
use nalgebra::storage::Owned;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dim, DimMin, Dyn, Matrix, OVector, RawStorageMut,
    RealField, Scalar, UninitMatrix, Vector, SVD, U1,
};

mod builder;
#[cfg(any(test, doctest))]
mod test;

use crate::util::{to_vector, Weights};
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
    pub fn nonlinear_parameters(&self) -> OVector<Model::ScalarType, Dyn> {
        self.problem.model().params()
    }

    /// convenience function to get the linear coefficients after the fit has
    /// finished
    ///
    /// in case of multiple datasets, the coefficients vectors for the individual
    /// members of the datasets are the colums of the matrix. If the problem
    /// only has a single right hand side, then the matrix will have one column
    /// only.
    pub fn linear_coefficients(&self) -> Option<DMatrix<Model::ScalarType>> {
        self.problem.linear_coefficients().cloned()
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
{
    /// create a new solver with default parameters. Uses the underlying
    /// solver of the `levenberg_marquardt` crate
    #[deprecated(
        since = "0.8.0",
        note = "use the default() method instead. If you want to specify solver parameters use the with_solver(...) constructor."
    )]
    pub fn new() -> Self
    where
        Model::ScalarType: RealField + Float,
    {
        Self::default()
    }

    /// creata a new solver using the given underlying solver. This allows
    /// us to configure the underlying with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self { solver }
    }

    /// Solve the fitting problem. This method is deprecated, use the fit method instead.
    #[deprecated(since = "0.7.0", note = "use the fit(...) method instead")]
    pub fn minimize(
        &self,
        problem: LevMarProblem<Model>,
    ) -> (LevMarProblem<Model>, MinimizationReport<Model::ScalarType>)
    where
        Model: SeparableNonlinearModel,
        LevMarProblem<Model>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float + FromPrimitive,
    {
        self.solver.minimize(problem)
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
    pub fn fit(&self, problem: LevMarProblem<Model>) -> Result<FitResult<Model>, FitResult<Model>>
    where
        Model: SeparableNonlinearModel,
        LevMarProblem<Model>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        #[allow(deprecated)]
        let (problem, report) = self.minimize(problem);
        let result = FitResult::new(problem, report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }

    /// Same as the [LevMarProblem::fit] function but also calculates some additional
    /// statistical information about the fit, if the fit was successful.
    ///
    /// # Returns
    ///
    /// See also the [LevMarProblem::fit] function, but on success also returns statistical
    /// information about the fit.
    #[allow(clippy::result_large_err)]
    pub fn fit_with_statistics(
        &self,
        problem: LevMarProblem<Model>,
    ) -> Result<(FitResult<Model>, FitStatistics<Model>), FitResult<Model>>
    where
        Model: SeparableNonlinearModel,
        LevMarProblem<Model>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
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

impl<Model> Default for LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Float,
{
    fn default() -> Self {
        Self {
            solver: LevenbergMarquardt::default(),
        }
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
#[allow(non_snake_case)]
pub struct LevMarProblem<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// the *weighted* data vector to which to fit the model `$\vec{y}_w$`
    /// **Attention** the data vector is weighted with the weights if some weights
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
    cached: Option<CachedCalculations<Model::ScalarType, Dyn, Dyn>>,
}

impl<Model> std::fmt::Debug for LevMarProblem<Model>
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

impl<Model> LevMarProblem<Model>
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
    pub fn linear_coefficients(&self) -> Option<&DMatrix<Model::ScalarType>> {
        self.cached.as_ref().map(|cache| &cache.linear_coefficients)
    }

    /// access the contained model immutably
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// get the weights of the data for the fitting problem
    pub fn weights(&self) -> &Weights<Model::ScalarType, Dyn> {
        &self.weights
    }

    /// the weighted data vector `$\vec{y}_w$` to which to fit the model. Note
    /// that the weights are already applied to the data vector and this
    /// is not the original data vector.
    pub fn weighted_data(&self) -> &DMatrix<Model::ScalarType> {
        &self.Y_w
    }
}

impl<Model> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn> for LevMarProblem<Model>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<
        (<Model::ScalarType as ComplexField>::RealField, usize),
        Dyn,
    >,
    DefaultAllocator:
        nalgebra::allocator::Allocator<<Model::ScalarType as ComplexField>::RealField, Dyn>,
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
        if let Some(CachedCalculations {
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

/// copy the
fn copy_matrix_to_column<T: Scalar + std::fmt::Display + Clone, S: RawStorageMut<T, Dyn>>(
    source: DMatrix<T>,
    target: &mut Matrix<T, Dyn, U1, S>,
) {
    //@todo make this more efficient...
    //@todo inefficient
    target.copy_from(&to_vector(source));
}
