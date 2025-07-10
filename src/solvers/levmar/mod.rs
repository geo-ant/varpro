use crate::fit::FitResult;
use crate::prelude::*;
use crate::problem::{CachedCalculations, RhsType, SeparableProblem, SingleRhs};
use crate::statistics::FitStatistics;
use crate::util::to_vector;
use levenberg_marquardt::LeastSquaresProblem;
/// Type alias for the solver of the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate.
/// This provides the Levenberg-Marquardt nonlinear least squares optimization algorithm.
// pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::storage::Owned;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dyn, Matrix, RawStorageMut, RealField, Scalar,
    UninitMatrix, Vector, U1,
};
use num_traits::{Float, FromPrimitive};
use std::ops::Mul;

#[cfg(any(test, doctest))]
mod test;

impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for SeparableProblem<Model, Rhs>
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
    ///
    /// This is an implementation of the [`LeastSquaresProblem::set_params`] method.
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
        // but remember that just slapping rayon on the column_iter DOES NOT
        // make it more efficient
        let CachedCalculations {
            current_residuals: _,
            current_svd,
            linear_coefficients,
        } = self.cached.as_ref()?;

        let data_cols = self.Y_w.ncols();
        let parameter_count = self.model.parameter_count();
        // this is not a great pattern, but the trait bounds on copy_from
        // as of now prevent us from doing something more idiomatic
        let mut jacobian_matrix = unsafe {
            UninitMatrix::uninit(
                Dyn(self.model.output_len() * data_cols),
                Dyn(parameter_count),
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

                // this orders the computations in a more efficient way for problems with multiple right hand sides,
                // which gives ~20-30% performance improvements for the benchmark cases
                let minus_ak = if data_cols <= parameter_count {
                    let Dk_C = Dk * linear_coefficients;
                    U * (&U_t * (&Dk_C)) - Dk_C
                } else {
                    // this version is
                    // 23-30% faster computations for MRHS benchmark
                    (U * (&U_t * (&Dk)) - Dk) * linear_coefficients
                };

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
    }
}

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

impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    /// creata a new solver using the given underlying solver. This allows
    /// us to configure the underlying solver with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self { solver }
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
    pub fn fit<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        #[allow(deprecated)]
        let (problem, report) = self.solver.minimize(problem);
        let result = FitResult::new(problem, report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }
}

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

        match FitStatistics::try_calculate(
            problem.model(),
            problem.weighted_data(),
            problem.weights(),
            coefficients,
        ) {
            Ok(statistics) => Ok((FitResult::new(problem, minimization_report), statistics)),
            Err(_) => Err(FitResult::new(problem, minimization_report)),
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
