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
    ComplexField, Const, DMatrix, DefaultAllocator, Dim, Dyn, Matrix, MatrixViewMut, RawStorageMut,
    RealField, Scalar, UninitMatrix, Vector, U1,
};

use nalgebra_lapack::colpiv_qr::{ColPivQrReal, ColPivQrScalar};
use num_traits::float::TotalOrder;
use num_traits::{Float, FromPrimitive};
use std::ops::Mul;

#[cfg(any(test, doctest))]
mod test;

// TODO QR
impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for SeparableProblem<Model, Rhs>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
    Model::ScalarType: ColPivQrReal + ColPivQrScalar + Float + RealField + TotalOrder,
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
            return;
        }

        // matrix of weighted model function values
        let Some(Phi) = self.model.eval().ok() else {
            self.cached = None;
            return;
        };

        let Phi_w = &self.weights * Phi;

        let Ok(decomposition) = nalgebra_lapack::ColPivQR::new(Phi_w) else {
            self.cached = None;
            return;
        };

        let Ok(linear_coefficients) = decomposition.solve(self.Y_w.clone()) else {
            self.cached = None;
            return;
        };

        self.cached = Some(CachedCalculations {
            // current_residuals,
            decomposition,
            linear_coefficients,
        })
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
        let cached = self.cached.as_ref()?;
        let mut current_residuals = self.Y_w.clone();
        // @todo handle errors
        cached
            .decomposition
            .q_tr_mul_mut(&mut current_residuals)
            .unwrap();

        let k = cached.decomposition.rank();
        current_residuals
            .view_mut((0, 0), (k as _, current_residuals.ncols()))
            .fill(Model::ScalarType::from_i8(0).unwrap());
        Some(to_vector(current_residuals))
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
            // current_residuals: _,
            decomposition,
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
                // let mut Dk = &self.weights * self.model.eval_partial_deriv(k)?; // will return none if this could not be calculated

                // // // TODO replace by correct error handling
                // let Dk = &self.weights * self.model.eval_partial_deriv(k)?;
                // let mut Dk_C = Dk * (-linear_coefficients);

                // // TODO replace by correct error handling
                // decomposition.q_tr_mul_mut(&mut Dk_C).unwrap();
                // let (m, n) = (Dk_C.nrows(), Dk_C.ncols());
                // let k = decomposition.rank();
                // Dk_C.view_mut((0, 0), (k as _, n))
                //     .fill(Model::ScalarType::from_i8(0).unwrap());

                // TODO NOTE good for MRHS (same perf as SVD)
                // this calculates (Q^T * Dk) * C, in an efficient way.
                // However, for single (or few) right hand sides, it's more efficient
                // to calculate Q^T * (Dk *C), but I couldn't yet figure this
                // out with the trait bounds using gemm and avoiding intermediate
                // allocations.
                // // TODO replace by correct error handling
                let mut Dk = &self.weights * self.model.eval_partial_deriv(k)?;
                decomposition.q_tr_mul_mut(&mut Dk).unwrap();
                let n = Dk.ncols();
                let k = decomposition.rank();
                Dk.view_mut((0, 0), (k as _, n))
                    .fill(Model::ScalarType::from_i8(0).unwrap());
                // let Dk_C = Dk * (-linear_coefficients);
                let view: MatrixViewMut<Model::ScalarType, Dyn, Dyn, _, _> =
                    jacobian_col.as_view_mut();
                view.reshape_generic::<Dyn, Dyn>(
                    Dk.shape_generic().0,
                    linear_coefficients.shape_generic().1,
                )
                .gemm(
                    Model::ScalarType::from_i8(-1).unwrap(),
                    &Dk,
                    &linear_coefficients,
                    Model::ScalarType::from_i8(0).unwrap(),
                );

                //@todo CAUTION this relies on the fact that the
                //elements are ordered in column major order but it avoids a copy
                // copy_matrix_to_column(Dk_C, &mut jacobian_col);
                Ok(())
            })
            .collect::<Result<_, _>>();

        // we need this check to make sure the jacobian is returned
        // as None on error.
        result.ok()?;

        Some(jacobian_matrix)
    }
}

// TODO SVD
// impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
//     for SeparableProblem<Model, Rhs>
// where
//     Model::ScalarType: Scalar + ComplexField + Copy,
//     <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
//         Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
//     Model: SeparableNonlinearModel,
//     DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
// {
//     type ResidualStorage = Owned<Model::ScalarType, Dyn>;
//     type JacobianStorage = Owned<Model::ScalarType, Dyn, Dyn>;
//     type ParameterStorage = Owned<Model::ScalarType, Dyn>;

//     #[allow(non_snake_case)]
//     /// Set the (nonlinear) model parameters `$\vec{\alpha}$` and update the internal state of the
//     /// problem accordingly. The parameters are expected in the same order that the parameter
//     /// names were provided in at model creation. So if we gave `&["tau","beta"]` as parameters at
//     /// model creation, the function expects the layout of the parameter vector to be `$\vec{\alpha}=(\tau,\beta)^T$`.
//     ///
//     /// This is an implementation of the [`LeastSquaresProblem::set_params`] method.
//     fn set_params(&mut self, params: &Vector<Model::ScalarType, Dyn, Self::ParameterStorage>) {
//         if self.model.set_params(params.clone()).is_err() {
//             self.cached = None;
//             return;
//         }
//         // matrix of weighted model function values
//         let Phi_w = self.model.eval().ok().map(|Phi| &self.weights * Phi);

//         // calculate the svd
//         let svd_epsilon = self.svd_epsilon;
//         let decomposition = Phi_w.as_ref().map(|Phi_w| Phi_w.clone().svd(true, true));
//         let linear_coefficients = decomposition
//             .as_ref()
//             .and_then(|decomp| decomp.solve(&self.Y_w, svd_epsilon).ok());

//         // calculate the residuals
//         let current_residuals = Phi_w
//             .zip(linear_coefficients.as_ref())
//             .map(|(Phi_w, coeff)| &self.Y_w - &Phi_w * coeff);

//         // if everything was successful, update the cached calculations, otherwise set the cache to none
//         if let (Some(current_residuals), Some(decomposition), Some(linear_coefficients)) =
//             (current_residuals, decomposition, linear_coefficients)
//         {
//             self.cached = Some(CachedCalculations {
//                 current_residuals,
//                 decomposition,
//                 linear_coefficients,
//             })
//         } else {
//             self.cached = None;
//         }
//     }

//     /// Retrieve the (nonlinear) model parameters as a vector `$\vec{\alpha}$`.
//     /// The order of the parameters in the vector is the same as the order of the parameter
//     /// names given on model creation. E.g. if the parameters at model creation where given as
//     /// `&["tau","beta"]`, then the returned vector is `$\vec{\alpha} = (\tau,\beta)^T$`, i.e.
//     /// the value of parameter `$\tau$` is at index `0` and the value of `$\beta$` at index `1`.
//     fn params(&self) -> Vector<Model::ScalarType, Dyn, Self::ParameterStorage> {
//         self.model.params()
//     }

//     /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
//     /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
//     /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
//     /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
//     /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
//     /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
//     /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
//     fn residuals(&self) -> Option<Vector<Model::ScalarType, Dyn, Self::ResidualStorage>> {
//         self.cached
//             .as_ref()
//             .map(|cached| to_vector(cached.current_residuals.clone()))
//     }

//     #[allow(non_snake_case)]
//     /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
//     /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
//     /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
//     fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
//         // TODO (Performance): make this more efficient by parallelizing
//         // but remember that just slapping rayon on the column_iter DOES NOT
//         // make it more efficient
//         let CachedCalculations {
//             current_residuals: _,
//             decomposition: current_svd,
//             linear_coefficients,
//         } = self.cached.as_ref()?;

//         let data_cols = self.Y_w.ncols();
//         let parameter_count = self.model.parameter_count();
//         // this is not a great pattern, but the trait bounds on copy_from
//         // as of now prevent us from doing something more idiomatic
//         let mut jacobian_matrix = unsafe {
//             UninitMatrix::uninit(
//                 Dyn(self.model.output_len() * data_cols),
//                 Dyn(parameter_count),
//             )
//             .assume_init()
//         };

//         let U = current_svd.u.as_ref()?; // will return None if this was not calculated
//         let U_t = U.transpose();

//         //let Sigma_inverse : DMatrix<Model::ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));
//         //let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

//         // we use a functional style calculation here that is more easy to
//         // parallelize with rayon later on. The only disadvantage is that
//         // we don't short circuit anymore if there is an error in calculation,
//         // but since that is the sad path anyways, we don't care about a
//         // performance hit in the sad path.
//         let result: Result<Vec<()>, Model::Error> = jacobian_matrix
//             .column_iter_mut()
//             .enumerate()
//             .map(|(k, mut jacobian_col)| {
//                 // weighted derivative matrix
//                 let Dk = &self.weights * self.model.eval_partial_deriv(k)?; // will return none if this could not be calculated

//                 // this orders the computations in a more efficient way for problems with multiple right hand sides,
//                 // which gives ~20-30% performance improvements for the benchmark cases
//                 let minus_ak = if data_cols <= parameter_count {
//                     let Dk_C = Dk * linear_coefficients;
//                     U * (&U_t * (&Dk_C)) - Dk_C
//                 } else {
//                     // this version is
//                     // 23-30% faster computations for MRHS benchmark
//                     (U * (&U_t * (&Dk)) - Dk) * linear_coefficients
//                 };

//                 //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
//                 //let Dk_t_rw : DVector<Model::ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
//                 //let _minus_bk : DVector<Model::ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));

//                 //@todo CAUTION this relies on the fact that the
//                 //elements are ordered in column major order but it avoids a copy
//                 copy_matrix_to_column(minus_ak, &mut jacobian_col);
//                 Ok(())
//             })
//             .collect::<Result<_, _>>();

//         // we need this check to make sure the jacobian is returned
//         // as None on error.
//         result.ok()?;

//         Some(jacobian_matrix)
//     }
// }

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

#[derive(Debug)]
pub struct LevMarProblem<Model, Solver, Rhs>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    Solver: LinearSolver<ScalarType = Model::ScalarType>,
    Rhs: RhsType,
    Model: SeparableNonlinearModel,
{
    problem: SeparableProblem<Model, Rhs>,
    cached: Option<Solver>,
}

pub trait LinearSolver {
    type ScalarType: Scalar;
    fn linear_coefficients_matrix(&self) -> Option<DMatrix<Self::ScalarType>>;
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

    fn linear_coefficients_matrix(&self) -> Option<DMatrix<Self::ScalarType>> {
        todo!()
    }
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
        Model::ScalarType: ColPivQrReal + ColPivQrScalar + Float + RealField + TotalOrder,
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

impl<Model, Rhs> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, ColPivQrLinearSolver<Model::ScalarType>, Rhs>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <<Model as SeparableNonlinearModel>::ScalarType as ComplexField>::RealField:
        Mul<Model::ScalarType, Output = Model::ScalarType> + Float,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
    Model::ScalarType: ColPivQrReal + ColPivQrScalar + Float + RealField + TotalOrder,
    Rhs: RhsType,
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
        if self.problem.model.set_params(params.clone()).is_err() {
            self.cached = None;
            return;
        }

        // matrix of weighted model function values
        let Some(Phi) = self.problem.model.eval().ok() else {
            self.cached = None;
            return;
        };

        let Phi_w = &self.problem.weights * Phi;

        let Ok(decomposition) = nalgebra_lapack::ColPivQR::new(Phi_w) else {
            self.cached = None;
            return;
        };

        let Ok(linear_coefficients) = decomposition.solve(self.problem.Y_w.clone()) else {
            self.cached = None;
            return;
        };

        self.cached = Some(ColPivQrLinearSolver {
            // current_residuals,
            decomposition,
            linear_coefficients,
        })
    }

    /// Retrieve the (nonlinear) model parameters as a vector `$\vec{\alpha}$`.
    /// The order of the parameters in the vector is the same as the order of the parameter
    /// names given on model creation. E.g. if the parameters at model creation where given as
    /// `&["tau","beta"]`, then the returned vector is `$\vec{\alpha} = (\tau,\beta)^T$`, i.e.
    /// the value of parameter `$\tau$` is at index `0` and the value of `$\beta$` at index `1`.
    fn params(&self) -> Vector<Model::ScalarType, Dyn, Self::ParameterStorage> {
        self.problem.model.params()
    }

    /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
    /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
    /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
    /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
    /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
    /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn residuals(&self) -> Option<Vector<Model::ScalarType, Dyn, Self::ResidualStorage>> {
        let cached = self.cached.as_ref()?;
        let mut current_residuals = self.problem.Y_w.clone();
        // @todo handle errors
        cached
            .decomposition
            .q_tr_mul_mut(&mut current_residuals)
            .unwrap();

        let k = cached.decomposition.rank();
        current_residuals
            .view_mut((0, 0), (k as _, current_residuals.ncols()))
            .fill(Model::ScalarType::from_i8(0).unwrap());
        Some(to_vector(current_residuals))
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing
        // but remember that just slapping rayon on the column_iter DOES NOT
        // make it more efficient
        let ColPivQrLinearSolver {
            // current_residuals: _,
            decomposition,
            linear_coefficients,
        } = self.cached.as_ref()?;

        let data_cols = self.problem.Y_w.ncols();
        let parameter_count = self.problem.model.parameter_count();
        // this is not a great pattern, but the trait bounds on copy_from
        // as of now prevent us from doing something more idiomatic
        let mut jacobian_matrix = unsafe {
            UninitMatrix::uninit(
                Dyn(self.problem.model.output_len() * data_cols),
                Dyn(parameter_count),
            )
            .assume_init()
        };

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
                // let mut Dk = &self.weights * self.model.eval_partial_deriv(k)?; // will return none if this could not be calculated

                // // // TODO replace by correct error handling
                // let Dk = &self.weights * self.model.eval_partial_deriv(k)?;
                // let mut Dk_C = Dk * (-linear_coefficients);

                // // TODO replace by correct error handling
                // decomposition.q_tr_mul_mut(&mut Dk_C).unwrap();
                // let (m, n) = (Dk_C.nrows(), Dk_C.ncols());
                // let k = decomposition.rank();
                // Dk_C.view_mut((0, 0), (k as _, n))
                //     .fill(Model::ScalarType::from_i8(0).unwrap());

                // TODO NOTE good for MRHS (same perf as SVD)
                // this calculates (Q^T * Dk) * C, in an efficient way.
                // However, for single (or few) right hand sides, it's more efficient
                // to calculate Q^T * (Dk *C), but I couldn't yet figure this
                // out with the trait bounds using gemm and avoiding intermediate
                // allocations.
                // // TODO replace by correct error handling
                let mut Dk = &self.problem.weights * self.problem.model.eval_partial_deriv(k)?;
                decomposition.q_tr_mul_mut(&mut Dk).unwrap();
                let n = Dk.ncols();
                let k = decomposition.rank();
                Dk.view_mut((0, 0), (k as _, n))
                    .fill(Model::ScalarType::from_i8(0).unwrap());
                // let Dk_C = Dk * (-linear_coefficients);
                let view: MatrixViewMut<Model::ScalarType, Dyn, Dyn, _, _> =
                    jacobian_col.as_view_mut();
                view.reshape_generic::<Dyn, Dyn>(
                    Dk.shape_generic().0,
                    linear_coefficients.shape_generic().1,
                )
                .gemm(
                    Model::ScalarType::from_i8(-1).unwrap(),
                    &Dk,
                    &linear_coefficients,
                    Model::ScalarType::from_i8(0).unwrap(),
                );

                //@todo CAUTION this relies on the fact that the
                //elements are ordered in column major order but it avoids a copy
                // copy_matrix_to_column(Dk_C, &mut jacobian_col);
                Ok(())
            })
            .collect::<Result<_, _>>();

        // we need this check to make sure the jacobian is returned
        // as None on error.
        result.ok()?;

        Some(jacobian_matrix)
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
            coefficients.as_view(),
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
