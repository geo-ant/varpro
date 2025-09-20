use super::{LevMarProblem, LinearSolver};
use crate::{model::SeparableNonlinearModel, problem::RhsType, util::to_vector};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dyn, Matrix, MatrixViewMut, Owned, RealField, Scalar,
    UninitMatrix, Vector,
};
use nalgebra_lapack::colpiv_qr::{ColPivQrReal, ColPivQrScalar};
use num_traits::{float::TotalOrder, Float, FromPrimitive};
use std::ops::Mul;

/// caches the calculations for the implementation of the LevMarProblem
/// with column-pivoted QR decomposition.
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

    fn linear_coefficients_matrix(self) -> DMatrix<Self::ScalarType> {
        self.linear_coefficients
    }
}

impl<Model, Rhs> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, ColPivQrLinearSolver<Model::ScalarType>>
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
        if self
            .separable_problem
            .model
            .set_params(params.clone())
            .is_err()
        {
            self.cached = None;
            return;
        }

        // matrix of weighted model function values
        let Some(Phi) = self.separable_problem.model.eval().ok() else {
            self.cached = None;
            return;
        };

        let Phi_w = &self.separable_problem.weights * Phi;

        let Ok(decomposition) = nalgebra_lapack::ColPivQR::new(Phi_w) else {
            self.cached = None;
            return;
        };

        let Ok(linear_coefficients) = decomposition.solve(self.separable_problem.Y_w.clone())
        else {
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
        self.separable_problem.model.params()
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
        let mut current_residuals = self.separable_problem.Y_w.clone();
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

        let data_cols = self.separable_problem.Y_w.ncols();
        let parameter_count = self.separable_problem.model.parameter_count();
        // this is not a great pattern, but the trait bounds on copy_from
        // as of now prevent us from doing something more idiomatic
        let mut jacobian_matrix = unsafe {
            UninitMatrix::uninit(
                Dyn(self.separable_problem.model.output_len() * data_cols),
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
                let mut Dk = &self.separable_problem.weights
                    * self.separable_problem.model.eval_partial_deriv(k)?;
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
