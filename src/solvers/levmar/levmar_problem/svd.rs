use super::{LevMarProblem, LinearSolver};
use crate::{model::SeparableNonlinearModel, problem::RhsType, util::to_vector};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dyn, Matrix, MatrixViewMut, Owned, RealField, Scalar,
    UninitMatrix, Vector, SVD,
};
use num_traits::{float::TotalOrder, Float, FromPrimitive, Zero};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub(crate) struct SvdSolver<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    /// The current residual matrix of model function values belonging to the current parameters
    pub(crate) current_residuals: DMatrix<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    pub(crate) decomposition: SVD<ScalarType, Dyn, Dyn>,
    /// the linear coefficients `$\boldsymbol C$` providing the current best fit
    pub(crate) linear_coefficients: DMatrix<ScalarType>,
}

impl<ScalarType> LinearSolver for SvdSolver<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    type ScalarType = ScalarType;

    fn linear_coefficients_matrix(&self) -> DMatrix<Self::ScalarType> {
        self.linear_coefficients.clone()
    }
}

impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, SvdSolver<Model::ScalarType>>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    <Model::ScalarType as ComplexField>::RealField: Float + Zero,
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
        if self
            .separable_problem
            .model
            .set_params(params.clone())
            .is_err()
        {
            self.cached = None;
            return;
        }

        let Ok(Phi) = self.separable_problem.model.eval() else {
            self.cached = None;
            return;
        };

        // matrix of weighted model function values
        // @note(geo-ant) this matrix multiplication is fast for diagonal or unit
        // weights, which is all we allow for now.
        let Phi_w = &self.separable_problem.weights * Phi;

        let max_dim_phiw = Phi_w.ncols().max(Phi_w.nrows());

        // calculate the svd
        let current_svd = Phi_w.clone().svd(true, true);

        // NOTE: since I don't want to have to deal with exposing the svd_epsilon
        // to the outside, I'll use the same heuristic that OCTAVE uses for it's
        // rank function, see https://octave.sourceforge.io/octave/function/rank.html.
        // This should be fine, since a much simpler heuristic was used previously
        // (just compare against a fixed number which was usually floating
        // point epilon) and that also seemed to work fine.
        let svd_epsilon = current_svd.singular_values.max()
            * <Model::ScalarType as ComplexField>::RealField::from_usize(max_dim_phiw)
                .expect("integer size out of floating point bounds")
            * <Model::ScalarType as ComplexField>::RealField::epsilon();

        let Ok(linear_coefficients) =
            current_svd.solve(&self.separable_problem.Y_w, svd_epsilon.real())
        else {
            self.cached = None;
            return;
        };

        // calculate the residuals
        let current_residuals = &self.separable_problem.Y_w - Phi_w * &linear_coefficients;

        // if everything was successful, update the cached calculations, otherwise set the cache to none
        self.cached = Some(SvdSolver {
            current_residuals,
            decomposition: current_svd,
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
        let SvdSolver {
            current_residuals: _,
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

        let U = decomposition.u.as_ref()?; // will return None if this was not calculated
        let U_t = U.transpose();

        let one = Model::ScalarType::from_i8(1).unwrap();

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
                let Dk = &self.separable_problem.weights
                    * self.separable_problem.model.eval_partial_deriv(k)?; // will return none if this could not be calculated

                // this creates a view of the jacobian column with the same
                // shape as Dk*C, which is the right thing to do, because
                // that column is just the vectorized (i.e. stacked column)
                // form of a matrix of that shape
                let view: MatrixViewMut<Model::ScalarType, Dyn, Dyn, _, _> =
                    jacobian_col.as_view_mut();
                let mut dkc_shaped_jacobian: Matrix<Model::ScalarType, Dyn, Dyn, _> = view
                    .reshape_generic::<Dyn, Dyn>(
                        Dk.shape_generic().0,
                        linear_coefficients.shape_generic().1,
                    );
                // this orders the computations in a more efficient way for problems with multiple right hand sides,
                // which gives ~20-30% performance improvements for the benchmark cases
                if data_cols <= parameter_count {
                    // formula:
                    // j_k = vec(U * (&U_t * (&Dk_C)) - Dk_C), where Dk_C = Dk*C
                    // where vec is an operation that stacks the columns of a matrix
                    // to form a column vector.
                    // We just use mul_to and gemm operations to avoid as many
                    // temporary allocations and copies as we can.

                    Dk.mul_to(linear_coefficients, &mut dkc_shaped_jacobian);
                    // now the jacobian contains Dk*C

                    // temporary result for U_t * Dk_C
                    let Ut_DkC = &U_t * &dkc_shaped_jacobian;
                    // then use gemm to calculate the full result according
                    // to the formula above
                    dkc_shaped_jacobian.gemm(one, &U, &Ut_DkC, -one);
                } else {
                    // formula: j_k = vec( (U * (&U_t * (&Dk)) - Dk) * linear_coefficients)
                    // even using this version without gemm and mul_to is
                    // 23-30% faster computations for MRHS benchmark.
                    //
                    // Additionally we also use primitive operations to avoid
                    // intermediate allocations and copies as much as we can

                    // intermediate result
                    let Ut_Dk = &U_t * &Dk;
                    let mut Dk = Dk;
                    Dk.gemm(one, &U, &Ut_Dk, -one);
                    // now Dk contains the result of (U * (&U_t * (&Dk)) - Dk)
                    Dk.mul_to(&linear_coefficients, &mut dkc_shaped_jacobian);
                };

                //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
                //let Dk_t_rw : DVector<Model::ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
                //let _minus_bk : DVector<Model::ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));

                Ok(())
            })
            .collect::<Result<_, _>>();

        // we need this check to make sure the jacobian is returned
        // as None on error.
        result.ok()?;

        Some(jacobian_matrix)
    }
}
