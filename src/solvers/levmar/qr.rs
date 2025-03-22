use super::{
    copy_matrix_to_column, to_vector, LevMarProblem, MatrixDecomposition, RhsType,
    SeparableNonlinearModel,
};
use faer::{
    linalg::solvers::SolveLstsqCore,
    mat::{AsMatMut, AsMatRef},
};
use faer_ext::{IntoFaer, IntoNalgebra};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{
    ComplexField, DMatrix, DMatrixView, DefaultAllocator, Dyn, Matrix, Owned, Scalar, UninitMatrix,
    Vector,
};
use num_traits::Float;
use std::ops::Mul;

/// a wrapper around the QR decomposition without column pivoting of a matrix.
/// The matrix to be decomposed is assumed to have full rank.
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct QrDecomposition<ScalarType: Scalar + ComplexField + faer_traits::RealField> {
    /// number of columns of the original matrix
    n: usize,
    qr_decomp: faer::linalg::solvers::Qr<ScalarType>,
    Q2_t: faer::Mat<ScalarType>,
}

impl<ScalarType: Scalar + ComplexField + faer_traits::RealField> MatrixDecomposition<ScalarType>
    for QrDecomposition<ScalarType>
{
    #[allow(non_snake_case)]
    fn linear_coefficients(&self, Y_w: &DMatrix<ScalarType>) -> Option<DMatrix<ScalarType>> {
        self.solve(Y_w.as_view())
    }
}

/// An abstraction over the QR decomposition of a matrix `$A$` with and
/// without column-pivoting. In the case of column-pivoting, the matrix
/// is implied to have full rank, whereas in the case of the
trait QrExt<ScalarType>: Sized + MatrixDecomposition<ScalarType>
where
    ScalarType: Scalar + ComplexField + faer_traits::RealField,
{
    /// calculate the decomposition, return None in case of failure
    fn new(matrix: DMatrixView<ScalarType>) -> Option<Self>;
    /// solve the equation `$A x = B$` for `$x$`, where `A` is the original
    /// matrix that was decomposed. Return None if solving failed.
    #[allow(non_snake_case)]
    fn solve(&self, B: DMatrixView<ScalarType>) -> Option<DMatrix<ScalarType>>;
    /// This calculates the matrix product `$Q_2^T * M$`, where `$M$` is a suitably
    /// shaped matrix and `$Q_2$` is part of the matrix `$Q$`, defined as follows:
    /// `$Q = \left[ Q_1 \vert Q_2\right]$`, where `$Q_1$` contains the first
    /// `$r = \text{rank}(A)$` columns of `$Q$` and `$Q_2$` contains the rest,
    /// where `$A$ \in mathbb{R}^{m\times n}` is the original matrix being decomposed.
    /// For a QR decomposition without column pivoting, is is assumed that
    /// `$A$` has full rank and thus `$r = n$`.
    #[allow(non_snake_case)]
    fn q2_tr_mul(&self, M: DMatrix<ScalarType>) -> DMatrix<ScalarType>;
}

impl<ScalarType: Scalar + ComplexField + faer_traits::RealField> QrExt<ScalarType>
    for QrDecomposition<ScalarType>
{
    fn new(matrix: DMatrixView<ScalarType>) -> Option<Self> {
        let m = matrix.nrows();
        let n = matrix.ncols();

        if matrix.nrows() <= matrix.ncols() {
            // we only deal with overdetermined systems here
            return None;
        }

        let faer_view = matrix.into_faer();
        let qr_decomp = faer_view.qr();
        let Q = qr_decomp.compute_Q();
        let (Q1, Q2) = Q.split_at_col(n);
        debug_assert_eq!(Q1.ncols(), n);
        debug_assert_eq!(Q2.ncols(), m - n);

        Some(Self {
            n,
            Q2_t: Q2.transpose().to_owned(),
            qr_decomp,
        })
    }

    #[allow(non_snake_case)]
    fn solve(&self, B: DMatrixView<ScalarType>) -> Option<DMatrix<ScalarType>> {
        let faer_view = B.into_faer();
        let mut X = faer_view.to_owned();
        self.qr_decomp
            .solve_lstsq_in_place_with_conj(faer::Conj::No, X.as_mat_mut());
        Some(X.as_mat_ref().into_nalgebra().into())
    }

    #[allow(non_snake_case)]
    fn q2_tr_mul(&self, mut M: DMatrix<ScalarType>) -> DMatrix<ScalarType> {
        let faer_view = <DMatrixView<_> as IntoFaer>::into_faer(M.as_view());
        let prod = self.Q2_t.as_mat_ref() * faer_view;
        prod.as_mat_ref().into_nalgebra().into()
    }
}

//@note(geo) we cannot simply blanket implement this for the Qr supertrait because
// that will get us errors about conflicting implementations, since Svd could
// theoretically also implement the Qr supertrait, although that's not going to happen
impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, QrDecomposition<Model::ScalarType>>
where
    Model::ScalarType: Scalar + ComplexField + Copy + faer_traits::RealField,
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
        let current_qr = Phi_w.and_then(|mat| QrDecomposition::new(mat.as_view()));
        self.cached = current_qr;
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
        let qr = &self.cached.as_ref()?;
        Some(to_vector(qr.q2_tr_mul(self.Y_w.clone())))
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<Model::ScalarType, Dyn, Dyn, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing
        if let Some(current_qr) = self.cached.as_ref() {
            // this is not a great pattern, but the trait bounds on copy_from
            // as of now prevent us from doing something more idiomatic
            let mut jacobian_matrix = unsafe {
                UninitMatrix::uninit(
                    Dyn(self.model.output_len() * self.Y_w.ncols()),
                    Dyn(self.model.parameter_count()),
                )
                .assume_init()
            };

            // matrix C of linear coefficients
            let C = current_qr.solve(self.Y_w.as_view())?;

            let current_qr = self.cached.as_ref()?;
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

                    //@todo(geo) the order of matrix operations should be evaluated based on
                    // the size of the C matrix
                    let minus_ak = current_qr.q2_tr_mul(Dk * &C);
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
