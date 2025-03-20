use super::{
    copy_matrix_to_column, to_vector, LevMarProblem, MatrixDecomposition, RhsType,
    SeparableNonlinearModel, Sequential,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{
    ComplexField, DMatrix, DefaultAllocator, Dyn, Matrix, OMatrix, Owned, Scalar, UninitMatrix,
    Vector,
};
use num_traits::Float;
use std::ops::Mul;

/// a wrapper around the QR decomposition without column pivoting of a matrix.
/// The matrix to be decomposed is assumed to have full rank.
#[derive(Debug)]
pub struct QrDecomposition<ScalarType: Scalar + ComplexField> {
    /// number of columns of the original matrix
    n: usize,
    current_qr: nalgebra::linalg::QR<ScalarType, Dyn, Dyn>,
}

impl<ScalarType: Scalar + ComplexField> MatrixDecomposition<ScalarType>
    for QrDecomposition<ScalarType>
{
    #[allow(non_snake_case)]
    fn linear_coefficients(&self, Y_w: &DMatrix<ScalarType>) -> Option<DMatrix<ScalarType>> {
        self.solve(Y_w)
    }
}

/// An abstraction over the QR decomposition of a matrix `$A$` with and
/// without column-pivoting. In the case of column-pivoting, the matrix
/// is implied to have full rank, whereas in the case of the
trait QrExt<ScalarType>: Sized + MatrixDecomposition<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    /// calculate the decomposition, return None in case of failure
    fn new(matrix: DMatrix<ScalarType>) -> Option<Self>;
    /// solve the equation `$A x = B$` for `$x$`, where `A` is the original
    /// matrix that was decomposed. Return None if solving failed.
    #[allow(non_snake_case)]
    fn solve(&self, B: &DMatrix<ScalarType>) -> Option<DMatrix<ScalarType>>;
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

impl<ScalarType: Scalar + ComplexField> QrExt<ScalarType> for QrDecomposition<ScalarType> {
    fn new(matrix: DMatrix<ScalarType>) -> Option<Self> {
        let n = matrix.ncols();
        let qr = nalgebra::linalg::QR::new(matrix);
        Some(Self { n, current_qr: qr })
    }

    #[allow(non_snake_case)]
    fn solve(&self, B: &DMatrix<ScalarType>) -> Option<DMatrix<ScalarType>> {
        self.current_qr.solve(B)
    }

    #[allow(non_snake_case)]
    fn q2_tr_mul(&self, mut M: DMatrix<ScalarType>) -> DMatrix<ScalarType> {
        // after this, M contains Q^T M
        self.current_qr.q_tr_mul(&mut M);
        // illegal dimensions for matrix M
        assert!(M.ncols() >= self.n, "illegal dimensions for matrix M");
        // this clipping corresponds to the product Q_2^T M
        let Q2M = M.view_range(self.n..M.ncols(), 0..);
        Q2M.into()
    }
}

//@note(geo) we cannot simply blanket implement this for the Qr supertrait because
// that will get us errors about conflicting implementations, since Svd could
// theoretically also implement the Qr supertrait, although that's not going to happen
impl<Model, Rhs: RhsType> LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>
    for LevMarProblem<Model, Rhs, Sequential, QrDecomposition<Model::ScalarType>>
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
        let current_qr = Phi_w.and_then(|mat| QrDecomposition::new(mat));
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
        if let Some(QrDecomposition { current_qr, .. }) = self.cached.as_ref() {
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
            let C = current_qr.solve(&self.Y_w)?;

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
