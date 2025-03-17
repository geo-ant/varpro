use nalgebra::{ComplexField, DMatrix, Dyn, Scalar};

use crate::util::Weights;

use super::SeparableNonlinearModel;

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct LevMarProblemQr<Model, QrDecomp, const MRHS: bool>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
    QrDecomp: AbstractQr<Model::ScalarType>,
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
    cached: Option<CachedCalculationsQr<Model::ScalarType, QrDecomp>>,
}

pub struct QrDecomposition<ScalarType: Scalar + ComplexField> {
    /// number of columns of the original matrix
    n: usize,
    qr: nalgebra::linalg::QR<ScalarType, Dyn, Dyn>,
}

/// An abstraction over the QR decomposition of a matrix `$A$` with and
/// without column-pivoting. In the case of column-pivoting, the matrix
/// is implied to have full rank, whereas in the case of the
pub trait AbstractQr<ScalarType>: Sized
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
    fn q2_tr_mul(&self, M: &DMatrix<ScalarType>) -> DMatrix<ScalarType>;
}

impl<ScalarType: Scalar + ComplexField> AbstractQr<ScalarType> for QrDecomposition<ScalarType> {
    fn new(matrix: DMatrix<ScalarType>) -> Option<Self> {
        let n = matrix.ncols();
        let qr = nalgebra::linalg::QR::new(matrix);
        Some(Self { n, qr })
    }

    #[allow(non_snake_case)]
    fn solve(&self, B: &DMatrix<ScalarType>) -> Option<DMatrix<ScalarType>> {
        self.qr.solve(B)
    }

    #[allow(non_snake_case)]
    fn q2_tr_mul(&self, M: &DMatrix<ScalarType>) -> DMatrix<ScalarType> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct CachedCalculationsQr<ScalarType, QrDecomp>
where
    ScalarType: Scalar + ComplexField,
    QrDecomp: AbstractQr<ScalarType>,
{
    /// The current residual matrix of model function values belonging to the current parameters
    current_residuals: DMatrix<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    current_svd: QrDecomp,
    /// the linear coefficients `$\boldsymbol C$` providing the current best fit
    linear_coefficients: DMatrix<ScalarType>,
}
