use crate::util::DiagMatrix;
use nalgebra::{ComplexField, DefaultAllocator, Dim, Matrix, OVector, RawStorageMut, Scalar};
use std::ops::Mul;

/// a variant for different weights that can be applied to a least squares problem
/// Right now covers only either unit weights (i.e. unweighted problem) or a diagonal
/// matrix for the weights. Can easily be extended in the future, because this structure
/// offers an interface for matrix-matrix multiplication and matrix-vector multiplication
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// unit weights, which means the problem is unweighted
    Unit,
    /// the weights are represented by a diagonal matrix
    Diagonal(DiagMatrix<ScalarType, D>),
}

impl<ScalarType, D> Weights<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// create diagonal weights with the given diagonal elements of a matrix.
    /// The resulting diagonal matrix is a square matrix with the given diagonal
    /// elements and all off-diagonal elements set to zero
    /// Make sure that the dimensions of the weights match the data that they
    /// should be applied to
    pub fn diagonal(diagonal: OVector<ScalarType, D>) -> Self {
        Self::from(DiagMatrix::from(diagonal))
    }

    /// check that the weights are appropriately sized for the given data vector, so that
    /// they can be applied without panic. For unit weights this is always true, but for diagonal
    /// weights it is not.
    /// # Arguments
    /// * `data_len`: the number of elements in the data vector
    pub fn is_size_correct_for_data_length(&self, data_len: usize) -> bool {
        match self {
            Weights::Unit => true,
            Weights::Diagonal(diag) => diag.size() == data_len,
        }
    }
}

/// Get a variant representing unit weights (i.e. unweighted problem)
impl<ScalarType, D> Default for Weights<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn default() -> Self {
        Self::Unit
    }
}

/// create diagonal weights using the given diagonal matrix
impl<ScalarType, D> From<DiagMatrix<ScalarType, D>> for Weights<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn from(diag: DiagMatrix<ScalarType, D>) -> Self {
        Self::Diagonal(diag)
    }
}

/// A convenience method that allows to multiply weights to a matrix from the left.
/// This performs the matrix multiplication corresponding to the weight matrix. However,
/// since the method knows e.g. if the weights are diagonal or unit it can take shortcuts
/// and make the operation more efficient. It is a no-op if the weights are unit.
/// # Panics
/// If the matrix matrix multiplication fails because of incorrect dimensions.
/// (unit weights never panic)
#[allow(non_snake_case)]
impl<ScalarType, R, C, S> Mul<Matrix<ScalarType, R, C, S>> for &Weights<ScalarType, R>
where
    ScalarType: Mul<ScalarType, Output = ScalarType> + Scalar + ComplexField,
    C: Dim,
    R: Dim,
    S: RawStorageMut<ScalarType, R, C>,
    DefaultAllocator: nalgebra::allocator::Allocator<R>,
    DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
{
    type Output = Matrix<ScalarType, R, C, S>;

    fn mul(self, rhs: Matrix<ScalarType, R, C, S>) -> Self::Output {
        match self {
            Weights::Unit => rhs,
            Weights::Diagonal(W) => W * rhs,
        }
    }
}

#[cfg(any(test, doctest))]
mod test {
    use crate::util::weights::Weights;
    use nalgebra::{DMatrix, DVector};

    #[test]
    #[allow(non_snake_case)]
    fn unit_weight_produce_correct_results_when_multiplied_to_matrix_or_vector() {
        let W = Weights::default();
        let v = DVector::from(vec![1., 3., 3., 7.]);
        let A = DMatrix::from_element(4, 4, 2.0);

        assert_eq!(&W * v.clone(), v);
        assert_eq!(&W * A.clone(), A);
    }

    #[test]
    #[allow(non_snake_case)]
    fn diagonal_weights_produce_correct_results_when_multiplied_to_matrix_or_vector() {
        let diagonal = DVector::from(vec![3., 78., 6., 5.]);
        let D = DMatrix::from_diagonal(&diagonal);
        let W = Weights::diagonal(diagonal);

        let v = DVector::from(vec![1., 3., 3., 7.]);
        let mut A = DMatrix::from_element(4, 2, 0.);
        A.set_column(0, &DVector::from(vec![32., 5., 86., 51.]));
        A.set_column(1, &DVector::from(vec![65., 46., 8., 85.]));

        assert_eq!(&D * &v, &W * v);
        assert_eq!(&D * &A, &W * A);
    }
}
