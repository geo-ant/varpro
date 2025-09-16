#[cfg(any(test, doctest))]
mod test;
use nalgebra::U1;
use nalgebra::{
    ComplexField, DVector, DefaultAllocator, Dim, Dyn, Matrix, OMatrix, OVector, RawStorageMut,
    Scalar,
};
use std::ops::Mul;

mod weights;
pub use weights::Weights;

/// A square diagonal matrix with dynamic dimension. Off-diagonal entries are assumed zero.
/// This internally stores only the diagonal elements
/// # Types
/// ScalarType: the numeric type of the matrix
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagMatrix<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    diagonal: OVector<ScalarType, D>,
}

impl<ScalarType, D> DiagMatrix<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// get the number of columns of the matrix
    /// The matrix is square, so this is equal to the number of rows
    pub fn ncols(&self) -> usize {
        self.size()
    }

    /// get the number of rows of the matrix
    /// The matrix is square, so this is equal to the number of columns
    pub fn nrows(&self) -> usize {
        self.size()
    }

    /// the size (i.e. number of rows == number of cols) of this square matrix
    pub fn size(&self) -> usize {
        self.diagonal.len()
    }

    /// Generate a square matrix containing the entries of the vector which
    /// contains only real field values of this (potentially) complex type
    pub fn from_real_field(diagonal: OVector<ScalarType::RealField, D>) -> Self
    where
        DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        Self::from(diagonal.map(ScalarType::from_real))
    }
}
/// Generate a square diagonal matrix from the given diagonal vector.
impl<ScalarType, D> From<OVector<ScalarType, D>> for DiagMatrix<ScalarType, D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn from(diagonal: OVector<ScalarType, D>) -> Self {
        Self { diagonal }
    }
}

/// Multiply this diagonal matrix from the left to a dynamically sized matrix.
/// # Panics
/// Panics if the dimensions of the matrices do not fit for matrix multiplication
/// # Result
/// The result of the matrix multiplication as a new dynamically sized matrix
impl<ScalarType, R, C, S> Mul<Matrix<ScalarType, R, C, S>> for &DiagMatrix<ScalarType, R>
where
    ScalarType: Mul<ScalarType, Output = ScalarType> + Scalar + ComplexField,
    C: Dim,
    R: Dim,
    S: RawStorageMut<ScalarType, R, C>,
    DefaultAllocator: nalgebra::allocator::Allocator<R>,
{
    type Output = Matrix<ScalarType, R, C, S>;

    fn mul(self, mut rhs: Matrix<ScalarType, R, C, S>) -> Self::Output {
        assert_eq!(
            self.ncols(),
            rhs.nrows(),
            "Matrix dimensions incorrect for diagonal matrix multiplication."
        );
        rhs.column_iter_mut()
            .for_each(|mut col| col.component_mul_assign(&self.diagonal));
        rhs
    }
}

#[inline]
/// helper function to turn a matrix into a vector by stacking the columns on top
/// of each other as described here https://en.wikipedia.org/wiki/Vectorization_(mathematics)
pub(crate) fn to_vector<T: Scalar + std::fmt::Debug + Clone>(
    mat: OMatrix<T, Dyn, Dyn>,
) -> DVector<T> {
    let new_rows = Dyn(mat.nrows() * mat.ncols());
    // this shouldn't allocate for valid DMatrix instances.
    mat.reshape_generic(new_rows, U1)
}
