#[cfg(test)]
mod test;

use nalgebra::{ClosedMul, ComplexField, DMatrix, DVector, Dyn, Scalar, OMatrix, Dim, UninitMatrix};
use std::ops::Mul;

/// A square diagonal matrix with dynamic dimension. Off-diagonal entries are assumed zero.
/// This internally stores only the diagonal elements
/// # Types
/// ScalarType: the numeric type of the matrix
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagDMatrix<ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    diagonal: DVector<ScalarType>,
}

impl<ScalarType: Scalar + ComplexField> DiagDMatrix<ScalarType> {
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
    pub fn from_real_field(diagonal: &DVector<ScalarType::RealField>) -> Self {
        Self::from(diagonal.map(ScalarType::from_real))
    }
}
/// Generate a square diagonal matrix from the given diagonal vector.
impl<ScalarType: Scalar + ComplexField, VectorType> From<VectorType> for DiagDMatrix<ScalarType>
where
    DVector<ScalarType>: From<VectorType>,
{
    fn from(diagonal: VectorType) -> Self {
        Self {
            diagonal: DVector::from(diagonal),
        }
    }
}

/// Multiply this diagonal matrix from the left to a dynamically sized matrix.
/// # Panics
/// Panics if the dimensions of the matrices do not fit for matrix multiplication
/// # Result
/// The result of the matrix multiplication as a new dynamically sized matrix
impl<ScalarType,C> Mul<&OMatrix<ScalarType,Dyn,C>> for &DiagDMatrix<ScalarType>
where
    ScalarType: ClosedMul + Scalar + ComplexField,
    C : Dim,
{
    type Output = OMatrix<ScalarType,Dyn,C>;

    fn mul(self, rhs: &OMatrix<ScalarType,Dyn,C>) -> Self::Output {
        assert_eq!(
            self.ncols(),
            rhs.nrows(),
            "Matrix dimensions incorrect for diagonal matrix multiplication."
        );

        let rows = Dyn(self.nrows());
        let cols = C::from_usize(rhs.ncols());
        // this isn't an awesome pattern, but it is safe since we fill the Matrix
        // with valid values below. Ideally we would first call the
        // copy_from functionality below, but we can't because the Scalar
        // trait bounds will screw us. See htt
        let mut result_matrix =
            unsafe { UninitMatrix::uninit(rows, cols).assume_init() };

        for (col_idx, mut col) in result_matrix.column_iter_mut().enumerate() {
            col.copy_from(&self.diagonal.component_mul(&rhs.column(col_idx)));
        }

        result_matrix
    }
}
