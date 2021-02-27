#[cfg(test)]
mod test;

use nalgebra::{DMatrix, DVector, Scalar, ClosedMul, Dim};
use std::ops::{Mul, MulAssign};

/// A diagonal square matrix with dynamic dimension
/// This internally stores only the diagonal elements
/// # Types
/// ScalarType: the numeric type of the matrix
pub struct DiagDMatrix<ScalarType>
where
    ScalarType: Scalar,
{
    diagonal: DVector<ScalarType>,
}

impl<ScalarType: Scalar> DiagDMatrix<ScalarType> {
    /// get the number of columns of the matrix
    /// The matrix is square, so this is equal to the number of rows
    pub fn ncols(&self) -> usize {
        self.diagonal.len()
    }

    /// get the number of rows of the matrix
    /// The matrix is square, so this is equal to the number of columns
    pub fn nrows(&self) -> usize {
        self.diagonal.len()
    }
}

impl<ScalarType: Scalar> From<DVector<ScalarType>> for DiagDMatrix<ScalarType> {
    fn from(diagonal: DVector<ScalarType>) -> Self {
        Self { diagonal }
    }
}

impl<ScalarType> Mul<&DMatrix<ScalarType>> for &DiagDMatrix<ScalarType>
where
    ScalarType: ClosedMul + Scalar,
{
    type Output = DMatrix<ScalarType>;

    fn mul(self, rhs: &DMatrix<ScalarType>) -> Self::Output {
        assert_eq!(
            self.ncols(),
            rhs.nrows(),
            "Matrix dimensions incorrect for diagonal matrix multiplication."
        );

        let mut result_matrix =
            unsafe { DMatrix::<ScalarType>::new_uninitialized(self.nrows(), rhs.ncols()) };

        for (col_idx, mut col) in result_matrix.column_iter_mut().enumerate() {
            col.copy_from(&self.diagonal.component_mul(&rhs.column(col_idx)));
        }
        result_matrix
    }
}
