#[cfg(any(test,doctest))]
mod test;

use nalgebra::{ClosedMul, ComplexField, Scalar, Dim, OVector, Matrix, RawStorageMut, DefaultAllocator};
use std::ops::Mul;

/// A square diagonal matrix with dynamic dimension. Off-diagonal entries are assumed zero.
/// This internally stores only the diagonal elements
/// # Types
/// ScalarType: the numeric type of the matrix
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagMatrix<ScalarType,D>
where
    ScalarType: Scalar + ComplexField,
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, D>
{
    diagonal: OVector<ScalarType,D>,
}

impl<ScalarType,D> DiagMatrix<ScalarType,D> 
    where ScalarType: Scalar + ComplexField,
    D :  Dim,    
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, D>,
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
    pub fn from_real_field(diagonal: OVector<ScalarType::RealField,D>) -> Self 
    where 
    DefaultAllocator: nalgebra::allocator::Allocator<<ScalarType as ComplexField>::RealField, D>,
    {
        Self::from(diagonal.map(ScalarType::from_real))
    }
}
/// Generate a square diagonal matrix from the given diagonal vector.
impl<ScalarType, D> From<OVector<ScalarType,D>> for DiagMatrix<ScalarType,D>
where
 ScalarType: Scalar + ComplexField,
    D : Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, D>
{
    fn from(diagonal: OVector<ScalarType,D>) -> Self {
        Self {
            diagonal,
        }
    }
}

/// Multiply this diagonal matrix from the left to a dynamically sized matrix.
/// # Panics
/// Panics if the dimensions of the matrices do not fit for matrix multiplication
/// # Result
/// The result of the matrix multiplication as a new dynamically sized matrix
impl<ScalarType,R,C,S> Mul<Matrix<ScalarType,R,C,S>> for &DiagMatrix<ScalarType,R>
where
    ScalarType: ClosedMul + Scalar + ComplexField ,
    C : Dim,
    R: Dim,
    S : RawStorageMut<ScalarType, R,C,>,
    DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, R>
{
    type Output = Matrix<ScalarType,R,C,S>;

    fn mul(self, mut rhs: Matrix<ScalarType,R,C,S>) -> Self::Output {
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
