use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, Scalar};

pub trait LinearSolver<T, R, C>:
    TryFrom<OMatrix<T, R, C>, Error = <Self as LinearSolver<T, R, C>>::Error>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    type Error: std::error::Error;
}
