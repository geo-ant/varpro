/// Contains solvers using the [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt)
/// crate.
///
/// This module provides implementations of optimization algorithms for solving the
/// nonlinear least squares problem in variable projection. Currently, it only contains
/// the [`levmar`] module which implements the Levenberg-Marquardt algorithm.
pub mod levmar;

/// linear solver components of the overall nonlinear problems
pub mod linear_solvers;
