//! Common imports that are frequently used when working with this crate.
//!
//! This module re-exports the most commonly used types and traits to make
//! it easier to get started with the library.

/// The trait for describing basis functions
pub use crate::basis_function::BasisFunction;
/// The builder for creating separable models
pub use crate::model::builder::SeparableModelBuilder;
/// The trait for describing separable nonlinear models
pub use crate::model::SeparableNonlinearModel;
/// The trait from the levenberg_marquardt crate that problems must implement
pub use levenberg_marquardt::LeastSquaresProblem;
