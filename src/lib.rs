//! TODO Document this crate
//! let's try some math
//! ```math
//!   f(\vec{x},\vec{\alpha}) = \sum_j f_j(\vec{x},\vec{\alpha}_j)
//! ```
//! Where `$\vec{\alpha}_j$` is a subset of the model parameters `$\vec{\alpha}$`.
//!
//! let's also try to link some other code [SeparableModelBuilder] but maybe I got to look
//! at the rustdoc documentation again

pub mod basis_function;
pub mod model;
pub mod prelude;
