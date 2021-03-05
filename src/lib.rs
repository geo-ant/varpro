//! The varpro crate enables nonlinear least squares fitting for separable models using the Variable Projection (VarPro) algorithm.
//!
//! # Introduction
//! A large class of nonlinear models consists of a mixture of truly nonlinear as well as linear model
//! parameters. These are the so called *separable models* which can be written as a linear combination
//! of `$N_{basis}$` nonlinear model basis functions. The purpose of this crate provide a simple interface to
//! robust and fast routines to fit separable models to data. Consider a data vector `$\vec{y}= (y_1,\dots,y_{N_{data}})^T$`, which
//! is sampled at grid points `$\vec{x}=(x_1,\dots,x_{N_{data}})^T$`, both with `$N_{data}$` elements. Our goal is to fit a nonlinear,
//! separable model to the data. Because the model is separable we can write it as
//!
//! ```math
//! \vec{f}(\vec{x},\vec{\alpha},\vec{c}) = \sum_{j=1}^{N_{basis}} c_j \vec{f}_j(\vec{x},S_j(\alpha))
//! ```
//! Lets look at the components of this equation in more detail. The vector valued function
//! `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the actual model we want to fit. It depends on
//! three arguments:
//! * `$\vec{x}$` is the independent variable which corresponds to the grid points. It can be a time,
//! location or anything else.
//! * `$\vec{\alpha}=(\alpha_1,\dots,\alpha_{N_{params}})^T$` is the vector of nonlinear model parameters.
//! We will get back to these later.
//! * `$\vec{c}=(c_1,\dots,c_{N_{basis}})^T$` is the vector of coefficients for the basis functions.
//!
//! ## Model Parameters and Basis Functions
//!
//! The model itself is given as a linear combination of *nonlinear* basis functions `$\vec{f}_j$` with
//! expansion coefficient `$c_j$`. The basis functions themselves only depend on the independent variable
//! `$\vec{x}$` and on a subset `$S_j(\alpha)$` of the nonlinear model parameters `$\vec{\alpha}$`.
//! Each basis function can depend on a different subset.
//!
//! ## What VarPro Computes
//! This crate finds the parameters `$\vec{\alpha}$` and `$\vec{c}$` that
//! fit the model `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` to the data `$\vec{y}$` obtained at grid points
//! `$\vec{x}$` using a least squares metric. Formally, varpro finds
//!
//! ```math
//! \arg\min_{\vec{\alpha},\vec{c}} ||\mathbf{W}(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}))||_2^2,
//! ```
//! where `$\mathbf{W}$` is a weight matrix that can be set to the identity matrix for unweighted
//! least squares.
//!
//! ## What Makes VarPro Special
//! The Variable Projection algorithm takes advantage of the fact that the model is a mix of linear
//! and nonlinear parts. VarPro uses a [clever algorithmic trick](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
//! to cast the minimization into a problem that only depends on the nonlinear model parameters
//! `$\vec{\alpha}$` and lets a nonlinear optimization backend handle this reduced problem. This crate
//! uses the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt/) crate as it's optimization backend. Other
//! optimization backends are planned for future releases.
//!
//! The VarPro algorithm implemented here follows (O'Leary2013), but does use the Kaufmann approximation
//! to calculate the Jacobian.
//!
//! # Usage and Workflow
//!
//! TODO DOCUMENT WORKFLOW
//!
//! # Example
//!
//! TODO GIVE EXAMPLE
//!
//! # References and Further Reading
//! (O'Leary2013) O’Leary, D.P., Rust, B.W. Variable projection for nonlinear least squares problems. *Comput Optim Appl* **54**, 579–593 (2013). DOI: [10.1007/s10589-012-9492-9](https://doi.org/10.1007/s10589-012-9492-9)
//! (attention: the paper contains errors that are fixed in [this blog article](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) of mine.
//! (Golub2003) Golub, G. , Pereyra, V Separable nonlinear least squares: the variable projection method and its applications. Inverse Problems **19** R1 (2003) [https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201](https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201)
pub mod basis_function;
pub mod model;
pub mod prelude;
pub mod solvers;

/// private module that contains helper functionality for linear algebra that is not yet
/// implemented in the nalgebra crate
mod linalg_helpers;

#[cfg(test)]
pub mod test_helpers;
