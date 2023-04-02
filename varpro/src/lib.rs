#![warn(missing_docs)]
//!
//! # Introduction
//!
//! A large class of nonlinear models consists of a mixture of both truly nonlinear and _linear_ model
//! parameters. These are called *separable models* and can be written as a linear combination
//! of `$N_{basis}$` nonlinear model basis functions. The purpose of this crate is to fit linear
//! models to data using fast and robust algorithms, while providing a simple interface.
//!
//! Consider a data vector `$\vec{y}= (y_1,\dots,y_{N_{data}})^T$`, which
//! is sampled at grid points `$\vec{x}=(x_1,\dots,x_{N_{data}})^T$`, both with `$N_{data}$` elements. Our goal is to fit a nonlinear,
//! separable model `$f$` to the data. Because the model is separable we can write it as
//!
//! ```math
//! \vec{f}(\vec{x},\vec{\alpha},\vec{c}) = \sum_{j=1}^{N_{basis}} c_j \vec{f}_j(\vec{x},S_j(\alpha))
//! ```
//!
//! Lets look at the components of this equation in more detail. The vector valued function
//! `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the actual model we want to fit. It depends on
//! three arguments:
//! * `$\vec{x}$` is the independent variable which corresponds to the grid points. Those can be a time,
//! location or anything at all, really.
//! * `$\vec{\alpha}=(\alpha_1,\dots,\alpha_{N_{params}})^T$` is the vector of nonlinear model parameters.
//! We will get back to these later.
//! * `$\vec{c}=(c_1,\dots,c_{N_{basis}})^T$` is the vector of coefficients for the basis functions.
//!
//! Note that we call `$\vec{\alpha}$` the _parameters_ and `$\vec{c}$` the _coefficients_ of the model.
//! We do this do make the distincion between _nonlinear parameters_ and _linear coefficients_.
//! Of course the model itself is parametrized on both `$\vec{\alpha}$` and `$\vec{c}$`.
//!
//! ## Model Parameters and Basis Functions
//!
//! The model itself is given as a linear combination of *nonlinear* basis functions `$\vec{f}_j$` with
//! expansion coefficient `$c_j$`. The basis functions themselves only depend on the independent variable
//! `$\vec{x}$` and on a subset `$S_j(\alpha)$` of the nonlinear model parameters `$\vec{\alpha}$`.
//! Each basis function can depend on a different subset, but there is no restriction on which
//! parameters a function can depend. Arbitrary functions might share some parameters. It's also
//! fine for functions to depend on some (or only) parameters that are exclusive to them.
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
//! The VarPro algorithm implemented here follows (O'Leary2013), but uses use the Kaufman approximation
//! to calculate the Jacobian.
//!
//! # Usage and Workflow
//!
//! The workflow for solving a least squares fitting problem with varpro is consists of the following steps.
//! 1. Create a [SeparableModel](crate::model::SeparableModel) which describes the model function using
//! the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder). This is done by
//! adding individual basis functions as well as their partial derivatives.
//! 2. Choose a nonlinear minimizer backend. Right now the only implemented nonlinear minimizer
//! is the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) algorithm
//! using the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt/) crate. So just proceed
//! to the next step.
//! 3. Cast the fitting problem into a [LevMarProblem](crate::solvers::levmar::LevMarProblem) using
//! the [LevMarProblemBuilder](crate::solvers::levmar::builder::LevMarProblemBuilder).
//! 4. Solve the fitting problem using the [LevMarSolver](crate::solvers::levmar::LevMarSolver), which
//! is an alias for the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt) struct and allows to set
//! additional parameters of the algorithm before performing the minimization.
//! 5. Check the minimization report and, if successful, retrieve the nonlinear parameters `$\alpha$`
//! using the [LevMarProblem::params](levenberg_marquardt::LeastSquaresProblem::params) and the linear
//! coefficients `$\vec{c}$` using [LevMarProblem::linear_coefficients](crate::solvers::levmar::LevMarProblem::linear_coefficients)
//!
//! # Example
//!
//! ## Preliminaries
//! The following example demonstrates how to fit a double exponential decay model with constant offset
//! to observations `$\vec{y}$` obtained at grid points `$\vec{x}$`. The model itself is a vector valued
//! function with the same number of elements as `$\vec{x}$` and `$\vec{y}$`. The component at index
//! `k` is given by
//! ```math
//! (\vec{f}(\vec{x},\vec{\alpha},\vec{c}))_k= c_1 \exp\left(-x_k/\tau_1\right)+c_2 \exp\left(-x_k/\tau_2\right)+c_3,
//! ```
//! which is just a fancy way of saying that the exponential functions are applied element-wise to the vector `$\vec{x}$`.
//!
//! We can see that the model depends on the nonlinear parameters `$\vec{\alpha}=(\tau_1,\tau_2)^T$`
//! and the linear coefficients `$\vec{c}=(c_1,c_2,c_3)^T$`. Both exponential functions can be modeled
//! as an exponential decay with signature `$\vec{f}_j(\vec{x},\tau)$`. In varpro, the basis functions
//! must take `$\vec{x}$` by reference and a number of parameters as further scalar arguments. So
//! the decay is implemented as:
//!
//! ```rust
//! use nalgebra::DVector;
//! fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//!     x.map(|x|(-x/tau).exp())
//! }
//! ```
//!
//! For our separable model we also need the partial derivatives of the basis functions with
//! respect to all the parameters that each basis function depends on. Thus, in
//! this case we have to provide `$\partial/\partial\tau\,\vec{f}_j(\vec{x},\tau)$`.
//! ```rust
//! use nalgebra::DVector;
//! fn exp_decay_dtau(tvec: &DVector<f64>,tau: f64) -> DVector<f64> {
//!     tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
//! }
//! ```
//!
//! We'll see in the example how the [function](crate::model::builder::SeparableModelBuilder::function) method
//! and the [partial_deriv](crate::model::builder::SeparableModelBuilderProxWithDerivatives::partial_deriv)
//! methods let us add the function and the derivative as base functions.
//!
//! There is a second type of basis function, which corresponds to coefficient `$c_3$`. This is a constant
//! function returning a vector of ones. It does not depend on any parameters, which is why there
//! is a separate way of adding these types of *invariant functions* to the model. For that, use
//! [invariant_function](crate::model::builder::SeparableModelBuilder::invariant_function)
//! of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder).
//!
//! ## Example Code
//! Using the functions above our example code of fitting a linear model to a vector of data `y` obtained
//! at grid points `x` looks like this:
//!
//! ```rust
//! use nalgebra::DVector;
//! use varpro::prelude::*;
//! use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};
//! # fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//! #     x.map(|x|(-x/tau).exp())
//! # }
//! # fn exp_decay_dtau(tvec: &DVector<f64>,tau: f64) -> DVector<f64> {
//! #    tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
//! # }
//! # fn fit_model_example(x:DVector<f64>, y:DVector<f64>) {
//! //1. create the model by giving only the nonlinear parameter names it depends on
//! let model = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
//!     // add the first exponential decay and its partial derivative to the model
//!     // give all parameter names that the function depends on
//!     // and subsequently provide the partial derivative for each parameter
//!     .function(&["tau1"], exp_decay)
//!     .partial_deriv("tau1", exp_decay_dtau)
//!     // add the second exponential decay and its partial derivative to the model
//!     .function(&["tau2"], exp_decay)
//!     .partial_deriv("tau2", exp_decay_dtau)
//!     // add the constant as a vector of ones as an invariant function
//!     .invariant_function(|x|DVector::from_element(x.len(),1.))
//!     // build the model
//!     .build()
//!     .unwrap();
//! // 2.,3: Cast the fitting problem as a nonlinear least squares minimization problem
//! let problem = LevMarProblemBuilder::new(model)
//!     .observations(y)
//!     .build()
//!     .expect("Building valid problem should not panic");
//! // 4. Solve using the fitting problem
//! let (solved_problem, report) = LevMarSolver::new().minimize(problem);
//! assert!(report.termination.was_successful());
//! // the nonlinear parameters after fitting
//! // they are in the same order as the parameter names given to the model
//! let alpha = solved_problem.params();
//! // the linear coefficients after fitting
//! // they are in the same order as the basis functions that were added to the model
//! let c = solved_problem.linear_coefficients().unwrap();
//! # }
//! ```
//!
//! # References and Further Reading
//! (O'Leary2013) O’Leary, D.P., Rust, B.W. Variable projection for nonlinear least squares problems. *Comput Optim Appl* **54**, 579–593 (2013). DOI: [10.1007/s10589-012-9492-9](https://doi.org/10.1007/s10589-012-9492-9)
//!
//! **attention**: the O'Leary paper contains errors that are fixed in [this blog article](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) of mine.
//!
//! (Golub2003) Golub, G. , Pereyra, V Separable nonlinear least squares: the variable projection method and its applications. Inverse Problems **19** R1 (2003) [https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201](https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201)

/// helper implementation to make working with basis functions more seamless
pub mod basis_function;
/// code pertaining to building and working with separable models
pub mod model;
/// commonly useful imports
pub mod prelude;
/// solvers for the nonlinear minimization problem
pub mod solvers;

/// private module that contains helper functionality for linear algebra that is not yet
/// implemented in the nalgebra crate
mod linalg_helpers;

#[cfg(test)]
pub mod test_helpers;
