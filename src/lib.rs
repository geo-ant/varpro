#![warn(missing_docs)]
//!
//! # Introduction
//!  
//! `varpro` is a crate for least squares fitting nonlinear models to observations. It works
//! for a large class of so called _separable_ nonlinear least squares problems.
//! It's fast, flexible, and it is simple to use.
//!
//! ## Overview
//!
//! Many nonlinear models consist of a mixture of both truly nonlinear and _linear_ model
//! parameters. These are called *separable models* and can be written as a linear combination
//! of `$N_{basis}$` nonlinear model basis functions.
//!
//! The purpose of this crate is to provide least
//! squares fitting of nonlinear separable models to observations. We also aim to provide excellent usability
//! because all too often it is overly hard to even formulate a fitting
//! problem in code. That is why this libary provides a prototyping API that allows to
//! formulate a fitting problem in a few lines of code using the
//! [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder). For
//! suitable problems this is likely already many times faster than just using nonlinear
//! least squares solvers directly. This is because this crates takes advantage of the Variable Projection
//! (VarPro) algorithm for fitting, which utilizes the separable structure
//! of the model.
//!
//! To shave off the last couple hundreds of microseconds,
//! you can manually implement the [SeparableNonlinearModel](crate::model::SeparableNonlinearModel) trait directly.
//! The test and benchmark suite of this crate should give you a good idea of how fast
//! the fitting can be and how to e.g. take advantage of caching intermediate results.
//!
//! ## The Fitting Problem
//!
//! Consider a data vector of observations `$\vec{y}= (y_1,\dots,y_{N_{data}})^T$`.
//! Our goal is to fit a nonlinear, separable model `$f$` to the data.
//! Because the model is separable we can write it as
//!
//! ```math
//! \vec{f}(\vec{\alpha},\vec{c}) = \sum_{j=1}^{N_{basis}} c_j \vec{f}_j(S_j(\vec{\alpha}))
//! ```
//!
//! Lets look at the components of this equation in more detail. The vector valued function
//! `$\vec{f}(\vec{\alpha},\vec{c}) \in \mathbb{C}^{N_{data}}$` is the model we want to fit. It depends on
//! two vector valued parameters:
//! * `$\vec{\alpha}=(\alpha_1,\dots,\alpha_{N_{params}})^T$` is the vector of nonlinear model parameters.
//! We will get back to these later.
//! * `$\vec{c}=(c_1,\dots,c_{N_{basis}})^T$` is the vector of coefficients for the basis functions.
//! Those are the linear model parameters.
//!
//! Note that we call `$\vec{\alpha}$` the _nonlinear parameters_ and `$\vec{c}$` the _coefficients_ of the model
//! just to make the distinction between the two types of parameters clear. The coefficients are
//! the linear parameters of the model, while `$\vec{\alpha}$` are the nonlinear parameters.
//! Of course the model itself is parametrized on both `$\vec{\alpha}$` and `$\vec{c}$`.
//!
//! Let's look at the individual basis functions `$\vec{f}_j$` in more detail and also
//! why the heck we made the basis functions depend this weird `$S_j(\vec{\alpha})$`.
//!
//! ## Model Parameters and Basis Functions
//!
//! The model itself is given as a linear combination of *nonlinear* basis functions `$\vec{f}_j$` with
//! expansion coefficient `$c_j$`. The basis functions themselves only depend on
//! a subset `$S_j(\vec{\alpha})$` of the nonlinear model parameters `$\vec{\alpha}$`.
//! Each basis function can depend on a different subset, but there is no restriction on which
//! parameters a function can depend on. Arbitrary functions might share none of, some of, or all of the parameters.
//! It's also fine for functions to depend on parameters that are exclusive to them.
//!
//! ## What VarPro Computes
//!
//! This crate finds the parameters `$\vec{\alpha}$` and `$\vec{c}$` that
//! fit the model `$\vec{f}(\vec{\alpha},\vec{c})$` to the observations `$\vec{y}$`
//! using a least squares metric. Formally, varpro finds
//!
//! ```math
//! \arg\min_{\vec{\alpha},\vec{c}} ||\mathbf{W}(\vec{y}-\vec{f}(\vec{\alpha},\vec{c}))||_2^2,
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
//! # Usage and Examples
//!
//! The first step in using this crate is to formulate the fitting problem.
//! This is done by either creating a type that implements the [SeparableNonlinearModel](crate::model::SeparableNonlinearModel) trait
//! or by using the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder) to create a model
//! in a few lines of code. See the examples for the trait and the builder how to
//! use either to generate a separable nonlinear model.
//!
//! The builder is great for getting started quickly and
//! is already much faster than using a nonlinear least squares solver directly.
//! For maximum performance, you can look into implementing the trait manually.
//!
//! ```rust
//! let model = /* generate model here */
//! # ();
//! ```
//!
//! The second step is to use a nonlinear minimization backend to find the parameters that fit the model to the data.
//! Right now the available backend is the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) algorithm
//! using the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt/) crate.
//! Thus, cast the fitting problem into a [LevMarProblem](crate::solvers::levmar::LevMarProblem) using
//! the [LevMarProblemBuilder](crate::solvers::levmar::LevMarProblemBuilder).
//!
//! To build the fitting problem, we need to provide the model, and the observations.
//! The initial guess for the nonlinear parameters `$\vec{\alpha}$` is a property
//! of the model.
//!
//! ```no_run
//! # let model : varpro::model::SeparableModel<f64> = unimplemented!();
//! # let y = nalgebra::DVector::from_vec(vec![0.0, 10.0]);
//! use varpro::solvers::levmar::LevMarProblemBuilder;
//! let problem = LevMarProblemBuilder::new(model)
//!               .observations(y)
//!               .build()
//!               .unwrap();
//! ```
//!
//! Next, solve the fitting problem using the [LevMarSolver](crate::solvers::levmar::LevMarSolver), which
//! is an alias for the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt) struct and allows to set
//! additional parameters of the algorithm before performing the minimization.
//!
//! The simplest way of performing the minimization (without setting any additional
//! parameters for the minimization is like so:
//!
//! ```no_run
//! # use varpro::model::*;
//! # let problem : varpro::solvers::levmar::LevMarProblem<SeparableModel<f64>> = unimplemented!();
//! use varpro::solvers::levmar::LevMarSolver;
//! let (problem, report) = LevMarSolver::new().minimize(problem);
//! ```
//! Finally, check the minimization report and, if successful, retrieve the nonlinear parameters `$\alpha$`
//! using the [LevMarProblem::params](levenberg_marquardt::LeastSquaresProblem::params) and the linear
//! coefficients `$\vec{c}$` using [LevMarProblem::linear_coefficients](crate::solvers::levmar::LevMarProblem::linear_coefficients)
//!
//! ```no_run
//! # use varpro::model::SeparableModel;
//! # use varpro::prelude::*;
//! # let problem : varpro::solvers::levmar::LevMarProblem<SeparableModel<f64>> = unimplemented!();
//! # use varpro::solvers::levmar::LevMarSolver;
//! # let (problem, report) = LevMarSolver::new().minimize(problem);
//! assert!(
//!     report.termination.was_successful(),
//!     "Termination not successful"
//! );
//! let alpha = problem.params();
//! let coeff = problem.linear_coefficients().unwrap();
//! ```
//! If the minimization was successful, the nonlinear parameters `$\vec{\alpha}$`
//! are now stored in the variable `alpha` and the linear coefficients `$\vec{c}$` are stored in `coeff`.
//!
//! # Example
//!
//! ## Preliminaries
//! The following example demonstrates how to fit a double exponential decay model with constant offset
//! to observations `$\vec{y}$` obtained at grid points `$\vec{x}$`. The model itself is a vector valued
//! function with the same number of elements as `$\vec{x}$` and `$\vec{y}$`. The component at index
//! `k` is given by
//!
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
//! and the [partial_deriv](crate::model::builder::SeparableModelBuilderProxyWithDerivatives::partial_deriv)
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
//!
//! # // create the data
//! # let x = DVector::from_vec(vec![0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]);
//! # let y = DVector::from_vec(vec![1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.01]);
//!
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
//!     // the independent variable (x-axis) is the same for all basis functions
//!     .independent_variable(x)
//!     // the initial guess for the nonlinear parameters is tau1=1, tau2=5
//!     .initial_parameters(vec![1.,5.])
//!     // build the model
//!     .build()
//!     .unwrap();
//! // 2.,3: Cast the fitting problem as a nonlinear least squares minimization problem
//! let problem = LevMarProblemBuilder::new(model)
//!     .observations(y)
//!     .build()
//!     .unwrap();
//! // 4. Solve using the fitting problem
//! let (solved_problem, report) = LevMarSolver::new().minimize(problem);
//! assert!(report.termination.was_successful());
//! // the nonlinear parameters after fitting
//! // they are in the same order as the parameter names given to the model
//! let alpha = solved_problem.params();
//! // the linear coefficients after fitting
//! // they are in the same order as the basis functions that were added to the model
//! let c = solved_problem.linear_coefficients().unwrap();
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
/// statistics of the fit
pub mod statistics;

/// helper module containing some commonly useful types
/// implemented in the nalgebra crate
pub mod util;

#[cfg(any(test, doctest))]
pub mod test_helpers;
