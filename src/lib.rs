#![warn(missing_docs)]
//!
//! # Introduction
//!  
//! `varpro` is a crate for least squares fitting nonlinear models to observations. It works
//! for a large class of so called _separable_ nonlinear least squares problems.
//! It's fast, flexible, and it is easy to use.
//!
//! ## Multiple Right Hand Sides
//!
//! Since version 0.8.0, support for _global fitting_ with multiple right hand
//! sides has been added to this library. This is a powerful technique for suitable
//! problems and is explained at the end of this introduction.
//!
//! ## Overview
//!
//! Many nonlinear models consist of a mixture of both truly nonlinear and _linear_ model
//! parameters. These are called *separable models* and can be written as a linear combination
//! of `$N_{basis}$` nonlinear model basis functions.
//!
//! The purpose of this crate is to provide least squares fitting of nonlinear
//! separable models to observations. We also aim to provide excellent usability
//! because all too often it hard to even formulate a fitting
//! problem in code. Thus, this libary provides an API that allows to
//! formulate a separable model in a few lines of code using the
//! [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder). For
//! separable problems, this is likely already many times faster than just using nonlinear
//! least squares solvers directly. This is because this crates takes advantage
//! of the Variable Projection (VarPro) algorithm for fitting, which makes use
//! of the separable structure of the model.
//!
//! To shave off the last couple hundreds of microseconds, you can manually implement
//! the [SeparableNonlinearModel](crate::model::SeparableNonlinearModel) trait directly
//! to describe your model function.
//!
//! The test and benchmark suite of this crate should give you a good idea of how fast
//! the fitting can be, and how to e.g. take advantage of caching intermediate results.
//!
//! ## The Fitting Problem
//!
//! Consider a data vector of observations `$\vec{y}= (y_1,\dots,y_{N_{data}})^T$`.
//! Our goal is to fit a nonlinear, separable model `$f$` to the data.
//! Because the model is separable, we can write it as
//!
//! ```math
//! \vec{f}(\vec{\alpha},\vec{c}) = \sum_{j=1}^{N_{basis}} c_j \vec{f}_j(S_j(\vec{\alpha}))
//! ```
//!
//! Lets look at the components of this equation in more detail. The vector valued function
//! `$\vec{f}(\vec{\alpha},\vec{c}) \in \mathbb{C}^{N_{data}}$` is the model we want to fit. It depends on
//! two vector valued parameters:
//!
//! * `$\vec{\alpha}=(\alpha_1,\dots,\alpha_{N_{params}})^T$` is the vector of nonlinear model parameters.
//!   We will get back to these later.
//! * `$\vec{c}=(c_1,\dots,c_{N_{basis}})^T$` is the vector of coefficients for the basis functions.
//!   Those are the linear model parameters.
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
//! linear coefficients `$c_j$`. The basis functions themselves only depend on
//! a subset `$S_j(\vec{\alpha})$` of the nonlinear model parameters `$\vec{\alpha}$`.
//! Each basis function can depend on a different subset, but there is no restriction on which
//! parameters a function can depend on. Arbitrary functions might share none of, some of,
//! or all of the nonlinear parameters. It's also fine for functions to depend on
//! parameters that are exclusive to them.
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
//! where `$\mathbf{W}$` is a weight matrix. It can be set to the identity matrix for
//! _unweighted_ least squares.
//!
//! ## What Makes VarPro Special
//!
//! The Variable Projection algorithm takes advantage of the fact that the model is a mix of linear
//! and nonlinear parts. VarPro uses a [clever algorithmic trick](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
//! to cast the minimization into a problem that only depends on the nonlinear model parameters
//! `$\vec{\alpha}$`. It then lets a nonlinear optimization backend handle this reduced problem.
//! Currently, this crate uses the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt/)
//! crate as it's optimization backend. Other optimization backends are planned
//! for future releases.
//!
//! The VarPro algorithm implemented here follows (O'Leary2013), but uses use the Kaufman approximation
//! to calculate the Jacobian.
//!
//! # Usage and Examples
//!
//! Using this crate is a step-by-step process:
//!
//! 1. Describe the separable model.
//! 2. Describe the fitting problem.
//! 3. Fit the problem with a model using a solver.
//!
//! The first step in using this crate is to formulate the separable model.
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
//! The second step is to describe the fitting problem by creating a
//! [`SeparableProblem`](crate::problem::SeparableProblem) using the
//! [`SeparableProblemBuilder`](crate::problem::SeparableProblemBuilder).
//! To build the fitting problem, we need to provide the model, and the observations.
//! The initial guess for the nonlinear parameters `$\vec{\alpha}$` is a property
//! of the model. We can additionally configure some other properties to influence the fit.
//!
//! ```no_run
//! # let model : varpro::model::SeparableModel<f64> = unimplemented!();
//! # let y = nalgebra::DVector::from_vec(vec![0.0, 10.0]);
//! use varpro::problem::*;
//! let problem = SeparableProblemBuilder::new(model)
//!               .observations(y)
//!               .build()
//!               .unwrap();
//! ```
//!
//! The third step is to use a solver, to fit the model to the problem.
//! Currently, the only the available solver is the
//! [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) algorithm
//! using the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt/) crate.
//! It is provided via the [`LevMarSolver`](crate::solver::levmar::LevMarSolver)
//! type, which allows to make additional configurations to the solver.
//!
//! ```no_run
//! # use varpro::model::*;
//! # use varpro::problem::*;
//! # let problem : varpro::problem::SeparableProblem<SeparableModel<f64>, SingleRhs> = unimplemented!();
//! use varpro::solvers::levmar::LevMarSolver;
//! let fit_result = LevMarSolver::default().fit(problem).unwrap();
//! ```
//!
//! If successful, retrieve the nonlinear parameters `$\alpha$` using the
//! [SeparableProblem::params](levenberg_marquardt::LeastSquaresProblem::params) and the linear
//! coefficients `$\vec{c}$` using [SeparableProblem::linear_coefficients](crate::problem::SeparableProblem::linear_coefficients)
//!
//! **Fit Statistics:** To get additional statistical information after the fit
//! has finished, use the [LevMarSolver::fit_with_statistics](crate::solvers::levmar::LevMarSolver::fit_with_statistics)
//! method.
//!
//! ```no_run
//! # use varpro::model::SeparableModel;
//! # use varpro::prelude::*;
//! # use varpro::problem::*;
//! # let problem : varpro::problem::SeparableProblem<SeparableModel<f64>, SingleRhs> = unimplemented!();
//! # use varpro::solvers::levmar::LevMarSolver;
//! # let fit_result = LevMarSolver::default().fit(problem).unwrap();
//! let alpha = fit_result.nonlinear_parameters();
//! let coeff = fit_result.linear_coefficients().unwrap();
//! ```
//! If the minimization was successful, the nonlinear parameters `$\vec{\alpha}$`
//! are now stored in the variable `alpha` and the linear coefficients `$\vec{c}$` are stored in `coeff`.
//!
//! # Example: Double Exponential Fitting
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
//! and the [partial_deriv](crate::model::builder::SeparableModelBuilder::partial_deriv)
//! methods let us add the function and the derivative as base functions.
//!
//! There is a second type of basis function, which corresponds to coefficient `$c_3$`. This is a constant
//! function returning a vector of ones. It does not depend on any parameters, which is why there
//! is a separate way of adding these types of *invariant functions* to the model. For that, use
//! [invariant_function](crate::model::builder::SeparableModelBuilder::invariant_function)
//! of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder).
//!
//! ## Example Code
//!
//! Using the functions above our example code of fitting a linear model to a vector of data `y` obtained
//! at grid points `x` looks like this:
//!
//! ```rust
//! use nalgebra::DVector;
//! use varpro::prelude::*;
//! use varpro::solvers::levmar::LevMarSolver;
//! use varpro::problem::*;
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
//! // 1. create the model by giving only the nonlinear parameter names it depends on
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
//!
//! // 2. Cast the fitting problem as a nonlinear least squares minimization problem
//! let problem = SeparableProblemBuilder::new(model)
//!     .observations(y)
//!     .build()
//!     .unwrap();
//!
//! // 3. Solve using the fitting problem
//! let fit_result = LevMarSolver::default()
//!     .fit(problem)
//!     .expect("fit must succeed");
//! // the nonlinear parameters after fitting
//! // they are in the same order as the parameter names given to the model
//! let alpha = fit_result.nonlinear_parameters();
//! // the linear coefficients after fitting
//! // they are in the same order as the basis functions that were added to the model
//! let c = fit_result.linear_coefficients().unwrap();
//! ```
//!
//! # Example 2: Mixed exponential and trigonometric model
//!
//! This example is taken from the matlab code that is published as part of the
//! O'Leary 2013 paper and fits a mixed exponential and trigonometric model to
//! some noisy and weighted data.
//!
//! In keeping with the element-wise notation above, we can write the model
//! as
//!
//! ```math
//! (\vec{f}(\vec{x},\vec{\alpha},\vec{c}))_k = c_1 \exp(-\alpha_2 x_k)\cdot\cos(\alpha_3 x_k)
//!     + c_2 \exp(-\alpha_1 x_k)\cdot\cos(\alpha_2 x_k).
//! ```
//!
//! The code to fit this model to some data is given below. Note also that
//! weights are given for the data points.
//!
//! ```rust
//! use nalgebra::DVector;
//! use varpro::prelude::*;
//! use varpro::solvers::levmar::LevMarSolver;
//! use varpro::problem::*;
//!
//! // build the model
//! fn phi1(t: &DVector<f64>, alpha2: f64, alpha3: f64) -> DVector<f64> {
//!     t.map(|t| f64::exp(-alpha2 * t) * f64::cos(alpha3 * t))
//! }
//! fn phi2(t: &DVector<f64>, alpha1: f64, alpha2: f64) -> DVector<f64> {
//!     t.map(|t| f64::exp(-alpha1 * t) * f64::cos(alpha2 * t))
//! }
//! // the data, weight and initial guesses for our fitting problem
//!
//! let t = DVector::from_vec(vec![
//!     0., 0.1, 0.22, 0.31, 0.46, 0.50, 0.63, 0.78, 0.85, 0.97,
//! ]);
//! let y = DVector::from_vec(vec![
//!     6.9842, 5.1851, 2.8907, 1.4199, -0.2473, -0.5243, -1.0156, -1.0260, -0.9165, -0.6805,
//! ]);
//! let w = DVector::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5]);
//! let initial_guess = vec![0.5, 2., 3.];
//!
//! let model = SeparableModelBuilder::new(["alpha1", "alpha2", "alpha3"])
//!     .initial_parameters(initial_guess)
//!     .independent_variable(t)
//!     // phi 1
//!     .function(["alpha2", "alpha3"], phi1)
//!     .partial_deriv("alpha2", |t: &DVector<f64>, alpha2: f64, alpha3: f64| {
//!         phi1(t, alpha2, alpha3).component_mul(&(-1. * t))
//!     })
//!     .partial_deriv("alpha3", |t: &DVector<f64>, alpha2: f64, alpha3: f64| {
//!         t.map(|t| -t * (-alpha2 * t).exp() * (alpha3 * t).sin())
//!     })
//!     .function(["alpha1", "alpha2"], phi2)
//!     .partial_deriv("alpha1", |t: &DVector<f64>, alpha1: f64, alpha2: f64| {
//!         phi2(t, alpha1, alpha2).component_mul(&(-1. * t))
//!     })
//!     .partial_deriv("alpha2", |t: &DVector<f64>, alpha1: f64, alpha2: f64| {
//!         t.map(|t| -t * (-alpha1 * t).exp() * (alpha2 * t).sin())
//!     })
//!     .build()
//!     .unwrap();
//!
//! // describe the fitting problem
//! let problem = SeparableProblemBuilder::new(model)
//!     .observations(y)
//!     .weights(w)
//!     .build()
//!     .unwrap();
//!
//! // fit the data
//! let fit_result = LevMarSolver::default()
//!                 .fit(problem)
//!                 .expect("fit must succeed");
//! // the nonlinear parameters
//! let alpha = fit_result.nonlinear_parameters();
//! // the linear coefficients
//! let c  = fit_result.linear_coefficients().unwrap();
//! ```
//! # Global Fitting with Multiple Right Hand Sides
//!
//! Instead of fitting a single data vector (i.e. a single _right hand side_),
//! this library can also solve a related, but slightly different problem. This
//! is the problem of global fitting for _multiple right hand sides_. The problem
//! statement is the following:
//!
//!  * We have not only one observation but a set `$\{\vec{y}_s\}$`, `$s=1,...,S$` of
//!    observations.
//!  * We want to fit the separable nonlinear function `$\vec{f}(\vec{\alpha},\vec{c})$`
//!    to all vectors of observations, but in such a way that the linear parameters
//!    are allowed to vary with `$s$`, but the nonlinear parameters
//!    are the same for the whole dataset.
//!
//! This is called global fitting with multiple right hand sides,
//! because the nonlinear parameters are not
//! allowed to change and are optimized for the complete dataset, whereas the linear
//! parameters are allowed to vary with each vector of observations. This is an application
//! where varpro really shines. Note that it is not the same as fitting the data
//! vectors independently. For a more in depth explanation
//! [see my article here](https://geo-ant.github.io/blog/2024/variable-projection-part-2-multiple-right-hand-sides/).
//!
//! To take advantage of global fitting we don't need to change anything about the
//! model, we just have to make a slight modification to the way we build a problem.
//! The crucial differences to the single right hand side case are:
//!
//! 1. We have to use the [`SeparableProblemBuilder::mrhs`](crate::problem::SeparableProblemBuilder::mrhs)
//!    constructor rather than `new`.
//! 2. We have to sort the right hand sides into a matrix, where each right
//!    hand side, which is a column-vector on its own, will become a column
//!    of the resulting matrix.
//!
//! For a set of observations `$\vec{y}_1,\dots,\vec{y}_S$` (column vectors) we
//! now have to pass a _matrix_ `$Y$` of observations, rather than a single
//! vector to the builder. As explained above, the resulting matrix would look
//! like this.
//!
//! ```math
//! \boldsymbol{Y}=\left(\begin{matrix}
//!  \vert &  & \vert \\
//!  \vec{y}_1 &  \dots & \vec{y}_S \\
//!  \vert &  & \vert \\
//! \end{matrix}\right)
//! ```
//!
//! The order of the vectors in the matrix doesn't matter, but it will determine
//! the order of the linear coefficients, see
//! [`SeparableProblemBuilder::mrhs`](crate::problem::SeparableProblem::mrhs)
//! for a more detailed explanation.
//!
//! ## Example
//!
//! ```no_run
//! # use varpro::prelude::*;
//! # use varpro::solvers::levmar::LevMarSolver;
//! # use varpro::problem::*;
//! # let model : varpro::model::SeparableModel::<f64> = unimplemented!();
//! # let Y : nalgebra::DMatrix::<f64> = unimplemented!();
//! # let w : nalgebra::DVector::<f64> = unimplemented!();
//!
//! // use the model as before but now invoke the `mrhs`
//! // constructor for the fitting problem
//! let problem = SeparableProblemBuilder::mrhs(model)
//! // the observations is a matrix where each column vector represents
//! // a single observation
//!     .observations(Y)
//!     .weights(w)
//!     .build()
//!     .unwrap();
//!
//! // fit the data
//! let fit_result = LevMarSolver::default()
//!                 .fit(problem)
//!                 .expect("fit must succeed");
//!
//! // the nonlinear parameters
//! // these parameters are fitted globally, meaning they
//! // are the same for each observation. Hence alpha is a single
//! // vector of parameters.
//! let alpha = fit_result.nonlinear_parameters();
//!
//! // the linear coefficients
//! // those are a matrix for global fitting with multiple right hand sides.
//! // Each colum corresponds to the linear coefficients for the same column
//! // in the matrix of observations Y.
//! # #[allow(non_snake_case)]
//! let C  = fit_result.linear_coefficients().unwrap();
//! ```
//!
//! The main difference to fitting problems with a single right hand side is that
//! the observations now become a matrix. Each column of this matrix is an
//! observation. Since the linear coefficients are allowed to vary, they now
//! also become a matrix instead of a single vector. Each column corresponds to
//! the best fit linear coefficients of the observations in the same matrix column.
//!
//! # References and Further Reading
//! (O'Leary2013) O’Leary, D.P., Rust, B.W. Variable projection for nonlinear least squares problems. *Comput Optim Appl* **54**, 579–593 (2013). DOI: [10.1007/s10589-012-9492-9](https://doi.org/10.1007/s10589-012-9492-9)
//!
//! **attention**: the O'Leary paper contains errors that are fixed in [this blog article](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) of mine.
//!
//! (Golub2003) Golub, G. , Pereyra, V Separable nonlinear least squares: the variable projection method and its applications. Inverse Problems **19** R1 (2003) [https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201](https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201)

/// helper implementation to make working with basis functions more seamless
pub mod basis_function;
/// defining separable models
pub mod model;
/// commonly useful imports
pub mod prelude;
/// defining the separable fitting problem
pub mod problem;
/// solvers for the nonlinear minimization problem
pub mod solvers;
/// statistics of the fit
pub mod statistics;
/// helper module containing some commonly useful types
/// implemented in the nalgebra crate
pub mod util;

/// a helper module for doctesting code in my readme
#[cfg(doctest)]
mod readme;

#[cfg(any(test, doctest))]
pub mod test_helpers;
