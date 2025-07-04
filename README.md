# varpro

![build](https://github.com/geo-ant/varpro/actions/workflows/build.yml/badge.svg?branch=main)
![tests](https://github.com/geo-ant/varpro/actions/workflows/tests.yml/badge.svg?branch=main)
![lints](https://github.com/geo-ant/varpro/actions/workflows/lints.yml/badge.svg?branch=main)
[![crates](https://img.shields.io/crates/v/varpro)](https://crates.io/crates/varpro)
[![Coverage Status](https://coveralls.io/repos/github/geo-ant/varpro/badge.svg?branch=main)](https://coveralls.io/github/geo-ant/varpro?branch=main)
![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![crates](https://raw.githubusercontent.com/geo-ant/user-content/refs/heads/main/ko-fi-support.svg)](https://ko-fi.com/geoant)

**Nonlinear function fitting made easy.**

## Introduction

This crate uses a powerful algorithm for fitting so-called _separable_ model
functions to data. It strives to provide both an easy to use interface with
great out-of-the-box performance, as well as an advanced interface for maximum
performance.

The lack of formulas on this site makes it hard to dive deep, but a brief
overview is provided in the next sections. [Refer to the documentation](https://docs.rs/varpro/)
for all the meaty details, including the math.

### What are Separable Models?

Put simply, separable models are nonlinear functions which can be 
written as a *linear combination* of some *nonlinear* basis functions.
A common use case for `varpro` is e.g. fitting sums of exponentials,
which is a notoriously ill-conditioned problem.

### What is VarPro?

Variable Projection (VarPro) is an algorithm that takes advantage of the separable
nature of the model to fit. Using clever math, the fitting problem is rewritten so that
it depends only on the nonlinear parameters of the separable model. This reduced problem
is solved by using a nonlinear minimization algorithm, such as Levenberg-Marquardt.

### When Should You Give it a Try?

VarPro can dramatically increase the robustness and speed of the fitting process
compared to using a general purpose nonlinear least squares fitting algorithm. When

* the model function you want to fit is a linear combination of nonlinear functions,
* _and_ you know the analytical derivatives of all those functions

_then_ you should give it a whirl. Also consider the section on global fitting below,
which provides another great use case for this crate.

## Example Usage

The following example shows, how to use this crate to fit a double exponential decay
with constant offset to a data vector `y` obtained at time points `t`. 
[Refer to the documentation](https://docs.rs/varpro/) for a more in-depth guide.

```rust
use varpro::prelude::*;
use varpro::solvers::levmar::LevMarSolver;
use varpro::problem::SeparableProblemBuilder;
use nalgebra::{dvector,DVector};

// Define the exponential decay e^(-t/tau).
// Both of the nonlinear basis functions in this example
// are exponential decays.
fn exp_decay(t :&DVector<f64>, tau : f64) 
  -> DVector<f64> {
  t.map(|t|(-t/tau).exp())
}

// the partial derivative of the exponential
// decay with respect to the nonlinear parameter tau.
// d/dtau e^(-t/tau) = e^(-t/tau)*t/tau^2
fn exp_decay_dtau(t: &DVector<f64>,tau: f64) 
  -> DVector<f64> {
  t.map(|t| (-t / tau)
    .exp() * t / tau.powi(2))
}

// temporal (or spatial) coordintates of the observations
let t = dvector![0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.];
// the observations we want to fit
let y = dvector![6.0,4.8,4.0,3.3,2.8,2.5,2.2,1.9,1.7,1.6,1.5];

// 1. create the model by giving only the nonlinear parameter names it depends on
let model = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
  // provide the nonlinear basis functions and their derivatives.
  // In general, base functions can depend on more than just one parameter.
  // first function:
  .function(&["tau1"], exp_decay)
  .partial_deriv("tau1", exp_decay_dtau)
  // second function and derivatives with respect to all parameters
  // that it depends on (just one in this case)
  .function(&["tau2"], exp_decay)
  .partial_deriv("tau2", exp_decay_dtau)
  // a constant offset is added as an invariant basefunction
  // as a vector of ones. It is multiplied with its own linear coefficient,
  // creating a fittable offset
  .invariant_function(|v|DVector::from_element(v.len(),1.))
  // give the coordinates of the problem
  .independent_variable(t)
  // provide guesses only for the nonlinear parameters in the
  // order that they were given on construction.
  .initial_parameters(vec![2.5,5.5])
  .build()
  .unwrap();
// 2. Cast the fitting problem as a nonlinear least squares minimization problem
let problem = SeparableProblemBuilder::new(model)
  .observations(y)
  .build()
  .unwrap();
// 3. Solve the fitting problem
let fit_result = LevMarSolver::default()
    .fit(problem)
    .expect("fit must exit successfully");
// 4. obtain the nonlinear parameters after fitting
let alpha = fit_result.nonlinear_parameters();
// 5. obtain the linear parameters
let c = fit_result.linear_coefficients().unwrap();
```

For more in-depth examples, please refer to the crate documentation.

### Fit Statistics

Additionally to the `fit` member function,
the `LevMarSolver` provides a `fit_with_statistics` function that calculates
an extra bit of useful statistical information.

### Global Fitting of Multiple Right Hand Sides

In the example above, we have passed a single column vector as the observations.
The library also allows fitting multiple right hand sides, by constructing a
problem via `SeparableProblem::mrhs`. When fitting multiple right hand sides,
`varpro` will performa a _global fit_, in which the nonlinear parameters are optimized
across all right hand sides, while linear coefficients of the fit are optimized for
each right hand side individually.

This is another application where varpro really shines, since it can take advantage
of the separable nature of the problem. It allows us to perform a global fit over thousands,
even tens of thousands of right hand sides in reasonable time (fractions of seconds to minutes),
where conventional nonlinear solvers must perform much more work.

### Maximum Performance and Advanced Use Cases

The example code above will already run many times faster
than just using a nonlinear solver without the magic of varpro.
But this crate offers an additional way to eek out the last bits of performance.

The `SeparableNonlinearModel` trait can be manually implemented to describe a
model function. This often allows us to shave off the last hundreds of microseconds
from the computation, e.g. by caching intermediate calculations. The crate documentation
contains detailed examples.

This is not only useful for performance, but also for use cases that are difficult
or impossible to accomodate using only the `SeparableModelBuilder`. The builder
was created for ease of use _and_ performance, but it has some limitations by design.

## Minimum Supported Rust Version (MSRV)

The minimum supported Rust version is `1.75.0`. Note, that running the tests and
benchmarks might require a more recent version of the compiler. Testing and
benchmarking is always performed with the most recent `stable` version of Rust at
the time of publishing.

## Acknowledgements

I am grateful to Professor [Dianne P. O'Leary](http://www.cs.umd.edu/~oleary/)
and [Bert W. Rust &#10013;](https://math.nist.gov/~BRust/), who published the paper that 
enabled me to understand varpro and come up with this implementation.
Professor O'Leary also graciously answered my questions on her paper and
some implementation details.

## References and Further Reading

(O'Leary2013) O’Leary, D.P., Rust, B.W. Variable projection for nonlinear least squares problems.
*Comput Optim Appl* **54**, 579–593 (2013). DOI: [10.1007/s10589-012-9492-9](https://doi.org/10.1007/s10589-012-9492-9)

**attention**: the O'Leary paper contains errors that are fixed (so I hope)
in [this blog article](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) of mine.

(Golub2003) Golub, G. , Pereyra, V Separable nonlinear least squares:
the variable projection method and its applications. Inverse Problems **19** R1 (2003)
[https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201](https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201)
