# varpro

![build](https://github.com/geo-ant/varpro/workflows/build/badge.svg?branch=main)
![tests](https://github.com/geo-ant/varpro/workflows/tests/badge.svg?branch=main)
![lints](https://github.com/geo-ant/varpro/workflows/lints/badge.svg?branch=main)
[![Coverage Status](https://coveralls.io/repos/github/geo-ant/varpro/badge.svg?branch=main)](https://coveralls.io/github/geo-ant/varpro?branch=main)
![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)

Nonlinear function fitting made simple. This library provides robust and fast 
least-squares fitting of a wide class of model functions to data.
It uses the VarPro algorithm to achieve this, hence the name.

## Brief Introduction

This crate implements a powerful algorithm
to fit model functions to data, but it is restricted to so called _separable_
models. See the next section for an explanation. The lack of formulas on this 
site makes it hard to get into the depth of the what and how of this crate at this point.
[Refer to the documentation](https://docs.rs/varpro/) for all the meaty details including the math.

### What are Separable Models?

Put simply, separable models are nonlinear functions that can be 
written as a *linear combination* of some *nonlinear* basis functions.
A common use case for VarPro is e.g. fitting sums of exponentials,
which is a notoriously ill-conditioned problem.

### What is VarPro?

Variable Projection (VarPro) is an algorithm that takes advantage of the fact 
that the given fitting problem can be separated into linear and truly nonlinear parameters.
The linear parameters are eliminated using some clever linear algebra
and the fitting problem is cast into a problem that only depends on the nonlinear parameters.
This reduced problem is then solved by using a common nonlinear fitting algorithm,
such as Levenberg-Marquardt (LM).

### When Should You Give it a Try?

VarPro can dramatically increase the robustness and speed of the fitting process
compared to using a "normal" nonlinear least squares fitting algorithm. When

* the model function you want to fit is a linear combination of nonlinear functions
* _and_ you know the analytical derivatives of all those functions

_then_ you should give it a whirl.

## Example Usage

The following example shows how to use varpro to fit a double exponential decay
with constant offset to a data vector `y` obtained at grid points `x`. 
[Refer to the documentation](https://docs.rs/varpro/) for a more in-depth guide.

The exponential decay and it's derivative are given as:

```rust
use nalgebra::DVector;
fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
  x.map(|x|(-x/tau).exp())
}

fn exp_decay_dtau(tvec: &DVector<f64>,tau: f64) -> DVector<f64> {
  tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
}
```

The steps to perform the fitting are:

```rust
use varpro::prelude::*;
use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};

let x = /*time or spatial coordinates of the observations*/;
let y = /*the observed data we want to fit*/;

// 1. create the model by giving only the nonlinear parameter names it depends on
let model = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
  .function(&["tau1"], exp_decay)
  .partial_deriv("tau1", exp_decay_dtau)
  .function(&["tau2"], exp_decay)
  .partial_deriv("tau2", exp_decay_dtau)
  .invariant_function(|x|DVector::from_element(x.len(),1.))
  .independent_variable(x)
  .initial_parameters(initial_params)
  .build()
  .unwrap();
// 2. Cast the fitting problem as a nonlinear least squares minimization problem
let problem = LevMarProblemBuilder::new(model)
  .observations(y)
  .build()
  .unwrap();
// 3. Solve the fitting problem
let fit_result = LevMarSolver::new()
    .fit(problem)
    .expect("fit must exit successfully");
// 4. obtain the nonlinear parameters after fitting
let alpha = fit_result.nonlinear_parameters();
// 5. obtain the linear parameters
let c = fit_result.linear_coefficients().unwrap();
```

For more examples please refer to the crate documentation.

### Fit Statistics

Additionally to the `fit` member function, the `LevMarSolver` also provides a 
`fit_with_statistics` function that calculates some additional statistical
information after the fit has finished.

### Maximum Performance

The example code above will already run many times faster
than just using a nonlinear solver without the magic of varpro.
But this crate offers an additional way to eek out the last bits of  performance.

The `SeparableNonlinearModel` trait can be manually
implemented to describe a model function. This often allows us to shave of the 
last hundreds of microseconds from the computation e.g. by caching intermediate
calculations. The crate documentation contains detailed examples.

## Acknowledgements

I am grateful to Professor [Dianne P. O'Leary](http://www.cs.umd.edu/~oleary/)
and [Bert W. Rust &#10013;](https://math.nist.gov/~BRust/) who published the paper that 
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
