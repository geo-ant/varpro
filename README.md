# varpro

![build](https://github.com/geo-ant/varpro/workflows/build/badge.svg?branch=main)
![tests](https://github.com/geo-ant/varpro/workflows/tests/badge.svg?branch=main)
![lints](https://github.com/geo-ant/varpro/workflows/lints/badge.svg?branch=main)
[![coverage](https://codecov.io/gh/geo-ant/varpro/branch/main/graph/badge.svg?token=1L2PEJFMXP)](https://codecov.io/gh/geo-ant/varpro)

This library enables robust and fast least-squares fitting of nonlinear, separable model functions to observations. It uses the VarPro algorithm to achieve this.

## Brief Introduction
The lack of formulas on this site makes it hard to get into the depth of the what and how of this crate at this point. [Refer to the documentation](https://docs.rs/varpro/) for all the meaty details including the math.

### What are Separable Model Functions?
Put simply, separable model functions are nonlinear functions that can be written as a *linear combination* of some *nonlinear* basis functions. A common use case for VarPro is e.g. fitting sums of exponentials, which is a notoriously ill-conditioned problem.

### What is VarPro?
Variable Projection (VarPro) is an algorithm that takes advantage of the fact that the fitting problem can be separated into linear and truly nonlinear parameters. The linear parameters are eliminated and the fitting problem is cast into a problem that only depends on the nonlinear parameters. This reduced problem is then solved by using a common nonlinear fitting algorithm, such as Levenberg-Marquardt (LM). The varpro library uses the [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt) crate as a backend for the LM minimizer. This crate further uses [nalgebra](https://crates.io/crates/nalgebra) for vector and matrix calculations.

### When Should You Give it a Try?
VarPro can dramatically increase the robustness and speed of the fitting process compared to using the nonlinear approach directly. When
* the model function you want to fit is a linear combination of nonlinear functions
* *and* you know the analytical derivatives of all those functions

*then* you should give it a whirl.

## Example
The following example shows how to use varpro to fit a double exponential decay with constant offset to a data vector `y` obtained at grid points `x`. [Refer to the documentation](https://docs.rs/varpro/) for a more in-depth guide.

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
// 1. create the model by giving only the nonlinear parameter names it depends on
let model = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
  .function(&["tau1"], exp_decay)
  .partial_deriv("tau1", exp_decay_dtau)
  .function(&["tau2"], exp_decay)
  .partial_deriv("tau2", exp_decay_dtau)
  .invariant_function(|x|DVector::from_element(x.len(),1.))
  .build()
  .unwrap();
// 2. Cast the fitting problem as a nonlinear least squares minimization problem
let problem = LevMarProblemBuilder::new()
  .model(&model)
  .x(x)
  .y(y)
  .initial_guess(&[1., 2.])
  .build()
  .expect("Building valid problem should not panic");
// 3. Solve the fitting problem
let (solved_problem, report) = LevMarSolver::new().minimize(problem);
assert!(report.termination.was_successful());
// 4. obtain the nonlinear parameters after fitting
let alpha = solved_problem.params();
// 5. obtain the linear parameters
let c = solved_problem.linear_coefficients().unwrap();
```

## References and Further Reading
(O'Leary2013) O’Leary, D.P., Rust, B.W. Variable projection for nonlinear least squares problems. *Comput Optim Appl* **54**, 579–593 (2013). DOI: [10.1007/s10589-012-9492-9](https://doi.org/10.1007/s10589-012-9492-9)

**attention**: the O'Leary paper contains errors that are fixed (so I hope) in [this blog article](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) of mine.

(Golub2003) Golub, G. , Pereyra, V Separable nonlinear least squares: the variable projection method and its applications. Inverse Problems **19** R1 (2003) [https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201](https://iopscience.iop.org/article/10.1088/0266-5611/19/2/201)
