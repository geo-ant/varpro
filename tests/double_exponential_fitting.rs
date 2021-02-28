mod common;

use common::evaluate_complete_model;
use common::get_double_exponential_model_with_constant_offset;
use common::linspace;
use nalgebra::{DVector, min};
use varpro::prelude::*;
use varpro::solvers::levmar::*;

use approx::assert_relative_eq;

#[test]
fn double_exponential_fitting_without_noise() {
    let model = get_double_exponential_model_with_constant_offset();
    // the independent variable
    let x = linspace(0., 12.5, 1024);
    // true parameters
    let tau1 = 0.75;
    let tau2 = 3.;
    // true coefficients
    let c1 = 4.;
    let c2 = 2.5;
    let c3 = 1.; //<- coefficient of constant offset
    // generate some data without noise
    let y = evaluate_complete_model(&model, &x, &[tau1, tau2], &DVector::from(vec![c1, c2, c3]));

    let problem = LevMarProblemBuilder::new()
        .model(&model)
        .x(x)
        .y(y)
        .initial_guess(&[2., 5.5])
        .build()
        .expect("Building valid problem should not panic");

    let (problem, report) = LevMarSolver::new().minimize(problem);
    assert!(report.termination.was_successful(), "Termination must be successful");
    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index,tau2_index) = if problem.params()[0] < problem.params()[1] {(0usize,1usize)} else {(1,0)};
    let tau1_calc = problem.params()[tau1_index];
    let tau2_calc = problem.params()[tau2_index];
    let c1_calc = problem.linear_coefficients().expect("linear coeffs must exist")[tau1_index];
    let c2_calc = problem.linear_coefficients().expect("linear coeffs must exist")[tau2_index];
    let c3_calc = problem.linear_coefficients().expect("linear coeffs must exist")[2];

    // assert that the calculated coefficients and nonlinear model parameters are correct
    assert_relative_eq!(c1,c1_calc, epsilon=1e-6);
    assert_relative_eq!(c2,c2_calc, epsilon=1e-6);
    assert_relative_eq!(c3,c3_calc, epsilon=1e-6);
    assert_relative_eq!(tau1,tau1_calc, epsilon=1e-6);
    assert_relative_eq!(tau2,tau2_calc, epsilon=1e-6);
}