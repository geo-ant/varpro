use super::*;
use crate::test_helpers::get_double_exponential_model_with_constant_offset;
use levenberg_marquardt::{differentiate_numerically, LevenbergMarquardt};
use approx::assert_relative_eq;
use crate::test_helpers::differentiation::numerical_derivative;

// test that the jacobian of the least squares problem is correct if the parameter guesses
// are correct. I observed that the numerical diff inside the levmar crate and my implementation
// produce significantly different results otherwise. I don't trust the levmar implementation that much,
// so I'll add another test based on my own numerical differentiation. But I'll keep this for a
// sanity check
// **ATTENTION** Also note that the numerical derivative provided by the levenberg-marquart crate
// stalls when one of the initial guesses is 1, because it weill use a step size of one to calculate
// the finite difference and make one of the taus zero, which means 1/tau diverges. I don't know
// exactly why it stalls though. This seems like bad behavior.
#[test]
fn jacobian_of_least_squares_prolem_is_correct_for_correct_parameter_guesses() {
    let model = get_double_exponential_model_with_constant_offset();
    //octave: t = linspace(0,10,11);
    let tvec = DVector::from(vec![0., 1., 2., 3., 4., 5., 6.,7.,8.,9.,10.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1
    let yvec = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227,1.2342,   1.1720,   1.1276,   1.0956
    ]);

    let mut problem = LevMarLeastSquaresProblemBuilder::new()
        .x(tvec)
        .y(yvec)
        .model(&model)
        .initial_guess(vec!{2.,4.})
        .build()
        .expect("Building a valid solver must not return an error.");

    let jacobian_numerical = differentiate_numerically(&mut problem).expect("Numerical differentiation must succeed.");
    let jacobian_calculated = problem.jacobian().expect("Jacobian must not be empty!");
    assert_relative_eq!(jacobian_numerical,jacobian_calculated,epsilon=1e-6);
}


#[test]
// I am implementing my own test that checks if my jacobian and residual calculations are
// correct even for params far away from the true tau1, tau2.
// What I am doing is to numerically differentiate the residual sum of squares using my own
// implementation of a numerical derivative of one parameter.
// Then I am checking the result with the calculated residuals and jacobian.
// I am using the formula for the partial derivatives of the residual sum of squares from
// [my post](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/) on varpro
// (found between numbered formulas 8 and 9).
fn jacobian_produces_correct_results_for_differentiating_the_residual_sum_of_squares() {
    let model = get_double_exponential_model_with_constant_offset();
    //octave: t = linspace(0,10,11);
    let tvec = DVector::from(vec![0., 1., 2., 3., 4., 5., 6.,7.,8.,9.,10.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1
    let yvec = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227,1.2342,   1.1720,   1.1276,   1.0956
    ]);

    let mut problem = LevMarLeastSquaresProblemBuilder::new()
        .x(tvec)
        .y(yvec)
        .model(&model)
        .initial_guess(vec!{1.,2.}) // these initial params don't for this test
        .build()
        .expect("Building a valid solver must not return an error.");

    let fixed_tau1 = 0.5;
    let fixed_tau2 = 7.5;

    // the numerical derivatives of the residual sum of squares
    let numerical_deriv_tau1 =  numerical_derivative(|tau1 : f64| {
        problem.set_params(&DVector::<f64>::from(vec!{tau1,fixed_tau2}));
        problem.residuals().unwrap().norm_squared()}, fixed_tau1);
    let numerical_deriv_tau2 = numerical_derivative(|tau2 : f64| {
        problem.set_params(&DVector::<f64>::from(vec!{fixed_tau1,tau2}));
        problem.residuals().unwrap().norm_squared()}, fixed_tau2);

    // calculate the partial derivatives based on the calculated jacobian and residualss
    problem.set_params(&DVector::<f64>::from(vec!{fixed_tau1,fixed_tau2}));
    let jacobian = problem.jacobian().expect("Jacobian must produce a result");
    let residuals = problem.residuals().expect("Residuals must produce a result");
    let calculated_derivative_tau1 :f64 = 2.*residuals.dot(&jacobian.column(0));
    let calculated_derivative_tau2 :f64 = 2.*residuals.dot(&jacobian.column(1));

    // check that the partial derivatives of the squared residual are sufficiently close
    assert_relative_eq!(numerical_deriv_tau1,calculated_derivative_tau1,epsilon=1e-6);
    assert_relative_eq!(numerical_deriv_tau2,calculated_derivative_tau2,epsilon=1e-6);
}
