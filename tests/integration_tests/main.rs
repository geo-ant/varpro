use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;

use nalgebra::OVector;
use nalgebra::Vector2;
use nalgebra::Vector3;
use nalgebra::U1;
use nalgebra::U2;
use nalgebra::U3;
use shared_test_code::evaluate_complete_model_at_params;
use shared_test_code::get_double_exponential_model_with_constant_offset;
use shared_test_code::linspace;
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::models::DoubleExponentialDecayFittingWithOffsetLevmar;
use shared_test_code::models::OLearyExampleModel;
use varpro::prelude::*;
use varpro::solvers::levmar::*;

use approx::assert_relative_eq;

#[test]
// sanity check my calculations above
fn sanity_check_jacobian_of_levenberg_marquardt_problem_is_correct() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 2.;
    let tau2 = 4.;
    let c1 = 2.;
    let c2 = 4.;
    let c3 = 0.2;
    let f = x.map(|x: f64| c1 * (-x / tau1).exp() + c2 * (-x / tau2).exp() + c3);

    let mut problem = DoubleExponentialDecayFittingWithOffsetLevmar::new(
        &[0.5 * tau1, 1.5 * tau2, 2. * c1, 3. * c2, 3. * c3],
        &x,
        &f,
    );

    // Let `problem` be an instance of `LeastSquaresProblem`
    let jacobian_numerical = levenberg_marquardt::differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-5);
}

#[test]
fn double_exponential_fitting_without_noise_produces_accurate_results() {
    // the independent variable
    let x = linspace(0., 12.5, 1024);
    let tau1_guess = 2.;
    let tau2_guess = 6.5;
    let mut model =
        get_double_exponential_model_with_constant_offset(x, vec![tau1_guess, tau2_guess]);
    // true parameters
    let tau1 = 1.;
    let tau2 = 3.;
    // true coefficients
    let c1 = 4.;
    let c2 = 2.5;
    let c3 = 1.; //<- coefficient of constant offset
                 // generate some data without noise
    let y = evaluate_complete_model_at_params(
        &mut model,
        DVector::from_vec(vec![tau1, tau2]),
        &DVector::from(vec![c1, c2, c3]),
    );

    let problem = LevMarProblemBuilder::new(model)
        .observations(y)
        .build()
        .expect("Building valid problem should not panic");

    let result = LevMarSolver::new()
        .fit(problem)
        .expect("fit must complete succesfully");
    assert!(
        result.minimization_report.termination.was_successful(),
        "Levenberg Marquardt did not converge"
    );

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if result.nonlinear_parameters()[0] < result.nonlinear_parameters()[1] {
            (0usize, 1usize)
        } else {
            (1, 0)
        };
    let tau1_calc = result.nonlinear_parameters()[tau1_index];
    let tau2_calc = result.nonlinear_parameters()[tau2_index];
    let c = result
        .linear_coefficients()
        .expect("linear coeffs must exist");
    let c1_calc = c[tau1_index];
    let c2_calc = c[tau2_index];
    let c3_calc = c[2];

    // assert that the calculated coefficients and nonlinear model parameters are correct
    assert_relative_eq!(c1, c1_calc, epsilon = 1e-8);
    assert_relative_eq!(c2, c2_calc, epsilon = 1e-8);
    assert_relative_eq!(c3, c3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);
}

#[test]
fn double_exponential_fitting_without_noise_produces_accurate_results_with_handrolled_model() {
    // true parameters
    let tau1 = 1.;
    let tau2 = 3.;
    // true coefficients
    let c1 = 4.;
    let c2 = 2.5;
    let c3 = 1.; //<- coefficient of constant offset
                 // the independent variable
    let x = linspace(0., 12.5, 1024);
    // guess for nonlinear params
    let tau1_guess = 2.;
    let tau2_guess = 6.5;

    let mut model = DoubleExpModelWithConstantOffsetSepModel::new(x, (tau1_guess, tau2_guess));
    let base_func_count = model.base_function_count();
    // generate some data without noise
    let y = evaluate_complete_model_at_params(
        &mut model,
        Vector2::new(tau1, tau2),
        &OVector::from_vec_generic(base_func_count, U1, vec![c1, c2, c3]),
    );

    let problem = LevMarProblemBuilder::new(model)
        .observations(y)
        .build()
        .expect("Building valid problem should not panic");

    let result = LevMarSolver::new()
        .fit(problem)
        .expect("fitting must exit succesfully");

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if result.nonlinear_parameters()[0] < result.nonlinear_parameters()[1] {
            (0usize, 1usize)
        } else {
            (1, 0)
        };
    let tau1_calc = result.nonlinear_parameters()[tau1_index];
    let tau2_calc = result.nonlinear_parameters()[tau2_index];
    let c = result
        .linear_coefficients()
        .expect("linear coeffs must exist");
    let c1_calc = c[tau1_index];
    let c2_calc = c[tau2_index];
    let c3_calc = c[2];

    // assert that the calculated coefficients and nonlinear model parameters are correct
    assert_relative_eq!(c1, c1_calc, epsilon = 1e-8);
    assert_relative_eq!(c2, c2_calc, epsilon = 1e-8);
    assert_relative_eq!(c3, c3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);

    assert!(
        result.minimization_report.termination.was_successful(),
        "Termination not successful"
    );
}

#[test]
fn double_check_to_make_sure_we_can_rely_on_the_model_to_generat_ground_truth() {
    let x = linspace(0., 12.5, 1024);
    let tau1_guess = 2.;
    let tau2_guess = 6.5;
    let mut model =
        get_double_exponential_model_with_constant_offset(x.clone(), vec![tau1_guess, tau2_guess]);
    // true parameters
    let tau1 = 1.;
    let tau2 = 3.;
    // true coefficients
    let c1 = 4.;
    let c2 = 2.5;
    let c3 = 1.; //<- coefficient of constant offset
                 // generate some data without noise
    let y = evaluate_complete_model_at_params(
        &mut model,
        DVector::from_vec(vec![tau1, tau2]),
        &DVector::from(vec![c1, c2, c3]),
    );
    let f = x.map(|x: f64| c1 * (-x / tau1).exp() + c2 * (-x / tau2).exp() + c3);
    assert_relative_eq!(y, f, epsilon = 1e-8);
}

#[test]
fn double_exponential_fitting_without_noise_produces_accurate_results_with_levenberg_marquardt() {
    // the independent variable
    let x = linspace(0., 12.5, 1024);
    let tau1_guess = 2.;
    let tau2_guess = 6.5;
    let mut model =
        get_double_exponential_model_with_constant_offset(x.clone(), vec![tau1_guess, tau2_guess]);
    // true parameters
    let tau1 = 1.;
    let tau2 = 3.;
    // true coefficients
    let c1 = 4.;
    let c2 = 2.5;
    let c3 = 1.; //<- coefficient of constant offset
                 // generate some data without noise
    let y = evaluate_complete_model_at_params(
        &mut model,
        DVector::from_vec(vec![tau1, tau2]),
        &DVector::from(vec![c1, c2, c3]),
    );

    // for solving the fitting problem using only the levenberg_marquardt crate                  the crate cannot deal with this:     &[tau1_guess,tau2_guess,c1,c2,c3]
    // we have to make the initial guesses closer to the true values for the Levenberg Marquart Algo
    let levenberg_marquart_problem = DoubleExponentialDecayFittingWithOffsetLevmar::new(
        &[tau1_guess, tau2_guess, c1, c2, c3],
        &x,
        &y,
    );

    let (levenberg_marquardt_solution, report) = LevenbergMarquardt::new()
        // if I don't set this, the solver will not converge
        .with_stepbound(1.)
        .minimize(levenberg_marquart_problem);

    assert!(
        report.termination.was_successful(),
        "Levenberg Marquardt did not converge"
    );

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if levenberg_marquardt_solution.params()[0] < levenberg_marquardt_solution.params()[1] {
            (0usize, 1usize)
        } else {
            (1, 0)
        };
    let tau1_calc = levenberg_marquardt_solution.params()[tau1_index];
    let tau2_calc = levenberg_marquardt_solution.params()[tau2_index];
    let c1_calc = levenberg_marquardt_solution.params()[2];
    let c2_calc = levenberg_marquardt_solution.params()[3];
    let c3_calc = levenberg_marquardt_solution.params()[4];

    // assert that the calculated coefficients and nonlinear model parameters are correct
    assert_relative_eq!(c1, c1_calc, epsilon = 1e-8);
    assert_relative_eq!(c2, c2_calc, epsilon = 1e-8);
    assert_relative_eq!(c3, c3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);
    assert!(
        report.termination.was_successful(),
        "Termination not successful"
    );
}

#[test]
// this also tests the correct application of weights
fn oleary_example_with_handrolled_model_produces_correct_results() {
    // those are the initial guesses from the example in the oleary matlab code
    let initial_guess = Vector3::new(0.5, 2., 3.);
    // these are the original timepoints from the matlab code
    let t = DVector::from_vec(vec![
        0., 0.1, 0.22, 0.31, 0.46, 0.50, 0.63, 0.78, 0.85, 0.97,
    ]);
    // the observations from the initial matlab
    let y = DVector::from_vec(vec![
        6.9842, 5.1851, 2.8907, 1.4199, -0.2473, -0.5243, -1.0156, -1.0260, -0.9165, -0.6805,
    ]);
    // and finally the weights for the observations
    // these do actually influence the fits in the second decimal place
    let w = DVector::from_vec(vec![1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5]);

    let model = OLearyExampleModel::new(t, initial_guess);
    let problem = LevMarProblemBuilder::new(model)
        .observations(y)
        .weights(w)
        .build()
        .unwrap();

    let (fit_result, statistics) = LevMarSolver::new()
        .fit_with_statistics(problem)
        .expect("fitting must exit succesfully");
    assert!(
        fit_result.minimization_report.termination.was_successful(),
        "fitting did not terminate successfully"
    );
    let alpha_fit = fit_result.nonlinear_parameters();
    let c_fit = fit_result
        .linear_coefficients()
        .expect("solved problem must have linear coefficients");
    // solved parameters from the matlab code
    // they note that many parameters fit the observations well
    let alpha_true =
        OVector::<f64, U3>::from_vec(vec![1.0132255e+00, 2.4968675e+00, 4.0625148e+00]);
    let c_true = OVector::<f64, U2>::from_vec(vec![5.8416357e+00, 1.1436854e+00]);
    assert_relative_eq!(alpha_fit, alpha_true, epsilon = 1e-5);
    assert_relative_eq!(c_fit, &c_true, epsilon = 1e-5);

    println!("cov = {}", statistics.covariance_matrix());
    println!("sigma = {}", statistics.regression_standard_error());
    println!("wresid = {}", statistics.weighted_residuals());
    let expected_weighted_residuals = DVector::from_column_slice(&[
        -1.1211e-03,
        3.1751e-03,
        -2.7656e-03,
        -1.4600e-03,
        1.2081e-03,
        2.2586e-03,
        -1.1101e-03,
        -2.2554e-03,
        1.3257e-03,
        1.4716e-03,
    ]);
    println!(
        "problem wresid = {}",
        fit_result.problem.residuals().unwrap()
    );
    assert_relative_eq!(
        expected_weighted_residuals,
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );

    // println!("jacobian analytical= {}", problem.jacobian().unwrap());
    // let jacobian_analytical = problem.jacobian().unwrap();
    // let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    // println!("jacobian numerical= {}", jacobian_numerical);
    // assert_relative_eq!(jacobian_analytical, jacobian_numerical, epsilon = 1e-4);
}
