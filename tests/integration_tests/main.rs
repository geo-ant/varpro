use approx::assert_relative_eq;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::OVector;
use nalgebra::U1;
use shared_test_code::evaluate_complete_model_at_params;
use shared_test_code::get_double_exponential_model_with_constant_offset;
use shared_test_code::levmar_mrhs::DoubleExponentialModelWithConstantOffsetLevmarMrhs;
use shared_test_code::linspace;
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::models::DoubleExponentialDecayFittingWithOffsetLevmar;
use shared_test_code::models::OLearyExampleModel;
use shared_test_code::models::o_leary_example_model;
use varpro::prelude::*;
use varpro::problem::SeparableProblemBuilder;
use varpro::solvers::levmar::*;

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
#[allow(non_snake_case)]
fn sanity_check_jacobian_of_levenberg_marquardt_problem_mrhs_is_correct() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 2.;
    let tau2 = 4.;
    // coefficients for the first dataset
    let a1 = 2.;
    let a2 = 4.;
    let a3 = 0.2;
    // coefficients for the second dataset
    let b1 = 5.;
    let b2 = 1.;
    let b3 = 9.;
    let mut Y = DMatrix::zeros(x.len(), 2);

    Y.set_column(
        0,
        &x.map(|x: f64| a1 * (-x / tau1).exp() + a2 * (-x / tau2).exp() + a3),
    );
    Y.set_column(
        1,
        &x.map(|x: f64| b1 * (-x / tau1).exp() + b2 * (-x / tau2).exp() + b3),
    );

    let initial_params = [
        0.5 * tau1,
        1.5 * tau2,
        2. * a1,
        3. * a2,
        3. * a3,
        1.2 * b1,
        3. * b2,
        5. * b3,
    ];
    let mut problem = DoubleExponentialModelWithConstantOffsetLevmarMrhs::new(initial_params, x, Y);
    assert_eq!(
        problem.params().as_slice(),
        &initial_params,
        "params not set correctly"
    );

    // Let `problem` be an instance of `LeastSquaresProblem`
    let jacobian_numerical = levenberg_marquardt::differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-4);
}

#[test]
fn double_exponential_fitting_without_noise_produces_accurate_results() {
    // the independent variable
    let x = linspace(0., 12.5, 1024);
    let tau1_guess = 2.;
    let tau2_guess = 6.5;
    let mut model =
        get_double_exponential_model_with_constant_offset(x, vec![tau1_guess, tau2_guess]);
    _ = format!("{:?}", model);
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

    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        .build()
        .expect("Building valid problem should not panic");
    _ = format!("{:?}", problem);

    let (fit_result, statistics) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fit must complete succesfully");
    assert!(
        fit_result.minimization_report.termination.was_successful(),
        "Levenberg Marquardt did not converge"
    );
    assert_relative_eq!(fit_result.best_fit().unwrap(), y, epsilon = 1e-5);
    assert_relative_eq!(
        fit_result.problem.residuals().unwrap(),
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if fit_result.nonlinear_parameters()[0] < fit_result.nonlinear_parameters()[1] {
            (0usize, 1usize)
        } else {
            (1, 0)
        };
    let tau1_calc = fit_result.nonlinear_parameters()[tau1_index];
    let tau2_calc = fit_result.nonlinear_parameters()[tau2_index];
    let c = fit_result
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
        OVector::from_column_slice_generic(Dyn(2), U1, &[tau1, tau2]),
        &OVector::from_vec_generic(Dyn(base_func_count), U1, vec![c1, c2, c3]),
    );

    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        .build()
        .expect("Building valid problem should not panic");

    let (fit_result, statistics) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fitting must exit succesfully");

    assert_relative_eq!(
        fit_result.problem.residuals().unwrap(),
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );

    assert_relative_eq!(fit_result.best_fit().unwrap(), y, epsilon = 1e-5);

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if fit_result.nonlinear_parameters()[0] < fit_result.nonlinear_parameters()[1] {
            (0usize, 1usize)
        } else {
            (1, 0)
        };
    let tau1_calc = fit_result.nonlinear_parameters()[tau1_index];
    let tau2_calc = fit_result.nonlinear_parameters()[tau2_index];
    let c = fit_result
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
        fit_result.minimization_report.termination.was_successful(),
        "Termination not successful"
    );
}

#[test]
fn double_check_to_make_sure_we_can_rely_on_the_model_to_generate_ground_truth() {
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
#[allow(non_snake_case)]
fn double_exponential_model_with_levenberg_marquardt_mrhs_produces_accurate_results() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 1.;
    let tau2 = 3.;
    // cannot guess too far away from the true params
    // otherwise the solution will not converge
    let tau1_guess = 2.0;
    let tau2_guess = 4.0;
    // coefficients for the first dataset
    let a1 = 2.;
    let a2 = 4.;
    let a3 = 0.2;
    // coefficients for the second dataset
    let b1 = 5.;
    let b2 = 1.;
    let b3 = 9.;
    let mut Y = DMatrix::zeros(x.len(), 2);

    Y.set_column(
        0,
        &x.map(|x: f64| a1 * (-x / tau1).exp() + a2 * (-x / tau2).exp() + a3),
    );
    Y.set_column(
        1,
        &x.map(|x: f64| b1 * (-x / tau1).exp() + b2 * (-x / tau2).exp() + b3),
    );

    let initial_params = [
        tau1_guess,
        tau2_guess,
        // introduce small variations around the true parameters
        // to make sure this works, but they cannot be too big because
        // otherwise the levmar minimizer will get upset and its hair will fall out
        0.9 * a1,
        1.1 * a2,
        1.1 * a3,
        1.1 * b1,
        1.1 * b2,
        0.9 * b3,
    ];
    let problem = DoubleExponentialModelWithConstantOffsetLevmarMrhs::new(initial_params, x, Y);
    let (levenberg_marquardt_solution, report) = LevenbergMarquardt::new()
        // if I don't set this, the solver will not converge
        .with_stepbound(1.)
        .with_patience(1000)
        .minimize(problem);

    assert!(
        report.termination.was_successful(),
        "Levenberg Marquardt did not converge"
    );
    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if levenberg_marquardt_solution.params()[0] < levenberg_marquardt_solution.params()[1] {
            (0, 1)
        } else {
            (1, 0)
        };
    let tau1_calc = levenberg_marquardt_solution.params()[tau1_index];
    let tau2_calc = levenberg_marquardt_solution.params()[tau2_index];
    let a1_calc = levenberg_marquardt_solution.params()[2 + tau1_index];
    let a2_calc = levenberg_marquardt_solution.params()[2 + tau2_index];
    let a3_calc = levenberg_marquardt_solution.params()[4];
    let b1_calc = levenberg_marquardt_solution.params()[5 + tau1_index];
    let b2_calc = levenberg_marquardt_solution.params()[5 + tau2_index];
    let b3_calc = levenberg_marquardt_solution.params()[7];

    assert_relative_eq!(a1, a1_calc, epsilon = 1e-8);
    assert_relative_eq!(a2, a2_calc, epsilon = 1e-8);
    assert_relative_eq!(a3, a3_calc, epsilon = 1e-8);
    assert_relative_eq!(b1, b1_calc, epsilon = 1e-8);
    assert_relative_eq!(b2, b2_calc, epsilon = 1e-8);
    assert_relative_eq!(b3, b3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);
}

#[test]
#[allow(non_snake_case)]
fn double_exponential_model_with_handrolled_model_mrhs_produces_accurate_results() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 1.;
    let tau2 = 3.;
    let tau1_guess = 2.5;
    let tau2_guess = 6.5;
    // coefficients for the first dataset
    let a1 = 2.;
    let a2 = 4.;
    let a3 = 0.2;
    // coefficients for the second dataset
    let b1 = 5.;
    let b2 = 1.;
    let b3 = 9.;
    let mut Y = DMatrix::zeros(x.len(), 2);

    Y.set_column(
        0,
        &x.map(|x: f64| a1 * (-x / tau1).exp() + a2 * (-x / tau2).exp() + a3),
    );
    Y.set_column(
        1,
        &x.map(|x: f64| b1 * (-x / tau1).exp() + b2 * (-x / tau2).exp() + b3),
    );

    let model = DoubleExpModelWithConstantOffsetSepModel::new(x, (tau1_guess, tau2_guess));
    let problem = SeparableProblemBuilder::mrhs(model)
        .observations(Y.clone())
        .build()
        .expect("building the lev mar problem must not fail");

    let fit_result = LevMarSolver::default()
        .fit(problem)
        .expect("fitting must not fail");

    assert_relative_eq!(fit_result.best_fit().unwrap(), Y, epsilon = 1e-5);

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if fit_result.nonlinear_parameters()[0] < fit_result.nonlinear_parameters()[1] {
            (0, 1)
        } else {
            (1, 0)
        };
    let tau1_calc = fit_result.nonlinear_parameters()[tau1_index];
    let tau2_calc = fit_result.nonlinear_parameters()[tau2_index];
    let coeff = fit_result
        .linear_coefficients()
        .expect("linear coefficients must exist");
    let a1_calc = coeff[tau1_index];
    let a2_calc = coeff[tau2_index];
    let a3_calc = coeff[2];
    let b1_calc = coeff[3 + tau1_index];
    let b2_calc = coeff[3 + tau2_index];
    let b3_calc = coeff[5];

    assert_relative_eq!(a1, a1_calc, epsilon = 1e-8);
    assert_relative_eq!(a2, a2_calc, epsilon = 1e-8);
    assert_relative_eq!(a3, a3_calc, epsilon = 1e-8);
    assert_relative_eq!(b1, b1_calc, epsilon = 1e-8);
    assert_relative_eq!(b2, b2_calc, epsilon = 1e-8);
    assert_relative_eq!(b3, b3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);
}

#[test]
#[allow(non_snake_case)]
fn triple_exponential_model_with_mrhs_produces_accurate_results_with_more_data_cols_than_params() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 1.;
    let tau2 = 3.;
    let tau1_guess = 2.5;
    let tau2_guess = 6.5;
    // coefficients for the first dataset
    let a1 = 2.;
    let a2 = 4.;
    let a3 = 0.2;
    // coefficients for the second dataset
    let b1 = 10.;
    let b2 = 12.;
    let b3 = 18.;
    // coefficients for third dataset
    let c1 = 5.;
    let c2 = 1.;
    let c3 = 9.;

    let mut Y = DMatrix::zeros(x.len(), 3);

    Y.set_column(
        0,
        &x.map(|x: f64| a1 * (-x / tau1).exp() + a2 * (-x / tau2).exp() + a3),
    );
    Y.set_column(
        1,
        &x.map(|x: f64| b1 * (-x / tau1).exp() + b2 * (-x / tau2).exp() + b3),
    );
    Y.set_column(
        2,
        &x.map(|x: f64| c1 * (-x / tau1).exp() + c2 * (-x / tau2).exp() + c3),
    );

    let model = DoubleExpModelWithConstantOffsetSepModel::new(x, (tau1_guess, tau2_guess));
    let problem = SeparableProblemBuilder::mrhs(model)
        .observations(Y.clone())
        .build()
        .expect("building the lev mar problem must not fail");

    let fit_result = LevMarSolver::default()
        .fit(problem)
        .expect("fitting must not fail");

    assert_relative_eq!(fit_result.best_fit().unwrap(), Y, epsilon = 1e-5);

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let (tau1_index, tau2_index) =
        if fit_result.nonlinear_parameters()[0] < fit_result.nonlinear_parameters()[1] {
            (0, 1)
        } else {
            (1, 0)
        };
    let tau1_calc = fit_result.nonlinear_parameters()[tau1_index];
    let tau2_calc = fit_result.nonlinear_parameters()[tau2_index];
    let coeff = fit_result
        .linear_coefficients()
        .expect("linear coefficients must exist");
    let a1_calc = coeff[tau1_index];
    let a2_calc = coeff[tau2_index];
    let a3_calc = coeff[2];
    let b1_calc = coeff[3 + tau1_index];
    let b2_calc = coeff[3 + tau2_index];
    let b3_calc = coeff[5];
    let c1_calc = coeff[6 + tau1_index];
    let c2_calc = coeff[6 + tau2_index];
    let c3_calc = coeff[8];

    println!("true: a = [{a1},{a2},{a3}], b = [{b1},{b2},{b3}], c = [{c1},{c2},{c3}]");
    println!(
        "calc: a = [{a1_calc},{a2_calc},{a3_calc}], b = [{b1_calc},{b2_calc},{b3_calc}], c = [{c1_calc},{c2_calc},{c3_calc}]"
    );

    assert_relative_eq!(a1, a1_calc, epsilon = 1e-8);
    assert_relative_eq!(a2, a2_calc, epsilon = 1e-8);
    assert_relative_eq!(a3, a3_calc, epsilon = 1e-8);
    assert_relative_eq!(b1, b1_calc, epsilon = 1e-8);
    assert_relative_eq!(b2, b2_calc, epsilon = 1e-8);
    assert_relative_eq!(b3, b3_calc, epsilon = 1e-8);
    assert_relative_eq!(c1, c1_calc, epsilon = 1e-8);
    assert_relative_eq!(c2, c2_calc, epsilon = 1e-8);
    assert_relative_eq!(c3, c3_calc, epsilon = 1e-8);
    assert_relative_eq!(tau1, tau1_calc, epsilon = 1e-8);
    assert_relative_eq!(tau2, tau2_calc, epsilon = 1e-8);
}

#[test]
fn double_exponential_model_with_noise_gives_same_confidence_interval_as_lmfit() {
    // I have python scripts using the lmfit package that allow me to test
    // my results.
    // this tests against the file python/multiexp_decay.py
    // see there for more details. The parameters are taken from there.

    let x = read_vec_f64(
        "test_assets/multiexp_decay/xdata_1000_64bit.raw",
        Some(1000),
    );
    let y = read_vec_f64(
        "test_assets/multiexp_decay/ydata_1000_64bit.raw",
        Some(1000),
    );
    let conf_radius = read_vec_f64("test_assets/multiexp_decay/conf_1000_64bit.raw", Some(1000));
    let covmat = read_vec_f64("test_assets/multiexp_decay/covmat_5x5_64bit.raw", Some(25));
    let model = DoubleExpModelWithConstantOffsetSepModel::new(DVector::from_vec(x), (1., 7.));
    let problem = SeparableProblemBuilder::new(model)
        .observations(DVector::from_vec(y))
        .build()
        .expect("building the lev mar problem must not fail");

    let (fit_result, fit_stat) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fitting must not fail");

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let tau1_calc = fit_result.nonlinear_parameters()[0];
    let tau2_calc = fit_result.nonlinear_parameters()[1];
    let coeff = fit_result
        .linear_coefficients()
        .expect("linear coefficients must exist");
    let a1_calc = coeff[0];
    let a2_calc = coeff[1];
    let a3_calc = coeff[2];

    // parameters are taken from the python/multiexp_decay
    // script in the root dir of this library. We compare against the
    // fit results of the python lmfit library
    // run the script to see the output
    assert_relative_eq!(2.19344628, a1_calc, epsilon = 1e-5);
    assert_relative_eq!(6.80462652, a2_calc, epsilon = 1e-5);
    assert_relative_eq!(1.59995673, a3_calc, epsilon = 1e-5);
    assert_relative_eq!(2.40392137, tau1_calc, epsilon = 1e-5);
    assert_relative_eq!(5.99571068, tau2_calc, epsilon = 1e-5);

    assert_relative_eq!(fit_stat.reduced_chi2(), 1.0109e-4, epsilon = 1e-8);

    // covariance matrix is correct
    let expected_covmat = DMatrix::from_row_slice(5, 5, &covmat);
    let calculated_covmat = fit_stat.covariance_matrix();
    assert_relative_eq!(expected_covmat, calculated_covmat, epsilon = 1e-6);

    // now for the confidence intervals
    assert_relative_eq!(
        DVector::from_vec(conf_radius),
        fit_stat.confidence_band_radius(0.88),
        epsilon = 1e-6,
    );
}

#[test]
fn weighted_double_exponential_model_with_noise_gives_same_confidence_interval_as_lmfit() {
    // I have python scripts using the lmfit package that allow me to test
    // my results.
    // this tests against the file python/weighted_multiexp_decay.py
    // see there for more details. The parameters are taken from there.

    let x = read_vec_f64(
        "test_assets/weighted_multiexp_decay/xdata_1000_64bit.raw",
        Some(1000),
    );
    let y = DVector::from_vec(read_vec_f64(
        "test_assets/weighted_multiexp_decay/ydata_1000_64bit.raw",
        Some(1000),
    ));
    let conf_radius = read_vec_f64(
        "test_assets/weighted_multiexp_decay/conf_1000_64bit.raw",
        Some(1000),
    );
    let covmat = read_vec_f64(
        "test_assets/weighted_multiexp_decay/covmat_5x5_64bit.raw",
        Some(25),
    );
    let model = DoubleExpModelWithConstantOffsetSepModel::new(DVector::from_vec(x), (1., 7.));
    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        // in the python script we also apply these weights
        .weights(y.map(|v| 1. / v.sqrt()))
        .build()
        .expect("building the lev mar problem must not fail");

    let (fit_result, fit_stat) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fitting must not fail");

    // extract the calculated paramters, because tau1 and tau2 might switch places here
    let tau1_calc = fit_result.nonlinear_parameters()[0];
    let tau2_calc = fit_result.nonlinear_parameters()[1];
    let coeff = fit_result
        .linear_coefficients()
        .expect("linear coefficients must exist");
    let a1_calc = coeff[0];
    let a2_calc = coeff[1];
    let a3_calc = coeff[2];

    // parameters are taken from the python/weighted_multiexp_decay
    // script in the root dir of this library. We compare against the
    // fit results of the python lmfit library
    // run the script to see the output
    assert_relative_eq!(2.24275841, a1_calc, epsilon = 1e-5);
    assert_relative_eq!(6.75609070, a2_calc, epsilon = 1e-5);
    assert_relative_eq!(1.59790007, a3_calc, epsilon = 1e-5);
    assert_relative_eq!(2.43119160, tau1_calc, epsilon = 1e-5);
    assert_relative_eq!(6.02052311, tau2_calc, epsilon = 1e-5);

    assert_relative_eq!(fit_stat.reduced_chi2(), 3.2117e-5, epsilon = 1e-8);
    assert_relative_eq!(
        fit_stat.reduced_chi2().sqrt(),
        fit_stat.regression_standard_error(),
        epsilon = 1e-8
    );

    // covariance matrix is correct
    let expected_covmat = DMatrix::from_row_slice(5, 5, &covmat);
    let calculated_covmat = fit_stat.covariance_matrix();
    assert_relative_eq!(expected_covmat, calculated_covmat, epsilon = 1e-6);

    // now for the confidence intervals
    assert_relative_eq!(
        DVector::from_vec(conf_radius),
        fit_stat.confidence_band_radius(0.88),
        epsilon = 1e-6,
    );
}

// helper function to read a vector of f64 from a file
fn read_vec_f64(path: impl AsRef<std::path::Path>, size_hint: Option<usize>) -> Vec<f64> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut vect = Vec::with_capacity(size_hint.unwrap_or(1024));

    let f = std::fs::File::open(path).expect("error opening file");
    let mut r = std::io::BufReader::new(f);

    loop {
        match r.read_f64::<LittleEndian>() {
            Ok(val) => {
                vect.push(val);
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(_e) => panic!("error parsing file"),
        }
    }

    vect
}

#[test]
// this also tests the correct application of weights
fn oleary_example_with_handrolled_model_produces_correct_results() {
    // those are the initial guesses from the example in the oleary matlab code
    let initial_guess = OVector::from_column_slice_generic(Dyn(3), U1, &[0.5, 2., 3.]);
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
    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        .weights(w)
        .build()
        .unwrap();

    let (fit_result, statistics) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fitting must exit succesfully");
    assert!(
        fit_result.minimization_report.termination.was_successful(),
        "fitting did not terminate successfully"
    );

    assert_relative_eq!(fit_result.best_fit().unwrap(), y, epsilon = 1e-2);

    let alpha_fit = fit_result.nonlinear_parameters();
    let c_fit = fit_result
        .linear_coefficients()
        .expect("solved problem must have linear coefficients");
    // solved parameters from the matlab code
    // they note that many parameters fit the observations well
    let alpha_true =
        OVector::<f64, Dyn>::from_vec(vec![1.0132255e+00, 2.4968675e+00, 4.0625148e+00]);
    let c_true = OVector::<f64, Dyn>::from_vec(vec![5.8416357e+00, 1.1436854e+00]);
    assert_relative_eq!(alpha_fit, alpha_true, epsilon = 1e-5);
    assert_relative_eq!(c_fit.into_owned(), c_true, epsilon = 1e-5);

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
    assert_relative_eq!(
        expected_weighted_residuals,
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        fit_result.problem.residuals().unwrap(),
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );

    let expected_sigma = 2.7539e-03;
    assert_relative_eq!(
        statistics.regression_standard_error(),
        expected_sigma,
        epsilon = 1e-5
    );

    let expected_covariance_matrix = nalgebra::dmatrix![
    4.4887e-03,  -4.4309e-03,  -2.1613e-04,  -4.6980e-04,  -1.9052e-03;
      -4.4309e-03,   4.3803e-03,   2.1087e-04,   4.7170e-04,   1.8828e-03;
      -2.1613e-04,   2.1087e-04,   2.6925e-04,  -3.6450e-05,   5.1919e-05;
      -4.6980e-04,   4.7170e-04,  -3.6450e-05,   8.5784e-05,   2.0534e-04;
      -1.9052e-03,   1.8828e-03,   5.1919e-05,   2.0534e-04,   8.2272e-04;
    ];
    assert_relative_eq!(
        statistics.covariance_matrix(),
        &expected_covariance_matrix,
        epsilon = 1e-5,
    );

    assert_relative_eq!(
        statistics.nonlinear_parameters_variance(),
        nalgebra::dvector![2.6925e-04, 8.5784e-05, 8.2272e-04],
        epsilon = 1e-5,
    );

    assert_relative_eq!(
        statistics.linear_coefficients_variance(),
        nalgebra::dvector![4.4887e-03, 4.3803e-03],
        epsilon = 1e-5,
    );

    let expected_correlation_matrix = nalgebra::dmatrix![
     1.0000,  -0.9993,  -0.1966,  -0.7571,  -0.9914;
    -0.9993,   1.0000,   0.1942,   0.7695,   0.9918;
    -0.1966,   0.1942,   1.0000,  -0.2398,   0.1103;
    -0.7571,   0.7695,  -0.2398,   1.0000,   0.7729;
    -0.9914,   0.9918,   0.1103,   0.7729,   1.0000;
      ];
    assert_relative_eq!(
        statistics.calculate_correlation_matrix(),
        &expected_correlation_matrix,
        epsilon = 1e-4
    );
}

#[test]
// this also tests the correct application of weights
fn test_oleary_example_with_separable_model() {
    // those are the initial guesses from the example in the oleary matlab code
    let initial_guess = vec![0.5, 2., 3.];
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

    let model = o_leary_example_model(t, initial_guess);
    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        .weights(w)
        .build()
        .unwrap();

    let (fit_result, statistics) = LevMarSolver::default()
        .fit_with_statistics(problem)
        .expect("fitting must exit succesfully");
    assert!(
        fit_result.minimization_report.termination.was_successful(),
        "fitting did not terminate successfully"
    );

    assert_relative_eq!(fit_result.best_fit().unwrap(), y, epsilon = 1e-2);

    let alpha_fit = fit_result.nonlinear_parameters();
    let c_fit = fit_result
        .linear_coefficients()
        .expect("solved problem must have linear coefficients");
    // solved parameters from the matlab code
    // they note that many parameters fit the observations well
    let alpha_true = DVector::from_vec(vec![1.0132255e+00, 2.4968675e+00, 4.0625148e+00]);
    let c_true = DVector::from_vec(vec![5.8416357e+00, 1.1436854e+00]);
    assert_relative_eq!(alpha_fit, alpha_true, epsilon = 1e-5);
    assert_relative_eq!(c_fit.into_owned(), &c_true, epsilon = 1e-5);

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
    assert_relative_eq!(
        expected_weighted_residuals,
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        fit_result.problem.residuals().unwrap(),
        statistics.weighted_residuals(),
        epsilon = 1e-5
    );

    let expected_sigma = 2.7539e-03;
    assert_relative_eq!(
        statistics.regression_standard_error(),
        expected_sigma,
        epsilon = 1e-5
    );

    let expected_covariance_matrix = DMatrix::from_row_slice(
        5,
        5,
        &[
            4.4887e-03,
            -4.4309e-03,
            -2.1613e-04,
            -4.6980e-04,
            -1.9052e-03,
            -4.4309e-03,
            4.3803e-03,
            2.1087e-04,
            4.7170e-04,
            1.8828e-03,
            -2.1613e-04,
            2.1087e-04,
            2.6925e-04,
            -3.6450e-05,
            5.1919e-05,
            -4.6980e-04,
            4.7170e-04,
            -3.6450e-05,
            8.5784e-05,
            2.0534e-04,
            -1.9052e-03,
            1.8828e-03,
            5.1919e-05,
            2.0534e-04,
            8.2272e-04,
        ],
    );
    assert_relative_eq!(
        statistics.covariance_matrix(),
        &expected_covariance_matrix,
        epsilon = 1e-5,
    );

    assert_relative_eq!(
        statistics.nonlinear_parameters_variance(),
        DVector::from_row_slice(&[2.6925e-04, 8.5784e-05, 8.2272e-04]),
        epsilon = 1e-5,
    );
    assert_relative_eq!(
        statistics.linear_coefficients_variance(),
        DVector::from_row_slice(&[4.4887e-03, 4.3803e-03]),
        epsilon = 1e-5,
    );

    let expected_correlation_matrix = DMatrix::from_row_slice(
        5,
        5,
        &[
            1.0000, -0.9993, -0.1966, -0.7571, -0.9914, -0.9993, 1.0000, 0.1942, 0.7695, 0.9918,
            -0.1966, 0.1942, 1.0000, -0.2398, 0.1103, -0.7571, 0.7695, -0.2398, 1.0000, 0.7729,
            -0.9914, 0.9918, 0.1103, 0.7729, 1.0000,
        ],
    );
    assert_relative_eq!(
        statistics.calculate_correlation_matrix(),
        &expected_correlation_matrix,
        epsilon = 1e-4
    );
}
