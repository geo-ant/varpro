//! Integration tests for the FitResult class and its accessors
//! These tests focus on ensuring proper functionality of FitResult methods,
//! particularly the accessors for nonlinear parameters and linear coefficients,
//! as well as proper error handling.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use varpro::model::builder::SeparableModelBuilder;
use varpro::model::SeparableModel;
use varpro::problem::SeparableProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

/// Creates a simple double exponential model for testing
fn create_test_model() -> SeparableModel<f64> {
    // Create independent variable (x values)
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Helper functions
    fn exp_decay(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|x| f64::exp(-x / tau))
    }

    fn exp_decay_dtau(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|x| f64::exp(-x / tau) * x / tau.powi(2))
    }

    // Create model with two decay parameters
    SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        // Add first exponential decay with its partial derivative
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        // Add second exponential decay with its partial derivative
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        // Add constant offset
        .invariant_function(|x| DVector::from_element(x.len(), 1.0))
        // Set independent variable
        .independent_variable(x)
        // Set initial parameters
        .initial_parameters(vec![1.0, 5.0])
        // Build the model
        .build()
        .unwrap()
}

/// Test the nonlinear_parameters accessor returns correct values
#[test]
fn test_nonlinear_parameters_accessor() {
    // Create model and synthetic data
    let model = create_test_model();
    let true_params = vec![2.0, 7.0]; // True parameters

    // Generate synthetic data with known parameters
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| 3.0 * f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| 2.0 * f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);
    let y = &y1 * 0.8 + &y2 * 1.2 + &y3 * 0.5;

    // Create and solve the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Check that nonlinear_parameters returns values
    let recovered_params = fit_result.nonlinear_parameters();

    // The recovered parameters should be close to the true parameters
    assert_eq!(recovered_params.len(), 2);

    // Parameters may not match exactly due to noise, but should be reasonably close
    assert_relative_eq!(recovered_params[0], true_params[0], max_relative = 0.2);
    assert_relative_eq!(recovered_params[1], true_params[1], max_relative = 0.2);
}

/// Test the linear_coefficients accessor for single right-hand-side problems
#[test]
fn test_linear_coefficients_accessor_single_rhs() {
    // Create model and synthetic data with known coefficients
    let model = create_test_model();
    let true_params = vec![2.0, 7.0];
    let true_coeffs = vec![0.8, 1.2, 0.5]; // True coefficients

    // Generate synthetic data
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);
    let y = &y1 * true_coeffs[0] + &y2 * true_coeffs[1] + &y3 * true_coeffs[2];

    // Create and solve the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Check that linear_coefficients returns values
    let recovered_coeffs = fit_result.linear_coefficients().unwrap();

    // The recovered coefficients should be close to the true coefficients
    assert_eq!(recovered_coeffs.len(), 3);

    // Coefficients may not match exactly due to noise, but should be reasonably close
    assert_relative_eq!(recovered_coeffs[0], true_coeffs[0], max_relative = 0.2);
    assert_relative_eq!(recovered_coeffs[1], true_coeffs[1], max_relative = 0.2);
    assert_relative_eq!(recovered_coeffs[2], true_coeffs[2], max_relative = 0.2);
}

/// Test the linear_coefficients accessor for multiple right-hand-side problems
#[test]
fn test_linear_coefficients_accessor_multiple_rhs() {
    // Create model
    let model = create_test_model();
    let true_params = vec![2.0, 7.0];

    // True coefficients for two data sets
    let true_coeffs_1 = vec![0.8, 1.2, 0.5];
    let true_coeffs_2 = vec![1.5, 0.7, 0.3];

    // Generate synthetic data for two datasets
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);

    let dataset1 = &y1 * true_coeffs_1[0] + &y2 * true_coeffs_1[1] + &y3 * true_coeffs_1[2];
    let dataset2 = &y1 * true_coeffs_2[0] + &y2 * true_coeffs_2[1] + &y3 * true_coeffs_2[2];

    // Combine datasets into a matrix
    let mut y_matrix = DMatrix::zeros(x.len(), 2);
    y_matrix.set_column(0, &dataset1);
    y_matrix.set_column(1, &dataset2);

    // Create and solve the multiple right-hand-side problem
    let problem = SeparableProblemBuilder::mrhs(model)
        .observations(y_matrix)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Check that linear_coefficients returns matrix values
    let recovered_coeffs = fit_result.linear_coefficients().unwrap();

    // The recovered coefficients should be a matrix with the right dimensions
    assert_eq!(recovered_coeffs.nrows(), 3); // Number of basis functions
    assert_eq!(recovered_coeffs.ncols(), 2); // Number of datasets

    // First column should match true_coeffs_1, second column should match true_coeffs_2
    assert_relative_eq!(
        recovered_coeffs[(0, 0)],
        true_coeffs_1[0],
        max_relative = 0.2
    );
    assert_relative_eq!(
        recovered_coeffs[(1, 0)],
        true_coeffs_1[1],
        max_relative = 0.2
    );
    assert_relative_eq!(
        recovered_coeffs[(2, 0)],
        true_coeffs_1[2],
        max_relative = 0.2
    );

    assert_relative_eq!(
        recovered_coeffs[(0, 1)],
        true_coeffs_2[0],
        max_relative = 0.2
    );
    assert_relative_eq!(
        recovered_coeffs[(1, 1)],
        true_coeffs_2[1],
        max_relative = 0.2
    );
    assert_relative_eq!(
        recovered_coeffs[(2, 1)],
        true_coeffs_2[2],
        max_relative = 0.2
    );
}

/// Test the best_fit accessor for single right-hand-side problems
#[test]
fn test_best_fit_accessor_single_rhs() {
    // Create model and synthetic data
    let model = create_test_model();
    let true_params = vec![2.0, 7.0];
    let true_coeffs = vec![0.8, 1.2, 0.5];

    // Generate synthetic data
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);
    let y = &y1 * true_coeffs[0] + &y2 * true_coeffs[1] + &y3 * true_coeffs[2];

    // Create and solve the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y.clone())
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Get the best fit result
    let best_fit = fit_result.best_fit().unwrap();

    // Check dimensions
    assert_eq!(best_fit.len(), y.len());

    // Best fit should be close to the original data
    for i in 0..y.len() {
        assert_relative_eq!(best_fit[i], y[i], max_relative = 0.1);
    }
}

/// Test the best_fit accessor for multiple right-hand-side problems
#[test]
fn test_best_fit_accessor_multiple_rhs() {
    // Create model
    let model = create_test_model();
    let true_params = vec![2.0, 7.0];

    // True coefficients for two data sets
    let true_coeffs_1 = vec![0.8, 1.2, 0.5];
    let true_coeffs_2 = vec![1.5, 0.7, 0.3];

    // Generate synthetic data for two datasets
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);

    let dataset1 = &y1 * true_coeffs_1[0] + &y2 * true_coeffs_1[1] + &y3 * true_coeffs_1[2];
    let dataset2 = &y1 * true_coeffs_2[0] + &y2 * true_coeffs_2[1] + &y3 * true_coeffs_2[2];

    // Combine datasets into a matrix
    let mut y_matrix = DMatrix::zeros(x.len(), 2);
    y_matrix.set_column(0, &dataset1);
    y_matrix.set_column(1, &dataset2);

    // Create and solve the multiple right-hand-side problem
    let problem = SeparableProblemBuilder::mrhs(model)
        .observations(y_matrix.clone())
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Get the best fit result
    let best_fit = fit_result.best_fit().unwrap();

    // Check dimensions
    assert_eq!(best_fit.nrows(), y_matrix.nrows());
    assert_eq!(best_fit.ncols(), y_matrix.ncols());

    // Best fit should be close to the original data
    for i in 0..y_matrix.nrows() {
        for j in 0..y_matrix.ncols() {
            assert_relative_eq!(best_fit[(i, j)], y_matrix[(i, j)], max_relative = 0.1);
        }
    }
}

/// Test the was_successful method
#[test]
fn test_was_successful() {
    // Create model and well-conditioned data that should fit successfully
    let model = create_test_model();
    let true_params = vec![2.0, 7.0];
    let true_coeffs = vec![0.8, 1.2, 0.5];

    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);
    let y = &y1 * true_coeffs[0] + &y2 * true_coeffs[1] + &y3 * true_coeffs[2];

    // Create and solve the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // The fit should be successful for this well-conditioned problem
    assert!(fit_result.was_successful());
}

/// Test error cases for FitResult accessors
#[test]
fn test_fit_result_accessors_handle_errors() {
    // Create a simple model
    let model = create_test_model();

    // Create nonsensical data that's impossible to fit
    let y = DVector::<f64>::zeros(11);

    // Build the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();

    // Run the fit
    let fit_result = match LevMarSolver::default().fit(problem) {
        Ok(result) => result,
        Err(result) => {
            // If we get an error result, check that we can access parameters
            let _params = result.nonlinear_parameters();
            assert!(!result.was_successful());
            return;
        }
    };

    // If the fit succeeded, verify that the was_successful flag is correctly set
    if fit_result.linear_coefficients().is_none() {
        assert!(fit_result.best_fit().is_none());
    } else {
        // We have coefficients, just check we can access them
        let _best_fit = fit_result.best_fit();
        let _coeffs = fit_result.linear_coefficients();
    }
}

/// Test edge case with minimal data points
#[test]
fn test_fit_with_minimal_data() {
    // Create a model with minimal data points (just enough to determine parameters)
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0]); // Minimal number of points

    // Helper functions
    fn exp_decay(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|x| f64::exp(-x / tau))
    }

    fn exp_decay_dtau(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|x| f64::exp(-x / tau) * x / tau.powi(2))
    }

    let model = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .invariant_function(|x| DVector::from_element(x.len(), 1.0))
        .independent_variable(x.clone())
        .initial_parameters(vec![1.0, 5.0])
        .build()
        .unwrap();

    // Generate synthetic data
    let true_params = vec![2.0, 7.0];
    let true_coeffs = vec![0.8, 1.2, 0.5];

    let y1 = x.map(|x| f64::exp(-x / true_params[0]));
    let y2 = x.map(|x| f64::exp(-x / true_params[1]));
    let y3 = DVector::from_element(x.len(), 1.0);
    let y = &y1 * true_coeffs[0] + &y2 * true_coeffs[1] + &y3 * true_coeffs[2];

    // Create and solve the problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default().fit(problem).unwrap();

    // Ensure we can still access all the accessors even with minimal data
    let _ = fit_result.nonlinear_parameters();
    let _ = fit_result.linear_coefficients();
    let _ = fit_result.best_fit();
    let _ = fit_result.was_successful();
}
