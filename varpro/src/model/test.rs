use nalgebra::DVector;
use assert_matches::assert_matches;
use crate::model::builder::SeparableModelBuilder;
use crate::model::errors::ModelError;
use crate::prelude::*;
use crate::test_helpers;

/// a dummy structure that implements a separable nonlinear model trait
/// but will panic when any of the methods is actually called
#[derive(Default, Copy, Clone)]
pub struct DummySeparableModel {}

impl SeparableNonlinearModel<f64> for DummySeparableModel {
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        todo!()
    }

    fn base_function_count(&self) -> usize {
        todo!()
    }

    fn eval(
        &self,
    ) -> Result<nalgebra::DMatrix<f64>, Self::Error> {
        todo!()
    }

    fn eval_partial_deriv(
        &self,
        _derivative_index: usize,
    ) -> Result<nalgebra::DMatrix<f64>, Self::Error> {
        todo!()
    }

    fn set_params(&mut self, _parameters : &[f64]) -> Result<(),Self::Error> {
        todo!()
    }

    fn params(&self) -> DVector<f64> {
        todo!()
    }
}

#[test]
fn model_gets_initialized_with_correct_parameter_names_and_count() {
    let model = test_helpers::get_double_exponential_model_with_constant_offset(DVector::zeros(10),vec![1.,2.]);
    assert_eq!(
        model.parameter_count(),
        2,
        "Double exponential model has 2 parameters"
    );
    assert_eq!(
        model.parameters(),
        &["tau1", "tau2"],
        "Double exponential model has 2 parameters"
    );
}

#[test]
// test that the eval method produces correct results and gives a matrix that is ordered correctly
fn model_function_eval_produces_correct_result() {

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let tau1 = 1.;
    let tau2 = 3.;

    let params = &[tau1, tau2];
    let model = test_helpers::get_double_exponential_model_with_constant_offset(DVector::zeros(10),params.to_vec());
    let eval_matrix = model
        .eval()
        .expect("Model evaluation should not fail");

    assert_eq!(
        DVector::from(eval_matrix.column(0)),
        test_helpers::exp_decay(&tvec, tau2),
        "first column must correspond to first model function: exp(-t/tau2)"
    );
    assert_eq!(
        DVector::from(eval_matrix.column(1)),
        test_helpers::exp_decay(&tvec, tau1),
        "second column must correspond to second model function: exp(-t/tau1)"
    );
    assert_eq!(
        DVector::from(eval_matrix.column(2)),
        DVector::from_element(tvec.len(), 1.),
        "third column must be vector of ones"
    );
}

#[test]
// test that when a base function does not return the same length result as its location argument,
// then the eval method fails
fn model_function_eval_fails_for_invalid_length_of_return_value_in_base_function() {
    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let model_with_bad_function = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], test_helpers::exp_decay)
        .partial_deriv("tau2", test_helpers::exp_decay_dtau)
        .function(&["tau1"], |_t: &DVector<_>, _tau| {
            DVector::from(vec![1., 3., 3., 7.])
        })
        .partial_deriv("tau1", test_helpers::exp_decay_dtau)
        .initial_parameters(vec![2.,4.])
        .independent_variable(tvec)
        .build()
        .expect("Model function creation should not fail, although function is bad");

    assert_matches!(model_with_bad_function.eval(),Err(ModelError::UnexpectedFunctionOutput{actual_length:4,..}),"Model must report an error when evaluated with a function that does not return the same length vector as independent variable");
}

#[test]
fn model_function_eval_fails_for_incorrect_number_of_model_parameters() {
    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let params = vec![1., 2., 3., 4., 5.];
    let model = test_helpers::get_double_exponential_model_with_constant_offset(tvec,params);
    assert_eq!(
        model.parameter_count(),
        2,
        "double exponential model should have 2 params"
    );
    // now deliberately provide a wrong number of params to eval
    assert_matches!(
        model.eval(),
        Err(ModelError::IncorrectParameterCount { .. })
    );
}

#[test]
// test that the correct derivative matrices are produced for a valid model
fn model_derivative_evaluation_produces_correct_result() {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let tau = 3.;
    let omega = 1.5;
    let params = &[tau, omega];

    let model = SeparableModelBuilder::<f64>::new(&["tau", "omega"])
        .independent_variable(tvec.clone())
        .initial_parameters(params.to_vec())
        .function(&["tau"], test_helpers::exp_decay)
        .partial_deriv("tau", test_helpers::exp_decay_dtau)
        .invariant_function(ones)
        .function(&["omega", "tau"], test_helpers::sin_ometa_t_plus_phi) // so we make phi=tau of the model. Bit silly, but to produce a function that contributes to all partial derivs
        .partial_deriv("tau", test_helpers::sin_ometa_t_plus_phi_dphi)
        .partial_deriv("omega", test_helpers::sin_ometa_t_plus_phi_domega)
        .build()
        .expect("Valid model creation should not fail");


    let deriv_tau = model
        .eval_partial_deriv(0)
        .expect("Derivative eval must not fail");
    let deriv_omega = model
        .eval_partial_deriv(1)
        .expect("Derivative eval must not fail");

    // DERIVATIVE WITH RESPECT TO TAU
    // assert that the matrix has the correct dimenstions
    assert!(
        deriv_tau.ncols() == 3 && deriv_tau.nrows() == tvec.len(),
        "Deriv tau matrix does not have correct dimensions"
    );
    // now check the columns of the deriv with respect to tau
    // d/d(tau) exp(-t/tau)
    assert_eq!(
        DVector::from(deriv_tau.column(0)),
        test_helpers::exp_decay_dtau(&tvec, tau)
    );
    // d/d(tau) constant function = 0
    assert_eq!(
        DVector::from(deriv_tau.column(1)),
        DVector::from_element(tvec.len(), 0.)
    );
    // d/d(tau) sin(omega*t+tau)
    assert_eq!(
        DVector::from(deriv_tau.column(2)),
        test_helpers::sin_ometa_t_plus_phi_dphi(&tvec, omega, tau)
    );

    // DERIVATIVE WITH RESPECT TO OMEGA
    assert!(
        deriv_omega.ncols() == 3 && deriv_omega.nrows() == tvec.len(),
        "Deriv omega matrix does not have correct dimensions"
    );
    // d/d(omega) exp(-t/tau) = 0
    assert_eq!(
        DVector::from(deriv_omega.column(0)),
        DVector::from_element(tvec.len(), 0.)
    );
    // d/d(omega) constant function = 0
    assert_eq!(
        DVector::from(deriv_omega.column(1)),
        DVector::from_element(tvec.len(), 0.)
    );
    // d/d(omega) sin(omega*t+tau)
    assert_eq!(
        DVector::from(deriv_omega.column(2)),
        test_helpers::sin_ometa_t_plus_phi_domega(&tvec, omega, tau)
    );
}

#[test]
// check that the following error cases are covered and return the appropriate errors
// * derivative is evaluated and includes a function that does not give the same length vector
// back as the location (x) argument
// * a derivative for a parameter that is not in the model is requested (by index or by name)
// * the derivative is requested for a wrong number of parameter arguments
fn model_derivative_evaluation_error_cases() {
    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let mut model_with_bad_function = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .independent_variable(tvec)
        .function(&["tau2"], test_helpers::exp_decay)
        .partial_deriv("tau2", test_helpers::exp_decay_dtau)
        .function(&["tau1"], test_helpers::exp_decay)
        .partial_deriv("tau1", |_t: &DVector<_>, _tau| {
            DVector::from(vec![1., 3., 3., 7.])
        })
        .initial_parameters(vec![2.,4.])
        .build()
        .expect("Model function creation should not fail, although function is bad");


    // deriv index 0 is tau1: this derivative is bad and should fail
    assert_matches!(
            model_with_bad_function.eval_partial_deriv( 0),
            Err(ModelError::UnexpectedFunctionOutput { .. })
        ,
        "Derivative for invalid function must fail with correct error"
    );

    // deriv index 0 is tau1: this derivative is good and should return an ok result
    assert!(
        model_with_bad_function
            .eval_partial_deriv(  1)
            .is_ok(),
        "Derivative eval for valid function should return Ok result"
    );

    // check an out of bounds index for the derivative
    assert_matches!(
            model_with_bad_function.eval_partial_deriv( 100),
            Err(ModelError::DerivativeIndexOutOfBounds { .. })
        ,
        "Derivative for invalid function must fail with correct error"
    );

    assert_matches!(
            model_with_bad_function.eval_partial_deriv( 3),
            Err(ModelError::DerivativeIndexOutOfBounds { .. })
        ,
        "Derivative for invalid function must fail with correct error"
    );

    // check that if an incorrect amount of parameters is provided, then the evaluation fails
    let _ = model_with_bad_function.set_params(&[1.,2.,3.,4.]);
    assert_matches!(
            model_with_bad_function.eval_partial_deriv( 1),
            Err(ModelError::IncorrectParameterCount { .. })
        ,
        "Derivative for invalid function must fail with correct error"
    );
}
