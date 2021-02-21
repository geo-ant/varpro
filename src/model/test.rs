//use crate::model::SeparableModel;

use crate::model::builder::SeparableModelBuilder;
use crate::model::errors::ModelError;
use crate::model::SeparableModel;
use nalgebra::{DVector, Scalar};
use num_traits::Float;

/// exponential decay f(t,tau) = exp(-t/tau)
fn exp_decay<ScalarType: Float + Scalar>(
    tvec: &DVector<ScalarType>,
    tau: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (-t / tau).exp())
}

/// derivative of exp decay with respect to tau
fn exp_decay_dtau<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    tau: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (-t / tau).exp() * t / (tau * tau))
}

/// function sin (omega*t+phi)
fn sin_ometa_t_plus_phi<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (omega * t + phi).sin())
}

/// derivative d/d(omega) sin (omega*t+phi)
fn sin_ometa_t_plus_phi_domega<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| t * (omega * t + phi).cos())
}

/// derivative d/d(phi) sin (omega*t+phi)
fn sin_ometa_t_plus_phi_dphi<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (omega * t + phi).cos())
}

/// A helper function that returns a double exponential decay model
/// f(x,tau1,tau2) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
fn get_double_exponential_model_with_constant_offset() -> SeparableModel<f64> {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .invariant_function(ones)
        .build()
        .expect("double exponential model builder should produce a valid model")
}

#[test]
// test that the eval method produces correct results and gives a matrix that is ordered correctly
fn model_function_eval_produces_correct_result() {
    let model = get_double_exponential_model_with_constant_offset();

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let tau1 = 1.;
    let tau2 = 3.;

    let params = &[tau1, tau2];
    let eval_matrix = model
        .matrix_eval(&tvec, params)
        .expect("Model evaluation should not fail");

    assert_eq!(
        DVector::from(eval_matrix.column(0)),
        exp_decay(&tvec, tau2),
        "first column must correspond to first model function: exp(-t/tau2)"
    );
    assert_eq!(
        DVector::from(eval_matrix.column(1)),
        exp_decay(&tvec, tau1),
        "second column must correspond to second model function: exp(-t/tau1)"
    );
    assert_eq!(
        DVector::from(eval_matrix.column(2)),
        DVector::from_element(tvec.len(), 1.),
        "third column must be vector of ones"
    );
}

#[test]
#[ignore]
// test that when a base function does not return the same length result as its location argument,
// then the eval method fails
fn model_function_eval_fails_for_invalid_length_of_return_value_in_base_function() {
    let model_with_bad_function = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau1"], |_t: &DVector<_>, _tau| {
            DVector::from(vec![1., 3., 3., 7.])
        })
        .partial_deriv("tau1", exp_decay_dtau)
        .build()
        .expect("Model function creation should not fail, although function is bad");

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

    assert!(matches!(model_with_bad_function.matrix_eval(&tvec,&[2.,4.]),Err(ModelError::UnexpectedFunctionOutput{actual_length:3,..})),"Model must report an error when evaluated with a function that does not return the same length vector as independent variable");
}

#[test]
// test that the correct derivative matrices are produced for a valid model
fn model_derivative_evaluation_produces_correct_result() {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    let model = SeparableModelBuilder::<f64>::new(&["tau", "omega"])
        .function(&["tau"], exp_decay)
        .partial_deriv("tau", exp_decay_dtau)
        .invariant_function(ones)
        .function(&["omega", "tau"], sin_ometa_t_plus_phi) // so we make phi=tau of the model. Bit silly, but to produce a function that contributes to all partial derivs
        .partial_deriv("tau", sin_ometa_t_plus_phi_dphi)
        .partial_deriv("omega", sin_ometa_t_plus_phi_domega)
        .build()
        .expect("Valid model creation should not fail");

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let tau = 3.;
    let omega = 1.5;
    let params = &[tau,omega];


    let deriv_tau = model.matrix_deriv(&tvec,params).eval_at_param_name("tau").expect("Derivative eval must not fail");
    let deriv_omega = model.matrix_deriv(&tvec,params).eval_at_param_name("omega").expect("Derivative eval must not fail");

    // DERIVATIVE WITH RESPECT TO TAU
    // assert that the matrix has the correct dimenstions
    assert!(deriv_tau.ncols() ==3 && deriv_tau.nrows() == tvec.len(), "Deriv tau matrix does not have correct dimensions");
    // now check the columns of the deriv with respect to tau
    // d/d(tau) exp(-t/tau)
    assert_eq!(DVector::from(deriv_tau.column(0)),exp_decay_dtau(&tvec,tau));
    // d/d(tau) constant function = 0
    assert_eq!(DVector::from(deriv_tau.column(1)),DVector::from_element(tvec.len(),0.));
    // d/d(tau) sin(omega*t+tau)
    assert_eq!(DVector::from(deriv_tau.column(2)),sin_ometa_t_plus_phi_dphi(&tvec,omega,tau));

    // DERIVATIVE WITH RESPECT TO OMEGA
    assert!(deriv_omega.ncols() ==3 && deriv_omega.nrows() == tvec.len(), "Deriv omega matrix does not have correct dimensions");
    // d/d(omega) exp(-t/tau) = 0
    assert_eq!(DVector::from(deriv_omega.column(0)),DVector::from_element(tvec.len(),0.));
    // d/d(omega) constant function = 0
    assert_eq!(DVector::from(deriv_omega.column(1)),DVector::from_element(tvec.len(),0.));
    // d/d(omega) sin(omega*t+tau)
    assert_eq!(DVector::from(deriv_omega.column(2)),sin_ometa_t_plus_phi_domega(&tvec,omega,tau));
}

#[test]
fn requesting_derivative_by_name_and_index_produces_same_results() {
    let model = get_double_exponential_model_with_constant_offset();
    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let params = &[2., 4.];

    assert_eq!(
        model.matrix_deriv(&tvec, params).eval_at(0).unwrap(),
        model
            .matrix_deriv(&tvec, params)
            .eval_at_param_name("tau1")
            .unwrap(),
        "Evaluating derivative by name or index must produce same result"
    );

    assert_eq!(
        model.matrix_deriv(&tvec, params).eval_at(1).unwrap(),
        model
            .matrix_deriv(&tvec, params)
            .eval_at_param_name("tau2")
            .unwrap(),
        "Evaluating derivative by name or index must produce same result"
    );
}

#[test]
// check that the following error cases are covered and return the appropriate errors
// * derivative is evaluated and includes a function that does not give the same length vector
// back as the location (x) argument
// * a derivative for a parameter that is not in the model is requested (by index or by name)
// * the derivative is requested for a wrong number of parameter arguments
fn model_derivative_evaluation_error_cases() {
    let model_with_bad_function = SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", |_t: &DVector<_>, _tau| {
            DVector::from(vec![1., 3., 3., 7.])
        })
        .build()
        .expect("Model function creation should not fail, although function is bad");

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

    // deriv index 0 is tau1: this derivative is bad and should fail
    assert!(
        matches!(
            model_with_bad_function.matrix_deriv(&tvec, &[2., 4.]).eval_at(0),
            Err(ModelError::UnexpectedFunctionOutput { .. })
        ),
        "Derivative for invalid function must fail with correct error"
    );

    // deriv index 0 is tau1: this derivative is good and should return an ok result
    assert!(
        model_with_bad_function
            .matrix_deriv(&tvec, &[2., 4.])
            .eval_at(1)
            .is_ok(),
        "Derivative eval for valid function should return Ok result"
    );

    // check that if an incorrect amount of parameters is provided, then the evaluation fails
    assert!(
        matches!(
            model_with_bad_function
                .matrix_deriv(&tvec, &[2., 4., 2., 2.])
                .eval_at(1),
            Err(ModelError::IncorrectParameterCount { .. })
        ),
        "Derivative for invalid function must fail with correct error"
    );

    // check an out of bounds index for the derivative
    assert!(
        matches!(
            model_with_bad_function.matrix_deriv(&tvec, &[2., 4.]).eval_at(100),
            Err(ModelError::DerivativeIndexOutOfBounds { .. })
        ),
        "Derivative for invalid function must fail with correct error"
    );

    // check that if a nonexistent parameter is requested by name, then the derivative evaluation fails
    assert!(
        matches!(
            model_with_bad_function
                .matrix_deriv(&tvec, &[2., 4.])
                .eval_at_param_name("frankenstein"),
            Err(ModelError::ParameterNotInModel { .. })
        ),
        "Derivative for invalid function must fail with correct error"
    );
}
