use nalgebra::DVector;
use assert_matches::assert_matches;
use super::*;

#[test]
// creating obviously invalid models must fail and return the correct error, i.e.
// * models without parameters
// * models with duplicate parameters
// * models with valid parameters, but without functions
fn builder_fails_for_invalid_model_parameters() {
    let result =
        SeparableModelBuilder::<f64>::new(&["a".to_string(), "b".to_string(), "b".to_string()])
            .build();
    assert_matches!(
         result, Err(ModelBuildError::DuplicateParameterNames {..}),
        "Duplicate parameter error must be emitted when creating model with duplicate params"
    );

    let result = SeparableModelBuilder::<f64>::new(&Vec::<String>::default()).build();
    assert_matches!(
        result, Err(ModelBuildError::EmptyParameters {..}),
        "Creating model with empty parameters must fail with correct error"
    );

    let result =
        SeparableModelBuilder::<f64>::new(&["a".to_string(), "b".to_string(), "c".to_string()])
            .build();
    assert_matches!(
        result, Err(ModelBuildError::EmptyModel {..}),
        "Creating model without functions must fail with correct error"
    );
}

#[test]
// test that the builder fails when the model depends on more parameters that the
// collection of function depends on
fn builder_fails_when_not_all_model_parameters_are_depended_on_by_the_modelfunctions() {
    let result =
        SeparableModelBuilder::<f64>::new(&["a".to_string(), "b".to_string(), "c".to_string()])
            .invariant_function(|_| unimplemented!())
            .function(
                &["a".to_string()],
                |_: &DVector<f64>, _: f64| unimplemented!(),
            )
            .partial_deriv("a", |_: &DVector<f64>, _: f64| unimplemented!())
            .build();
    assert_matches!(
        result, Err(ModelBuildError::UnusedParameter {..}),
        "Duplicate parameter error must be emitted when creating model with duplicate params"
    );
}

#[test]
// test that the builder fails when not all required derivatives are given for a function
fn builder_fails_when_not_all_required_partial_derivatives_are_given_for_function() {
    let result = SeparableModelBuilder::<f64>::new(&["a".to_string(), "b".to_string()])
        .function(
            &["a".to_string(), "b".to_string()],
            |_: &DVector<f64>, _: f64, _: f64| unimplemented!(),
        )
        .partial_deriv("a", |_: &DVector<f64>, _: f64, _: f64| unimplemented!())
        .build();
    assert_matches!(
        result, Err(ModelBuildError::MissingDerivative {..}),
        "Duplicate parameter error must be emitted when creating model with duplicate params"
    );
}

fn identity_function<T: Clone>(x: &T) -> T {
    x.clone()
}

#[test]
// test that the builder correctly produces a model with functions with and without derivatives,
// when the parameters and functions are valid
fn builder_produces_correct_model_from_functions() {
    let model = SeparableModelBuilder::<f64>::new(&[
        "t0".to_string(),
        "tau".to_string(),
        "omega1".to_string(),
        "omega2".to_string(),
    ])
    .invariant_function(|x| 2. * identity_function(x)) // double the x value
    .function(&["t0".to_string(), "tau".to_string()], exponential_decay)
    .partial_deriv("tau", exponential_decay_dtau)
    .partial_deriv("t0", exponential_decay_dt0)
    .invariant_function(identity_function)
    .function(&["omega1".to_string()], sinusoid_omega)
    .partial_deriv("omega1", sinusoid_omega_domega)
    .function(&["omega2".to_string()], sinusoid_omega)
    .partial_deriv("omega2", sinusoid_omega_domega)
    .build()
    .expect("Valid builder calls should produce a valid model function.");

    // now check that each function behaves as expected given the model parameters
    let ts = DVector::<f64>::from(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
    ]);
    let t0 = 3.;
    let tau = 2.;
    let omega1 = std::f64::consts::FRAC_1_PI * 3.;
    let omega2 = std::f64::consts::FRAC_1_PI * 2.;

    let params = vec![t0, tau, omega1, omega2];

    // assert that the correct number of functions is in the set
    assert_eq!(
        model.basefunctions.len(),
        5,
        "Number of functions in model is incorrect"
    );

    // check the first function f(t) = 2t
    let func = &model.basefunctions[0];
    assert!(
        func.derivatives.is_empty(),
        "This function should have no derivatives"
    );
    assert_eq!(
        (func.function)(&ts, &params),
        2. * ts.clone(),
        "Function should be f(x)=2x"
    );

    // check the second function f(t,t0,tau) = exp( -(t-t0)/tau )
    let func = &model.basefunctions[1];
    assert_eq!(func.derivatives.len(), 2, "Incorrect number of derivatives");
    assert_eq!(
        (func.function)(&ts, &params),
        exponential_decay(&ts, t0, tau),
        "Incorrect function value"
    );
    assert_eq!(
        (func.derivatives.get(&0).unwrap())(&ts, &params),
        exponential_decay_dt0(&ts, t0, tau),
        "Incorrect first derivative value"
    );
    assert_eq!(
        (func.derivatives.get(&1).unwrap())(&ts, &params),
        exponential_decay_dtau(&ts, t0, tau),
        "Incorrect second derivative value"
    );

    // check that the third function is f(t) = t
    let func = &model.basefunctions[2];
    assert!(
        func.derivatives.is_empty(),
        "This function should have no derivatives"
    );
    assert_eq!(
        (func.function)(&ts, &params),
        ts.clone(),
        "Function should be f(x)=2x"
    );

    // check that the fourth function is f(t) = sin(omega1*t)
    let func = &model.basefunctions[3];
    assert_eq!(func.derivatives.len(), 1, "Incorrect number of derivatives");
    assert_eq!(
        (func.function)(&ts, &params),
        sinusoid_omega(&ts, omega1),
        "Incorrect function value"
    );
    assert_eq!(
        (func.derivatives.get(&2).unwrap())(&ts, &params),
        sinusoid_omega_domega(&ts, omega1),
        "Incorrect first derivative value"
    );

    // check that the fifth function is f(t) = sin(omega2*t)
    let func = &model.basefunctions[4];
    assert_eq!(func.derivatives.len(), 1, "Incorrect number of derivatives");
    assert_eq!(
        (func.function)(&ts, &params),
        sinusoid_omega(&ts, omega2),
        "Incorrect function value"
    );
    assert_eq!(
        (func.derivatives.get(&3).unwrap())(&ts, &params),
        sinusoid_omega_domega(&ts, omega2),
        "Incorrect first derivative value"
    );
}

/// a function that calculates exp( -(t-t0)/tau)) for every location t
/// **ATTENTION** for this kind of exponential function the shift will
/// just be a linear multiplier exp(t0/tau), so it might not a good idea to include it in the fitting
pub fn exponential_decay(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| (-(t - t0) / tau).exp())
}

/// partial derivative of the exponential decay with respect to t0
pub fn exponential_decay_dt0(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    exponential_decay(tvec, t0, tau).map(|val| val / tau)
}

/// partial derivative of the exponential decay with respect to tau
pub fn exponential_decay_dtau(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((-t - t0) / tau).exp() * (t0 - t) / tau.powi(2))
}

/// implements the function sin(omega*t) for argument t
pub fn sinusoid_omega(tvec: &DVector<f64>, omega: f64) -> DVector<f64> {
    tvec.map(|t| (omega * t).sin())
}

/// implements the derivative function d/domega sin(omega*t) = omega * cos(omega*t) for argument t
pub fn sinusoid_omega_domega(tvec: &DVector<f64>, omega: f64) -> DVector<f64> {
    tvec.map(|t| (omega * t).cos() * omega)
}
