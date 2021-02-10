use super::*;
use crate::test_helpers::{
    exponential_decay, exponential_decay_dt0, exponential_decay_dtau, sinusoid_omega,
    sinusoid_omega_domega,
};
use nalgebra::U11;

#[test]
// creating obviously invalid models must fail and return the correct error, i.e.
// * models without parameters
// * models with duplicate parameters
// * models with valid parameters, but without functions
fn builder_fails_for_invalid_model_parameters() {
    let result = SeparableModelBuilder::<f64, U11>::new(&[
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
    ])
    .build();
    assert!(
        matches! {result, Err(ModelError::DuplicateParameterNames {..})},
        "Duplicate parameter error must be emitted when creating model with duplicate params"
    );

    let result = SeparableModelBuilder::<f64, U11>::new(&Vec::<String>::default()).build();
    assert!(
        matches! {result, Err(ModelError::EmptyParameters {..})},
        "Creating model with empty parameters must fail with correct error"
    );

    let result = SeparableModelBuilder::<f64, U11>::new(&[
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ])
    .build();
    assert!(
        matches! {result, Err(ModelError::EmptyModel {..})},
        "Creating model without functions must fail with correct error"
    );
}

#[test]
// test that the builder fails when the model depends on more parameters that the
// collection of function depends on
fn builder_fails_when_not_all_model_parameters_are_depended_on_by_the_modelfunctions() {
    let result = SeparableModelBuilder::<f64, U11>::new(&[
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
    ])
    .invariant_function(|_| unimplemented!())
    .function(&["a".to_string()], |_, _| unimplemented!())
    .partial_deriv("a", |_, _| unimplemented!())
    .build();
    assert!(
        matches! {result, Err(ModelError::UnusedParameter {..})},
        "Duplicate parameter error must be emitted when creating model with duplicate params"
    );
}

#[test]
// test that the builder fails when not all required derivatives are given for a function
fn builder_fails_when_not_all_required_partial_derivatives_are_given_for_function() {
    let result = SeparableModelBuilder::<f64, U11>::new(&["a".to_string(), "b".to_string()])
        .function(&["a".to_string(), "b".to_string()], |_, _| unimplemented!())
        .partial_deriv("a", |_, _| unimplemented!())
        .build();
    assert!(
        matches! {result, Err(ModelError::MissingDerivative {..})},
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
    let model = SeparableModelBuilder::<f64, Dynamic>::new(&[
        "t0".to_string(),
        "tau".to_string(),
        "omega".to_string(),
    ])
        .invariant_function(|x|2.*identity_function(x))// double the x value
    .function(&["t0".to_string(), "tau".to_string()], |x, params| {
        exponential_decay(x, params[0], params[1])
    })
    .partial_deriv("tau", |x, params| {
        exponential_decay_dtau(x, params[0], params[1])
    })
    .partial_deriv("t0", |x, params| {
        exponential_decay_dt0(x, params[0], params[1])
    })
    .invariant_function(identity_function)
    .function(&["omega".to_string()], |x, params| {
        sinusoid_omega(x, params[0])
    })
    .partial_deriv("omega", |x, params| sinusoid_omega_domega(x, params[0]))
    .build()
    .expect("Valid builder calls should produce a valid model function.");

    // now check that each function behaves as expected given the model parameters
    let ts = OwnedVector::<f64, Dynamic>::from(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
    ]);
    let t0 = 3.;
    let tau = 2.;
    let omega = std::f64::consts::FRAC_1_PI * 3.;

    let params = OwnedVector::<f64, Dynamic>::from(vec!{t0,tau,omega});

    // assert that the correct number of functions is in the set
    assert_eq!(
        model.modelfunctions.len(),
        4,
        "Number of functions in model is incorrect"
    );

    // check the first function f(t) = 2t
    let func = &model.modelfunctions[0];
    assert!(func.derivatives.is_empty(), "This function should have no derivatives");
    assert_eq!((func.function)(&ts,&params),2.* ts.clone(),"Function should be f(x)=2x");

    // check the second function f(t,t0,tau) = exp( -(t-t0)/tau )
    let func = &model.modelfunctions[1];
    assert_eq!(func.derivatives.len(),2, "Incorrect number of derivatives");
    assert_eq!((func.function)(&ts,&params),exponential_decay(&ts,t0,tau),"Incorrect function value");
    assert_eq!((func.derivatives.get(&0).unwrap())(&ts,&params),exponential_decay_dt0(&ts,t0,tau),"Incorrect first derivative value");
    assert_eq!((func.derivatives.get(&1).unwrap())(&ts,&params),exponential_decay_dtau(&ts,t0,tau),"Incorrect second derivative value");

    // check that the third function is f(t) = t
    let func = &model.modelfunctions[2];
    assert!(func.derivatives.is_empty(), "This function should have no derivatives");
    assert_eq!((func.function)(&ts,&params), ts.clone(),"Function should be f(x)=2x");

    // check that the fourth funciton is f(t) = sin(omega*t)
    let func = &model.modelfunctions[3];
    assert_eq!(func.derivatives.len(),1, "Incorrect number of derivatives");
    assert_eq!((func.function)(&ts,&params),sinusoid_omega(&ts,omega),"Incorrect function value");
    assert_eq!((func.derivatives.get(&2).unwrap())(&ts,&params),sinusoid_omega_domega(&ts,omega),"Incorrect first derivative value");
}
