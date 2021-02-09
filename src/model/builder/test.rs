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
    .function(&["t0".to_string(), "tau".to_string()], |x, params| {
        exponential_decay(x, params[0], params[1])
    })
    .partial_deriv("t0", |x, params| {
        exponential_decay_dt0(x, params[0], params[1])
    })
    .partial_deriv("tau", |x, params| {
        exponential_decay_dtau(x, params[0], params[1])
    })
    .invariant_function(identity_function)
    .function(&["omega".to_string()], |x, params| {
        sinusoid_omega(x, params[0])
    })
    .partial_deriv("omega", |x, params| sinusoid_omega_domega(x, params[0]))
    .build()
    .expect("Valid builder calls should produce a valid model function.");

    todo!("Make sure the model actually consists of the correct functions")
}
