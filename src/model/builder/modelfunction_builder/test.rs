use crate::model::builder::modelfunction_builder::ModelFunctionBuilder;
use crate::model::errors::ModelError;
use crate::model::OwnedVector;
use nalgebra::{Dim, Dynamic};

/// a function that calculates exp( (t-t0)/tau)) for every location t
fn exponential_decay<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((t - t0) / tau).exp())
}

/// partial derivative of the exponential decay with respect to t0
fn exponential_decay_dt0<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    exponential_decay(tvec, t0, tau).map(|val| val / tau)
}

/// partial derivative of the exponential decay with respect to tau
fn exponential_decay_dtau<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((t - t0) / tau).exp() * (t0 - t) / tau.powi(2))
}

#[test]
// check that the modelfunction builder assigns the function and derivatives correctly
// and that they can be called using the model parameters and produce the correct results
fn modelfunction_builder_creates_correct_modelfunction_with_valid_parameters() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "bar".to_string(),
        "tau".to_string(),
    ];
    let mf = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .partial_deriv("t0", |t, params| {
            exponential_decay_dt0(t, params[0], params[1])
        })
        .partial_deriv("tau", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .build()
        .expect("Modelfunction builder with valid parameters should not return an error");

    let t0 = 2.;
    let tau = 1.5;
    let model_params = OwnedVector::<f64, Dynamic>::from(vec![-2., t0, -1., tau]);
    let t =
        OwnedVector::<f64, Dynamic>::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    assert_eq!(
        (mf.function)(&t, &model_params),
        exponential_decay(&t, t0, tau),
        "Function must produce correct results"
    );
    assert_eq!(
        (mf.derivatives
            .get(&1)
            .expect("Derivative for t0 must be in set"))(&t, &model_params),
        exponential_decay_dt0(&t, t0, tau),
        "Derivative for t0 must produce correct results"
    );
    assert_eq!(
        (mf.derivatives
            .get(&3)
            .expect("Derivative for tau must be in set"))(&t, &model_params),
        exponential_decay_dtau(&t, t0, tau),
        "Derivative for tau must produce correct results"
    );
}

#[test]
// test that the modelfunction builder fails with invalid model paramters, i.e
// when the model parameters contain duplicates or are empty
fn modelfunction_builder_fails_with_invalid_model_parameters() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "foo".to_string(),
        "tau".to_string(),
    ];
    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters.clone(),
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert!(
        matches!(result, Err(ModelError::DuplicateParameterNames { .. })),
        "Modelfunction builder must indicate duplicate parameters!"
    );

    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        Vec::<String>::default(),
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .build();

    assert!(
        matches!(result, Err(ModelError::EmptyParameters)),
        "Builder must indicate error when model parameters are emtpy"
    );
}

#[test]
// test that the modelfunction builder fails with invalid model paramters, i.e
// when the model parameters contain duplicates or are empty
fn modelfunction_builder_fails_with_invalid_function_parameters() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "bar".to_string(),
        "tau".to_string(),
    ];
    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters.clone(),
        ["tau".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert!(
        matches!(result, Err(ModelError::DuplicateParameterNames { .. })),
        "Modelfunction builder must indicate duplicate parameters!"
    );

    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        Vec::<String>::default(),
        Vec::<String>::default().as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .build();

    assert!(
        matches!(result, Err(ModelError::EmptyParameters)),
        "Builder must indicate error when function parameters are emtpy"
    );
}

#[test]
// builder fails when a derivative is given that is not in the set of function parameters
// (although the parameter might be in the model)
// builder fails when not all derivatives have been provided
fn modelfunction_builder_fails_when_invalid_derivatives_are_given() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "bar".to_string(),
        "tau".to_string(),
    ];
    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters.clone(),
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .partial_deriv("bar", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert!(
        matches!(result, Err(ModelError::InvalidDerivative { .. })),
        "Modelfunction builder must when non-existing derivative is given for function!"
    );

    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .partial_deriv("tau", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .build();

    assert!(
        matches!(result, Err(ModelError::MissingDerivative { .. })),
        "Builder must indicate that one derivative is missing"
    );
}

// check that the modelfunction builder fails when duplicate derivatives are given
fn modelfunction_builder_fails_when_duplicate_derivatives_are_given() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "bar".to_string(),
        "tau".to_string(),
    ];
    let result = ModelFunctionBuilder::<f64, Dynamic>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        |t, params| exponential_decay(t, params[0], params[1]),
    )
        .partial_deriv("tau", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .partial_deriv("tau", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .build();

    assert!(
        matches!(result, Err(ModelError::DuplicateDerivative { .. })),
        "Builder must indicate that one derivative is missing"
    );
}

