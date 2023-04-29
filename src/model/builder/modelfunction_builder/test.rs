use crate::model::builder::error::ModelBuildError;
use crate::model::builder::modelfunction_builder::ModelBasisFunctionBuilder;
use crate::model::builder::test::{
    exponential_decay, exponential_decay_dt0, exponential_decay_dtau,
};
use assert_matches::assert_matches;
use nalgebra::DVector;
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
    let mf = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .partial_deriv("t0", exponential_decay_dt0)
    .partial_deriv("tau", exponential_decay_dtau)
    .build()
    .expect("Modelfunction builder with valid parameters should not return an error");

    let t0 = 2.;
    let tau = 1.5;
    let model_params = vec![-2., t0, -1., tau];
    let t = DVector::<f64>::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
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
    let result = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert_matches!(
        result,
        Err(ModelBuildError::DuplicateParameterNames { .. }),
        "Modelfunction builder must indicate duplicate parameters!"
    );

    let result = ModelBasisFunctionBuilder::<f64>::new(
        Vec::<String>::default(),
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .build();

    assert_matches!(
        result,
        Err(ModelBuildError::EmptyParameters),
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
    let result = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters,
        ["tau".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert_matches!(
        result,
        Err(ModelBuildError::DuplicateParameterNames { .. }),
        "Modelfunction builder must indicate duplicate parameters!"
    );

    let result = ModelBasisFunctionBuilder::<f64>::new(
        Vec::<String>::default(),
        Vec::<String>::default(),
        exponential_decay,
    )
    .build();

    assert_matches!(
        result,
        Err(ModelBuildError::EmptyParameters),
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
    let result = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters.clone(),
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .partial_deriv("bar", exponential_decay_dtau)
    .build();

    // the derivatives are also incomplete, but this should be the first recorded error
    assert_matches!(
        result,
        Err(ModelBuildError::InvalidDerivative { .. }),
        "Modelfunction builder must when non-existing derivative is given for function!"
    );

    let result = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .partial_deriv("tau", exponential_decay_dtau)
    .build();

    assert_matches!(
        result,
        Err(ModelBuildError::MissingDerivative { .. }),
        "Builder must indicate that one derivative is missing"
    );
}

// check that the modelfunction builder fails when duplicate derivatives are given
#[test]
fn modelfunction_builder_fails_when_duplicate_derivatives_are_given() {
    let model_parameters = vec![
        "foo".to_string(),
        "t0".to_string(),
        "bar".to_string(),
        "tau".to_string(),
    ];
    let result = ModelBasisFunctionBuilder::<f64>::new(
        model_parameters,
        ["t0".to_string(), "tau".to_string()].as_ref(),
        exponential_decay,
    )
    .partial_deriv("tau", exponential_decay_dtau)
    .partial_deriv("tau", exponential_decay_dtau)
    .build();

    assert_matches!(
        result,
        Err(ModelBuildError::DuplicateDerivative { .. }),
        "Builder must indicate that one derivative is missing"
    );
}
