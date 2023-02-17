use crate::solvers::levmar::builder::LevMarBuilderError;
use crate::solvers::levmar::weights::Weights;
use crate::solvers::levmar::LevMarProblemBuilder;
use crate::{linalg_helpers::DiagDMatrix, model::test::DummySeparableModel};
use nalgebra::{DMatrix, DVector};
use assert_matches::assert_matches;

#[test]
fn new_builder_starts_with_empty_fields() {
    let model = DummySeparableModel::default();
    let builder = LevMarProblemBuilder::<f64, _>::new(model);
    let LevMarProblemBuilder {
        x,
        y,
        separable_model: _model,
        parameter_initial_guess,
        epsilon,
        weights,
    } = builder;
    assert!(x.is_none());
    assert!(y.is_none());
    assert!(parameter_initial_guess.is_none());
    assert!(epsilon.is_none());
    assert_eq!(weights, Weights::Unit);
}

#[test]
#[allow(clippy::float_cmp)] //clippy moans, but it's wrong (again!)
#[allow(non_snake_case)]
fn builder_assigns_fields_correctly() {
    let model = DummySeparableModel::default();
    // octave x = linspace(0,10,11);
    let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let initial_guess = vec![1., 2.];

    // build a problem with default epsilon
    let builder = LevMarProblemBuilder::new(model)
        .x(x.clone())
        .y(y.clone())
        .initial_guess(initial_guess.as_slice());
    let problem = builder
        .clone()
        .build()
        .expect("Valid builder should not fail build");

    assert_eq!(problem.x, x);
    assert_eq!(problem.y_w, y);
    assert_eq!(problem.model_parameters, initial_guess);
    //assert!(problem.model.as_ref().as_ptr()== model.as_ptr()); // this don't work, boss
    assert_eq!(problem.svd_epsilon, f64::EPSILON); //clippy moans, but it's wrong (again!)
    assert!(
        problem.cached.as_ref().unwrap().current_svd.u.is_some(),
        "SVD U must have been calculated on initialization"
    );
    assert!(
        problem.cached.as_ref().unwrap().current_svd.v_t.is_some(),
        "SVD V^T must have been calculated on initialization"
    );

    // now check that the given epsilon is also passed correctly to the model
    // and also that the weights are correctly passed and used to weigh the original data
    let weights = 2. * &y;
    let W = DMatrix::from_diagonal(&weights);

    let problem = builder
        .epsilon(-1.337) // check that negative values are converted to absolutes
        .weights(weights.clone())
        .build()
        .expect("Valid builder should not fail");
    assert_eq!(problem.svd_epsilon, 1.337);
    assert_eq!(
        problem.y_w,
        &W * &y,
        "Data must be correctly weighted with weights"
    );
    if let Weights::Diagonal(diag) = problem.weights {
        assert_eq!(
            diag,
            DiagDMatrix::from(weights),
            "Diagonal weight matrix must be correctly passed on"
        );
    } else {
        panic!("Simple weights call must produce diagonal weight matrix!");
    }
}

#[test]
fn builder_gives_errors_for_missing_mandatory_parameters() {
    let model = DummySeparableModel::default();
    // octave x = linspace(0,10,11);
    let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let initial_guess = vec![1., 2.];

    let _builder = LevMarProblemBuilder::new(model)
        .x(x.clone())
        .y(y.clone())
        .initial_guess(initial_guess.as_slice());

    assert_matches!(
        LevMarProblemBuilder::new(model)
            .y(y.clone())
            .initial_guess(initial_guess.as_slice())
            .build(),
        Err(LevMarBuilderError::XDataMissing)
    );
    assert_matches!(
        LevMarProblemBuilder::new(model)
            .x(x.clone())
            .initial_guess(initial_guess.as_slice())
            .build(),
        Err(LevMarBuilderError::YDataMissing)
    );
    assert_matches!(
        LevMarProblemBuilder::new(model).x(x).y(y).build(),
        Err(LevMarBuilderError::InitialGuessMissing)
    );
}

#[test]
fn builder_gives_errors_for_semantically_wrong_parameters() {
    let model = DummySeparableModel::default();
    // octave x = linspace(0,10,11);
    let x = DVector::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let _initial_guess = vec![1., 2.];

    assert_matches!(
            LevMarProblemBuilder::new(model)
                .x(x.clone())
                .y(y.clone())
                .initial_guess(&[1.])
                .build(),
            Err(LevMarBuilderError::InvalidParameterCount { .. })
        ,
        "invalid parameter count must produce correct error"
    );

    assert_matches!(
            LevMarProblemBuilder::new(model)
                .x(DVector::from(vec! {1.,2.,3.}))
                .y(y.clone())
                .initial_guess(&[1., 2.])
                .build(),
            Err(LevMarBuilderError::InvalidLengthOfData { .. })
        ,
        "invalid parameter count must produce correct error"
    );

    assert_matches!(
            LevMarProblemBuilder::new(model)
                .x(DVector::from(Vec::<f64>::new()))
                .y(DVector::from(Vec::<f64>::new()))
                .initial_guess(&[1., 2.])
                .build(),
            Err(LevMarBuilderError::ZeroLengthVector)
        ,
        "zero parameter count must produce correct error"
    );

    assert_matches!(
            LevMarProblemBuilder::new(model)
                .x(x)
                .y(y)
                .initial_guess(&[1., 2.])
                .weights(vec! {1.,2.,3.})
                .build(),
            Err(LevMarBuilderError::InvalidLengthOfWeights { .. })
        ,
        "invalid length of weights"
    );
}

