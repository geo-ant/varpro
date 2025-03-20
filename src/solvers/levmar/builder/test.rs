use crate::model::test::MockSeparableNonlinearModel;
use crate::solvers::levmar::builder::LevMarBuilderError;
use crate::solvers::levmar::LevMarProblemBuilder;
use crate::util::DiagMatrix;
use crate::util::Weights;
use assert_matches::assert_matches;
use nalgebra::{DMatrix, DVector};

#[test]
fn new_builder_starts_with_empty_fields() {
    let model = MockSeparableNonlinearModel::default();
    let builder = LevMarProblemBuilder::new(model);
    let LevMarProblemBuilder {
        Y: y,
        separable_model: _model,
        epsilon,
        weights,
        phantom: _,
    } = builder;
    assert!(y.is_none());
    assert!(epsilon.is_none());
    assert_eq!(weights, Weights::Unit);
}

#[test]
#[allow(clippy::float_cmp)] //clippy moans, but it's wrong (again!)
#[allow(non_snake_case)]
fn builder_assigns_fields_correctly_simple_case() {
    let mut model = MockSeparableNonlinearModel::default();
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let y_len = y.len();
    let params_array = [1., 2., 3.];
    let params_vector = DVector::from_column_slice(&params_array);
    model.expect_output_len().return_const(y_len);
    model.expect_params().return_const(params_vector.clone());
    model
        .expect_set_params()
        .withf(move |p| p == &params_vector.clone())
        .returning(|_| Ok(()));
    model
        .expect_eval()
        .returning(move || Ok(DMatrix::zeros(y_len, y_len))); // the returned matrix eval is not used in this test

    // build a problem with default epsilon
    let builder = LevMarProblemBuilder::new(model).observations(y.clone());
    let problem = builder
        .build()
        .expect("Valid builder should not fail build");

    assert_eq!(problem.Y_w, y);
    assert_eq!(problem.svd_epsilon, f64::EPSILON); //clippy moans, but it's wrong (again!)
    assert!(
        problem.cached.is_some(),
        "cached calculations are assigned when problem is built"
    );
}

#[test]
#[allow(clippy::float_cmp)] //clippy moans, but it's wrong (again!)
#[allow(non_snake_case)]
fn builder_assigns_fields_correctly_with_weights_and_epsilon() {
    let mut model = MockSeparableNonlinearModel::default();

    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);

    let y_len = y.len();
    let params_array = [1., 2., 3.];
    let params_vector = DVector::from_column_slice(&params_array);
    model.expect_output_len().return_const(y_len);
    model.expect_params().return_const(params_vector.clone());
    model
        .expect_set_params()
        .withf(move |p| p == &params_vector.clone())
        .returning(|_| Ok(()));
    model
        .expect_eval()
        .returning(move || Ok(DMatrix::zeros(y_len, y_len))); // the returned matrix eval is not used in this test
                                                              // now check that the given epsilon is also passed correctly to the model
                                                              // and also that the weights are correctly passed and used to weigh the original data
    let weights = 2. * &y;
    let W = DMatrix::from_diagonal(&weights);

    let problem = LevMarProblemBuilder::new(model)
        .observations(y.clone())
        .epsilon(-1.337) // check that negative values are converted to absolutes
        .weights(weights.clone())
        .build()
        .expect("Valid builder should not fail");
    assert_eq!(problem.svd_epsilon, 1.337);
    assert_eq!(
        problem.Y_w,
        &W * &y,
        "Data must be correctly weighted with weights"
    );
    if let Weights::Diagonal(diag) = problem.weights {
        assert_eq!(
            diag,
            DiagMatrix::from(weights),
            "Diagonal weight matrix must be correctly passed on"
        );
    } else {
        panic!("Simple weights call must produce diagonal weight matrix!");
    }
}

#[test]
fn builder_gives_errors_for_missing_y_data() {
    let model = MockSeparableNonlinearModel::default();

    assert_matches!(
        LevMarProblemBuilder::new(model).build(),
        Err(LevMarBuilderError::YDataMissing)
    );
}

#[test]
fn builder_gives_errors_for_wrong_data_length() {
    let mut model = MockSeparableNonlinearModel::default();
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let _initial_guess = [1., 2.];

    let wrong_output_len = y.len() - 1;
    model
        .expect_output_len()
        .returning(move || wrong_output_len);

    assert_matches!(
        LevMarProblemBuilder::new(model).observations(y).build(),
        Err(LevMarBuilderError::InvalidLengthOfData { .. }),
        "invalid parameter count must produce correct error"
    );
}

#[test]
fn builder_gives_errors_for_zero_length_data() {
    let mut model = MockSeparableNonlinearModel::default();
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let _initial_guess = [1., 2.];

    let output_len = y.len();
    model.expect_output_len().returning(move || output_len);

    assert_matches!(
        LevMarProblemBuilder::new(model)
            .observations(DVector::from(Vec::<f64>::new()))
            .build(),
        Err(LevMarBuilderError::ZeroLengthVector),
        "zero parameter count must produce correct error"
    );
}

#[test]
fn builder_gives_errors_for_wrong_length_of_weights() {
    let mut model = MockSeparableNonlinearModel::default();
    //octave y = 2*exp(-t/2)+exp(-t/4)+1;
    let y = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227, 1.2342, 1.1720, 1.1276, 1.0956,
    ]);
    let _initial_guess = [1., 2.];

    let output_len = y.len();
    model.expect_output_len().returning(move || output_len);

    assert_matches!(
        LevMarProblemBuilder::new(model)
            .observations(y)
            .weights(DVector::from_vec(vec! {1.,2.,3.}))
            .build(),
        Err(LevMarBuilderError::InvalidLengthOfWeights { .. }),
        "invalid length of weights"
    );
}
