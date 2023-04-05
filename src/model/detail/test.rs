use super::*;
use assert_matches::assert_matches;

// a dummy function that disregards the x argument and just returns the parameters
// useful to test if the wrapper has correctly distributed the parameters
fn dummy_unit_function_for_parameters<ScalarType>(
    _x: &DVector<ScalarType>,
    param1: ScalarType,
    param2: ScalarType,
) -> DVector<ScalarType>
where
    ScalarType: Scalar,
{
    DVector::from(vec![param1, param2])
}

// dummy function that just returns the given x
fn dummy_unit_function_for_x(x: &DVector<f64>, _param1: f64, _param2: f64) -> DVector<f64> {
    DVector::<f64>::clone(x)
}

#[test]
fn test_has_only_unique_elements() {
    assert!(!has_only_unique_elements(vec![10, 20, 30, 10, 50]));
    assert!(has_only_unique_elements(vec![10, 20, 30, 40, 50]));
    assert!(has_only_unique_elements(Vec::<u8>::new()));
}

// make sure the check parameter names function behaves as intended
#[test]
fn test_check_parameter_names() {
    assert_matches!(
        check_parameter_names(&Vec::<String>::default()),
        Err(ModelBuildError::EmptyParameters)
    );
    assert!(check_parameter_names(&["a"]).is_ok());
    assert!(check_parameter_names(&["a", "b", "c"]).is_ok());
    assert_matches!(
        check_parameter_names(&["a", "b", "b"]),
        Err(ModelBuildError::DuplicateParameterNames { .. })
    );
    assert_matches!(
        check_parameter_names(&["a,b", "c"]),
        Err(ModelBuildError::CommaInParameterNameNotAllowed { .. })
    );
}

#[test]
fn test_create_index_mapping() {
    let full_set = ['A', 'B', 'C', 'D'];
    assert_eq!(
        create_index_mapping(&full_set, &Vec::<char>::new()),
        Ok(Vec::new()),
        "Empty subset produces must produce empty index list"
    );
    assert!(
        create_index_mapping(&Vec::<char>::new(), &['B', 'A']).is_err(),
        "Empty full set must produce an error"
    );
    assert_eq!(
        create_index_mapping(&Vec::<char>::new(), &Vec::<char>::new()),
        Ok(Vec::new()),
        "Empty subset must produce empty index list even if full set is empty"
    );
    assert_eq!(
        create_index_mapping(&full_set, &['B', 'A']),
        Ok(vec! {1, 0}),
        "Indices must be correctly assigned"
    );
    assert!(
        create_index_mapping(&full_set, &['Z', 'Q']).is_err(),
        "Indices that are not in the full set must produce an error"
    );
    assert_eq!(
        create_index_mapping(&['A', 'A', 'B', 'D'], &['B', 'A']),
        Ok(vec! {2, 0}),
        "For duplicates in the full set, the first index is used"
    );
}

#[test]
// test that the creation of a wrapped basis function fails if
// * duplicate paramters are in the function parameter list
// * function parameter list is empty
// * function parameter list has a different number of arguments than the variadic basefunction takes
fn test_create_wrapped_function_gives_error_for_empty_function_parameters_or_duplicate_elements(
) {
    let model_parameters = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    // we can't use assert_matches here because the Ok variant does not 
    // implement Debug
    assert!(
        matches!(
            create_wrapped_basis_function(
                &model_parameters,
                &Vec::<String>::new(),
                dummy_unit_function_for_x
            ),
            Err(ModelBuildError::EmptyParameters)
        ),
        "creating wrapper function with empty parameter list should report error"
    );
    assert!(
        matches!(
            create_wrapped_basis_function(
                &model_parameters,
                &["a", "a"],
                dummy_unit_function_for_x
            ),
            Err(ModelBuildError::DuplicateParameterNames { .. })
        ),
        "creating wrapper function with duplicates in function params should report error"
    );
    assert!(matches!(
        create_wrapped_basis_function(
            &model_parameters,
            &["a","b","c","d","e"],
            dummy_unit_function_for_x
        ),
        Err(ModelBuildError::IncorrectParameterCount {..})),
        "creating wrapper function with a different number of function parameters than the argument list of function takes"
    );
}

#[test]
fn creating_wrapped_basis_function_dispatches_elements_correctly_to_underlying_function() {
    let model_parameters = vec!["a", "b", "c", "d"];
    let function_parameters = vec!["c", "a"];
    let x = DVector::<f64>::from(vec![1., 3., 3., 7.]);
    let params = vec![1., 2., 3., 4.];

    // check that the dummy unit functions work as expected
    assert_eq!(
        dummy_unit_function_for_parameters(&x, params[0], params[1]),
        DVector::from(vec! {params[0],params[1]}),
        "dummy function must return parameters passed to it"
    );
    assert_eq!(
        dummy_unit_function_for_x(&x, params[0], params[1]),
        x,
        "dummy function must return the x argument passed to it"
    );

    // check that the wrapped function indeed redistributes the parameters expected
    let expected_out_params = DVector::<f64>::from(vec![3., 1.]);
    let wrapped_function_params = create_wrapped_basis_function(
        model_parameters.as_slice(),
        function_parameters.as_slice(),
        dummy_unit_function_for_parameters,
    )
    .unwrap();
    assert_eq!(
        wrapped_function_params(&x, params.as_slice()),
        expected_out_params,
        "Wrapped function must assign the correct function params from model params"
    );

    // check that the wrapped function passes just passes the x location parameters
    let wrapped_function_x = create_wrapped_basis_function(
        &model_parameters,
        &function_parameters,
        dummy_unit_function_for_x,
    )
    .unwrap();
    assert_eq!(
        wrapped_function_x(&x, params.as_slice()),
        x,
        "Wrapped function must pass the location argument unaltered"
    );
}

