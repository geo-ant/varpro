use std::collections::HashSet;
use std::hash::Hash;

use nalgebra::{DVector, Scalar};

use crate::basis_function::BasisFunction;
use crate::model::errors::*;

/// check that the parameter names obey the following rules
/// * the set of parameters is not empty
/// * the set of parameters contains only unique elements
/// # Returns
/// Ok if the conditions hold, otherwise an error variant.
pub fn check_parameter_names<StrType>(param_names: &[StrType]) -> Result<(), ModelBuildError>
where
    StrType: Hash + Eq + Clone + Into<String>,
{
    if param_names.is_empty() {
        return Err(ModelBuildError::EmptyParameters);
    }

    if !has_only_unique_elements(param_names.iter()) {
        let function_parameters: Vec<String> =
            param_names.iter().cloned().map(|n| n.into()).collect();
        Err(ModelBuildError::DuplicateParameterNames {
            function_parameters,
        })
    } else {
        Ok(())
    }
}

/// check if set is comprised of unique elements only
/// https://stackoverflow.com/questions/46766560/how-to-check-if-there-are-duplicates-in-a-slice
fn has_only_unique_elements<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}

/// Create an index mapping from a subset to the indices of a full set
/// e.g when the full set is [A,B,C,D] and the subset is [C,A] then the
/// index mapping is [2,0], because we are looking for the indices of the
/// elements in the subset in the full set.
/// The function returns the vector of indices or an error variant if something goes wrong
/// if the subset is empty, the result is an Ok variant with an empty vector.
/// if the full set contains duplicates, the index for the first element is used for the index mapping
pub fn create_index_mapping<T1, T2>(
    full: &[T1],
    subset: &[T2],
) -> Result<Vec<usize>, ModelBuildError>
where
    T1: Clone + PartialEq + PartialEq<T2>,
    T2: Clone + PartialEq + Into<String>,
{
    let indices = subset.iter().map(|value_subset| {
        full.iter()
            .position(|value_full| value_full == value_subset)
            .ok_or_else(|| ModelBuildError::FunctionParameterNotInModel {
                function_parameter: value_subset.clone().into(),
            })
    });
    // see https://stackoverflow.com/questions/26368288/how-do-i-stop-iteration-and-return-an-error-when-iteratormap-returns-a-result
    // the FromIterator trait of Result allows us to go from Vec<Result<A,B>> to Result<Vec<A>,B>
    indices.collect()
}


/// wraps model basis function so that it can be called with the whole model parameters
/// instead of the subset. Since we know that the function parameters we can select the subset
/// of model parameters and dispatch them to our model function
/// # Arguments
/// * `model_parameters`: the parameters that the complete model that this function belongs to
/// depends on
/// * `function_parameters`: the parameters (in right to left order) that the basisfunction depends
/// on. Must be a subset of the model parameters.
/// * `function` a basis function that depends on a number of parameters
/// # Result
/// Say our model depends on parameters `$\vec{p}=(\alpha,\beta,\gamma)$` and the `function` argument
/// is a basis function `$f(\vec{x},\gamma,\beta)$`. Then calling `create_wrapped_basis_function(&["alpha","beta","gamma"],&["gamma","alpha"],f)`
/// creates a wrapped function `$\tilde{f}(\vec{x},\vec{p})$` which can be called with `$\tilde{f}(\vec{x},\vec{p})=f(\vec{x},\gamma,\beta$`.
#[allow(clippy::type_complexity)]
pub fn create_wrapped_basis_function<ScalarType, ArgList, F,StrType,StrType2>(
    model_parameters: &[StrType],
    function_parameters: &[StrType2],
    function: F,
) -> Result<
    Box<dyn Fn(&DVector<ScalarType>, &[ScalarType]) -> DVector<ScalarType>>,
    ModelBuildError,
>
    where
        ScalarType: Scalar,
        F: BasisFunction<ScalarType,ArgList> +  'static,
        StrType: Into<String> + Clone + Hash + Eq + PartialEq<StrType2>,
        StrType2: Into<String> + Clone + Hash + Eq,
        String: PartialEq<StrType> + PartialEq<StrType2>,

{
    check_parameter_names(&model_parameters)?;
    check_parameter_names(&function_parameters)?;

    //check that the parameter count in the string parameter list is equal to the
    //parameter count in the argument of the function
    if function_parameters.len() != function.argument_count() {
        return Err(ModelBuildError::IncorrectParameterCount {
            params: function_parameters.iter().cloned().map(|p|p.into()).collect(),
            string_params_count: function_parameters.len(),
            function_argument_count: function.argument_count(),
        })
    }

    let index_mapping = create_index_mapping(model_parameters, function_parameters)?;

    let wrapped = move |x: &DVector<ScalarType>, params: &[ScalarType]| {
        //todo: refactor this, since this is unelegant and not parallelizable
        let mut parameters_for_function = Vec::<ScalarType>::with_capacity(index_mapping.len());
        for param_idx in index_mapping.iter() {
            parameters_for_function.push(params[*param_idx].clone());
        }
        function.eval(x,&parameters_for_function)
    };

    Ok(Box::new(wrapped))
}

#[cfg(test)]
mod test {
    use super::*;

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
        DVector::from(vec!{param1,param2})
    }

    // dummy function that just returns the given x
    fn dummy_unit_function_for_x(x: &DVector<f64>, _param1 : f64, _param2 : f64,) -> DVector<f64> {
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
        assert!(check_parameter_names(&Vec::<String>::default()).is_err());
        assert!(check_parameter_names(&["a"]).is_ok());
        assert!(check_parameter_names(&["a", "b", "c"]).is_ok());
        assert!(check_parameter_names(&["a", "b", "b"]).is_err());
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
        assert!(matches!(
            create_wrapped_basis_function(
                &model_parameters,
                &Vec::<String>::new(),
                dummy_unit_function_for_x
            )
            ,Err(ModelBuildError::EmptyParameters)),
            "creating wrapper function with empty parameter list should report error"
        );
        assert!(matches!(
            create_wrapped_basis_function(
                &model_parameters,
                &["a", "a"],
                dummy_unit_function_for_x
            )
            ,Err(ModelBuildError::DuplicateParameterNames{..})),
            "creating wrapper function with duplicates in function params should report error"
        );
        assert!(matches!(
            create_wrapped_basis_function(
                &model_parameters,
                &["a","b","c","d","e"],
                dummy_unit_function_for_x
            )
            ,Err(ModelBuildError::IncorrectParameterCount {..})),
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
            dummy_unit_function_for_parameters(&x, params[0],params[1]),
            DVector::from(vec!{params[0],params[1]}),
            "dummy function must return parameters passed to it"
        );
        assert_eq!(
            dummy_unit_function_for_x(&x, params[0],params[1]),
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
}
