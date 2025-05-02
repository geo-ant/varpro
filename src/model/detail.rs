use nalgebra::{DVector, Scalar};
use std::collections::HashSet;
use std::hash::Hash;

use crate::basis_function::BasisFunction;
use crate::model::builder::error::ModelBuildError;

/// check that the parameter names obey the following rules
/// * the set of parameters is not empty
/// * the set of parameters contains only unique elements
/// * any of the parameter names contains a comma. This indicates most likely a typo when giving the parameter list
///
/// # Returns
///
/// Ok if the conditions hold, otherwise an error variant.
pub fn check_parameter_names<StrType>(param_names: &[StrType]) -> Result<(), ModelBuildError>
where
    StrType: Hash + Eq + Clone + Into<String>,
{
    if param_names.is_empty() {
        return Err(ModelBuildError::EmptyParameters);
    }

    // TODO (Performance): this is inefficient. Refactor the interface of this method
    if let Some(param_name) = param_names.iter().find(|&p| p.clone().into().contains(',')) {
        return Err(ModelBuildError::CommaInParameterNameNotAllowed {
            param_name: param_name.clone().into(),
        });
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
///   depends on
/// * `function_parameters`: the parameters (in right to left order) that the basisfunction depends
///   on. Must be a subset of the model parameters.
/// * `function` a basis function that depends on a number of parameters
///
/// # Result
///
/// Say our model depends on parameters `$\vec{p}=(\alpha,\beta,\gamma)$` and the `function` argument
/// is a basis function `$f(\vec{x},\gamma,\beta)$`. Then calling `create_wrapped_basis_function(&["alpha","beta","gamma"],&["gamma","alpha"],f)`
/// creates a wrapped function `$\tilde{f}(\vec{x},\vec{p})$` which can be called with `$\tilde{f}(\vec{x},\vec{p})=f(\vec{x},\gamma,\beta$`.
#[allow(clippy::type_complexity)]
pub fn create_wrapped_basis_function<ScalarType, ArgList, F, StrType, StrType2>(
    model_parameters: &[StrType],
    function_parameters: &[StrType2],
    function: F,
) -> Result<
    Box<dyn Fn(&DVector<ScalarType>, &[ScalarType]) -> DVector<ScalarType> + Send + Sync>,
    ModelBuildError,
>
where
    ScalarType: Scalar,
    F: BasisFunction<ScalarType, ArgList> + 'static,
    StrType: Into<String> + Clone + Hash + Eq + PartialEq<StrType2>,
    StrType2: Into<String> + Clone + Hash + Eq,
    String: PartialEq<StrType> + PartialEq<StrType2>,
{
    check_parameter_names(model_parameters)?;
    check_parameter_names(function_parameters)?;
    check_parameter_count(function_parameters, &function)?;

    let index_mapping = create_index_mapping(model_parameters, function_parameters)?;

    let wrapped = move |x: &DVector<ScalarType>, params: &[ScalarType]| {
        // TODO (Performance): refactor this, since this is not elegant and not parallelizeable
        let mut parameters_for_function = Vec::<ScalarType>::with_capacity(index_mapping.len());
        for param_idx in index_mapping.iter() {
            parameters_for_function.push(params[*param_idx].clone());
        }
        function.eval(x, &parameters_for_function)
    };

    Ok(Box::new(wrapped))
}

/// Check that the parameter count in the list of function parameters really does match the number of
/// parameters the function type requires. If so return Ok(()), otherwise return an error
pub fn check_parameter_count<StrType, ScalarType, F, ArgList>(
    function_parameters: &[StrType],
    _function: &F,
) -> Result<(), ModelBuildError>
where
    StrType: Into<String> + Clone,
    F: BasisFunction<ScalarType, ArgList> + 'static,
    ScalarType: Scalar,
{
    if function_parameters.len() == F::ARGUMENT_COUNT {
        Ok(())
    } else {
        Err(ModelBuildError::IncorrectParameterCount {
            actual: function_parameters.len(),
            expected: F::ARGUMENT_COUNT,
        })
    }
}

#[cfg(any(test, doctest))]
mod test;
