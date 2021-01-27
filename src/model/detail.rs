use crate::model::errors::*;
use std::hash::Hash;
use std::collections::HashSet;

/// check that the parameter names obey the following rules
/// * the set of parameters is not empty
/// * the set of parameters contains only unique elements
/// # Returns
/// Ok if the conditions hold, otherwise an error variant.
pub fn check_parameter_names<StrType>(param_names: &[StrType]) -> Result<(), ModelfunctionError>
    where StrType: AsRef<str>,
          StrType: Hash + Eq {
    if param_names.is_empty() {
        return Err(ModelfunctionError::EmptyParameters);
    }

    if !has_only_unique_elements(param_names.iter()) {
        return Err(ModelfunctionError::DuplicateParameterNames);
    }

    Ok(())
}

// check if set is comprised of unique elements only
// https://stackoverflow.com/questions/46766560/how-to-check-if-there-are-duplicates-in-a-slice
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
pub fn create_index_mapping<T : PartialEq>(full : &[T], subset : &[T]) -> Result<Vec<usize>,ModelfunctionError> {
    let indices  = subset.iter().map(|value_subset|full.iter().position(|value_full|value_full==value_subset).ok_or(ModelfunctionError::InvalidParametersInSubset));
    // see https://stackoverflow.com/questions/26368288/how-do-i-stop-iteration-and-return-an-error-when-iteratormap-returns-a-result
    // the FromIterator trait of Result allows us to go from Vec<Result<A,B>> to Result<Vec<A>,B>
    indices.collect()
}


#[cfg(test)]
mod test {
    use super::*;

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
        assert!(check_parameter_names(&vec!{"a"}).is_ok());
        assert!(check_parameter_names(&vec!{"a","b","c"}).is_ok());
        assert!(check_parameter_names(&vec!{"a","b","b"}).is_err());
    }

    #[test]
    fn test_create_index_mapping() {
        let full_set = ['A','B','C','D'];
        assert_eq!(create_index_mapping(&full_set,&Vec::new()),Ok(Vec::new()),"Empty subset produces must produce empty index list");
        assert!(create_index_mapping(&Vec::new(),&vec!{'B','A'}).is_err(),"Empty full set must produce an error");
        assert_eq!(create_index_mapping(&Vec::<usize>::new(),&Vec::new()),Ok(Vec::new()),"Empty subset must produce empty index list even if full set is empty");

        assert_eq!(create_index_mapping(&full_set,&vec!{'B','A'}),Ok(vec!{1,0}),"Indices must be correctly assigned");
        assert!(create_index_mapping(&full_set,&vec!{'Z','Q'}).is_err(), "Indices that are not in the full set must produce an error");

        assert_eq!(create_index_mapping(&['A','A','B','D'],&vec!{'B','A'}),Ok(vec!{2,0}),"For duplicates in the full set, the first index is used");

    }

}