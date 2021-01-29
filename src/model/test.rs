
use super::*;
use nalgebra::{U11, U2};

#[test]
fn modelfunction_new_creates_model_with_correct_parameters() {
    let parameter_names = vec!{"a","b","c"};
    let model = SeparableModel::<f64,U11>::new(parameter_names.clone()).unwrap();
    assert_eq!(&parameter_names,model.parameters());
    assert_eq!(model.parameter_count(), parameter_names.len());
}

#[test]
fn modelfunction_new_fails_for_invalid_parameters() {
    assert!(SeparableModel::<f64,U11>::new(vec!{"a","b","a"}).is_err(),"Model constructor must only allow unique parameter names.");
    assert!(SeparableModel::<f64,U2>::new(Vec::<String>::default()).is_err(),"Model must only allow non-empty parameter list.");
    assert!(SeparableModel::<f64,Dynamic>::new(vec!{"alpha","beta","gamma"}).is_ok(),"Model constructor must work for valid parameter list");
}
