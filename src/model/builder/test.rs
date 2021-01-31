use super::*;
use nalgebra::{U11, U2};

#[test]
fn builder_constructor_constructs_model_with_correct_parameters() {
    let parameter_names = vec!{"a","b","c"};
    let model = SeparableModelBuilder::<f64,U11>::with_parameters(parameter_names.clone()).build().unwrap();
    assert_eq!(&parameter_names,model.parameters());
    assert_eq!(model.parameter_count(), parameter_names.len());
}

#[test]
fn modelfunction_new_fails_for_invalid_parameters() {
    assert!(SeparableModelBuilder::<f64,U11>::with_parameters(vec!{"a","b","a"}).build().is_err(),"Model constructor must only allow unique parameter names.");
    assert!(SeparableModelBuilder::<f64,U2>::with_parameters(Vec::<String>::default()).build().is_err(),"Model must only allow non-empty parameter list.");
    assert!(SeparableModelBuilder::<f64,Dynamic>::with_parameters(vec!{"alpha","beta","gamma"}).build().is_ok(),"Model constructor must work for valid parameter list");
}
