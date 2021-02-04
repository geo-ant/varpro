use super::*;
use nalgebra::{U11, U2,Dynamic};

#[test]
fn builder_constructor_constructs_model_with_correct_parameters() {
    let parameter_names = vec!{"a".to_string(),"b".to_string(),"c".to_string()};
    let model = SeparableModelBuilder::<f64,U11>::with_parameters(&parameter_names).build().unwrap();
    assert_eq!(&parameter_names,model.parameters());
    assert_eq!(model.parameter_count(), parameter_names.len());
}

#[test]
fn modelfunction_new_fails_for_invalid_parameters() {
    assert!(SeparableModelBuilder::<f64,U11>::with_parameters(&["a","b","a"]).build().is_err(),"Model constructor must only allow unique parameter names.");
    assert!(SeparableModelBuilder::<f64,U2>::with_parameters(&Vec::<String>::default()).build().is_err(),"Model must only allow non-empty parameter list.");
    assert!(SeparableModelBuilder::<f64,Dynamic>::with_parameters(&["alpha","beta","gamma"]).build().is_ok(),"Model constructor must work for valid parameter list");
}
