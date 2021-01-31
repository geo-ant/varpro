extern crate nalgebra;
extern crate snafu;
extern crate levenberg_marquardt;

use crate::model::builder::SeparableModelBuilder;
use nalgebra::Dynamic;

pub mod model;
pub mod prelude;

#[allow(dead_code)]
fn unit_function<T:Clone>(x : &T) -> T {
    x.clone()
}

#[test]
fn test_some_syntax() {
    //let mut model = SeparableModel::<f32,Dynamic>::new(vec!["a","b"]).unwrap();
    //model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");
    //model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");

    let model = SeparableModelBuilder::<f32,Dynamic>::with_parameters(vec!["a","b"])
        .push_invariant_function(&unit_function)
        .push_invariant_function(&unit_function)
        .build()
        .unwrap();
}