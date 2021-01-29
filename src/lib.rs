extern crate nalgebra;
extern crate snafu;
extern crate levenberg_marquardt;




pub mod model;

#[allow(dead_code)]
fn unit_function<T:Clone>(x : &T) -> T {
    x.clone()
}

#[test]
fn test_some_syntax() {
    let mut model = SeparableModel::<f32,Dynamic>::new(vec!["a","b"]).unwrap();
    model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");
    model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");
}