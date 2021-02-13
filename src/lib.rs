//! TODO Document this crate
//! let's try some math
//! ```math
//!   f(\vec{x},\vec{\alpha}) = \sum_j f_j(\vec{x},\vec{\alpha}_j)
//! ```
//! Where `$\vec{\alpha}_j$` is a subset of the model parameters `$\vec{\alpha}$`.
//!
//! let's also try to link some other code [SeparableModelBuilder] but maybe I got to look
//! at the rustdoc documentation again

pub mod model;
pub mod prelude;
#[cfg(test)]
mod test_helpers;

#[test]
fn test_some_syntax() {
    //let mut model = SeparableModel::<f32,Dynamic>::new(vec!["a","b"]).unwrap();
    //model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");
    //model.independent_function(&unit_function).push().expect("pushing a parameter independent model function must not result in an error");

    // let _model = SeparableModelBuilder::<f32, Dynamic>::with_parameters(&["a", "b"])
    //     .push_invariant_function(&unit_function)
    //     .push_invariant_function(&unit_function)
    //     .build()
    //     .unwrap();
}
