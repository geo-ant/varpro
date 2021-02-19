//use crate::model::SeparableModel;

use crate::model::builder::SeparableModelBuilder;
use crate::model::SeparableModel;
use nalgebra::{DVector, Scalar};
use num_traits::Float;

/// exponential decay f(t,tau) = exp(-t/tau)
fn exp_decay<ScalarType: Float + Scalar>(
    tvec: &DVector<ScalarType>,
    tau: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (-t / tau).exp())
}

/// A helper function that returns a double exponential decay model
/// f(x,tau1,tau2) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
fn get_double_exponential_model_with_constant_offset() -> SeparableModel<f64> {
    let exp_decay_dtau = |t: &DVector<_>, tau| t / (tau * tau) * exp_decay(t, tau);
    let ones = |t:&DVector<_>| DVector::from_element(t.len(), 1.);

    SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .invariant_function(ones)
        .build()
        .expect("double exponential model builder should produce a valid model")
}

#[test]
// test that the eval method produces correct results and gives a matrix that is ordered correctly
fn model_function_eval_produces_correct_result() {
    let model = get_double_exponential_model_with_constant_offset();

    let tvec = DVector::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let tau1 = 1.;
    let tau2 = 3.;

    let params = &[tau1, tau2];
    let eval_matrix = model
        .eval(&tvec, params)
        .expect("Model evaluation should not fail");

    assert_eq!(DVector::from(eval_matrix.column(0)), exp_decay(&tvec, tau2), "first column must correspond to first model function: exp(-t/tau2)");
    assert_eq!(DVector::from(eval_matrix.column(1)), exp_decay(&tvec, tau1), "second column must correspond to second model function: exp(-t/tau1)");
    assert_eq!(DVector::from(eval_matrix.column(2)), DVector::from_element(tvec.len(),1.),"third column must be vector of ones");
}

#[test]
#[ignore]
// test that when a base function does not return the same length result as its location argument,
// then the eval method fails
// TODO: it's fine to check that model above, but we also need some model where more derivatives are nonzero for the same param
fn model_function_eval_fails_for_invalid_length_of_return_value_in_base_function() {}
