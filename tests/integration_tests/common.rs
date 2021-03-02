use num_traits::Float;
use nalgebra::{DVector, Scalar, ComplexField};
use varpro::model::SeparableModel;
use varpro::model::builder::SeparableModelBuilder;

/// create holding `count` the elements from range [first,last] with linear spacing. (equivalent to matlabs linspace)
pub fn linspace<ScalarType:Float+Scalar>(first : ScalarType, last:ScalarType, count : usize) -> DVector<ScalarType>{
    let n_minus_one = ScalarType::from(count-1).expect("Could not convert usize to Float");
    let lin : Vec<ScalarType> = (0..count).map(|n|first+(first-last)/(n_minus_one)*ScalarType::from(n).expect("Could not convert usize to Float")).collect();
    DVector::from(lin)
}

/// evaluete the vector valued function of a model by evaluating the model at the given location
/// `x` with (nonlinear) parameters `params` and by calculating the linear superposition of the basisfunctions
/// with the given linear coefficients `linear_coeffs`.
pub fn evaluate_complete_model<ScalarType>(model : &SeparableModel<ScalarType>, x: &DVector<ScalarType>, params: &[ScalarType], linear_coeffs : &DVector<ScalarType>) -> DVector<ScalarType>
where ScalarType: Scalar + ComplexField,
{
    (&model.eval(x, params).expect("Evaluating model must not produce error"))*linear_coeffs
}

/// exponential decay f(t,tau) = exp(-t/tau)
pub fn exp_decay<ScalarType: Float + Scalar>(
    tvec: &DVector<ScalarType>,
    tau: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (-t / tau).exp())
}

/// derivative of exp decay with respect to tau
pub fn exp_decay_dtau<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    tau: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (-t / tau).exp() * t / (tau * tau))
}

/// A helper function that returns a double exponential decay model
/// f(x,tau1,tau2) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
/// Model parameters are: tau1, tau2
pub fn get_double_exponential_model_with_constant_offset() -> SeparableModel<f64> {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .invariant_function(ones)
        .build()
        .expect("double exponential model builder should produce a valid model")
}