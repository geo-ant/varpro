use nalgebra::{DVector, Scalar};
use num_traits::Float;

use crate::model::builder::SeparableModelBuilder;
use crate::model::SeparableModel;

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

/// function sin (omega*t+phi)
pub fn sin_ometa_t_plus_phi<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (omega * t + phi).sin())
}

/// derivative d/d(omega) sin (omega*t+phi)
pub fn sin_ometa_t_plus_phi_domega<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| t * (omega * t + phi).cos())
}

/// derivative d/d(phi) sin (omega*t+phi)
pub fn sin_ometa_t_plus_phi_dphi<ScalarType: Scalar + Float>(
    tvec: &DVector<ScalarType>,
    omega: ScalarType,
    phi: ScalarType,
) -> DVector<ScalarType> {
    tvec.map(|t| (omega * t + phi).cos())
}

/// A helper function that returns a double exponential decay model
/// f(x,tau1,tau2) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
pub fn get_double_exponential_model_with_constant_offset() -> SeparableModel<f64> {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .invariant_function(ones)
        .build()
        .expect("double exponential model builder should produce a valid model")
}
