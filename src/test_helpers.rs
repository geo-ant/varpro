//! This module includes helper functionality that is useful for testing across all modules

use nalgebra::DVector;

/// a function that calculates exp( -(t-t0)/tau)) for every location t
/// **ATTENTION** for this kind of exponential function the shift will
/// just be a linear multiplier exp(t0/tau), so it might not a good idea to include it in the fitting
pub fn exponential_decay(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| (-(t - t0) / tau).exp())
}

/// partial derivative of the exponential decay with respect to t0
pub fn exponential_decay_dt0(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    exponential_decay(tvec, t0, tau).map(|val| val / tau)
}

/// partial derivative of the exponential decay with respect to tau
pub fn exponential_decay_dtau(tvec: &DVector<f64>, t0: f64, tau: f64) -> DVector<f64> {
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((-t - t0) / tau).exp() * (t0 - t) / tau.powi(2))
}

/// implements the function sin(omega*t) for argument t
pub fn sinusoid_omega(tvec: &DVector<f64>, omega: f64) -> DVector<f64> {
    tvec.map(|t| (omega * t).sin())
}

/// implements the derivative function d/domega sin(omega*t) = omega * cos(omega*t) for argument t
pub fn sinusoid_omega_domega(tvec: &DVector<f64>, omega: f64) -> DVector<f64> {
    tvec.map(|t| (omega * t).cos() * omega)
}
