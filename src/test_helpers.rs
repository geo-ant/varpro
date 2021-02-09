//! This module includes helper functionality that is useful for testing across all modules

use nalgebra::Dim;

use crate::model::OwnedVector;

/// a function that calculates exp( (t-t0)/tau)) for every location t
pub fn exponential_decay<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((t - t0) / tau).exp())
}

/// partial derivative of the exponential decay with respect to t0
pub fn exponential_decay_dt0<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    exponential_decay(tvec, t0, tau).map(|val| val / tau)
}

/// partial derivative of the exponential decay with respect to tau
pub fn exponential_decay_dtau<NData>(
    tvec: &OwnedVector<f64, NData>,
    t0: f64,
    tau: f64,
) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    assert!(tau > 0f64, "Parameter tau must be greater than zero");
    tvec.map(|t| ((t - t0) / tau).exp() * (t0 - t) / tau.powi(2))
}

/// implements the function sin(omega*t) for argument t
pub fn sinusoid_omega<NData>(tvec: &OwnedVector<f64, NData>, omega:f64) -> OwnedVector<f64,NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    tvec.map(|t| (omega*t).sin())
}

/// implements the derivative function d/domega sin(omega*t) = omega * cos(omega*t) for argument t
pub fn sinusoid_omega_domega<NData>(tvec: &OwnedVector<f64, NData>, omega:f64) -> OwnedVector<f64,NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
{
    tvec.map(|t| (omega*t).cos()*omega)
}
