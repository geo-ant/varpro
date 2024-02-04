use nalgebra::DVector;

#[derive(Clone, PartialEq, Debug)]
/// this is a double exponential model with constant offset
/// that does not cache its current evaluations when setting the
/// parameters. This is to test the suspicion that caching
/// the evaluation is slower for large model sizes rather than
/// recalculating
pub struct DoubleExpModelWithConstantOffsetSepModelUncached {
    /// the x vector associated with this model
    x_vector: DVector<f64>,
    /// the current nonlinear parameters of the model
    params: DVector<f64>,
}
