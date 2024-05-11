/// a helper trait for floating point numbers that can be cast from
/// f64. This is only implemented for f32 and f64. Casting f64 into
/// f32 is typically associated with a loss of precision.
pub trait CastF64 {
    /// helper for the constant 0 (zero)
    const ZERO: Self;
    /// helper for the constant 1 (one)
    const ONE: Self;

    /// make an f64 into a value of this type
    fn from_f64(value: f64) -> Self;

    /// make a value of this type into an f64
    fn into_f64(self) -> f64;
}

impl CastF64 for f64 {
    const ZERO: Self = 0.;
    const ONE: Self = 1.;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn into_f64(self) -> Self {
        self
    }
}

impl CastF64 for f32 {
    const ZERO: Self = 0.;
    const ONE: Self = 1.;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as _
    }

    #[inline]
    fn into_f64(self) -> f64 {
        self as _
    }
}
