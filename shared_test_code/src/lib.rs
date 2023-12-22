#![warn(missing_docs)]
//! a helper crate which carries common code used by the benchtests and the
//! integration tests.
use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{
    ComplexField, DVector, DefaultAllocator, Dim, Dyn, Matrix, OVector, RawStorage, Scalar,
};
use num_traits::Float;
use varpro::model::builder::SeparableModelBuilder;
use varpro::model::SeparableModel;
use varpro::prelude::SeparableNonlinearModel;

/// contains models both for the levmar crate as well as the
/// varpro crate
pub mod models;

/// create holding `count` the elements from range [first,last] with linear spacing. (equivalent to matlabs linspace)
pub fn linspace<ScalarType: Float + Scalar>(
    first: ScalarType,
    last: ScalarType,
    count: usize,
) -> DVector<ScalarType> {
    let n_minus_one = ScalarType::from(count - 1).expect("Could not convert usize to Float");
    let lin: Vec<ScalarType> = (0..count)
        .map(|n| {
            first
                + (first - last) / (n_minus_one)
                    * ScalarType::from(n).expect("Could not convert usize to Float")
        })
        .collect();
    DVector::from(lin)
}

/// evaluete the vector valued function of a model by evaluating the model at the given location
/// `x` with (nonlinear) parameters `params` and by calculating the linear superposition of the basisfunctions
/// with the given linear coefficients `linear_coeffs`.
pub fn evaluate_complete_model_at_params<Model>(
    model: &'_ mut Model,
    params: OVector<Model::ScalarType, Dyn>,
    linear_coeffs: &OVector<Model::ScalarType, Dyn>,
) -> OVector<Model::ScalarType, Dyn>
where
    Model::ScalarType: Scalar + ComplexField,
    Model: SeparableNonlinearModel,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Dyn, Dyn>,
    DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Dyn>,
{
    let original_params = model.params();
    model
        .set_params(params)
        .expect("Setting params must not fail");
    let eval = (&model
        .eval()
        .expect("Evaluating model must not produce error"))
        * linear_coeffs;
    model
        .set_params(original_params)
        .expect("Setting params must not fail");
    eval
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
pub fn get_double_exponential_model_with_constant_offset(
    x: DVector<f64>,
    initial_params: Vec<f64>,
) -> SeparableModel<f64> {
    let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

    SeparableModelBuilder::new(&["tau1", "tau2"])
        .initial_parameters(initial_params)
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .invariant_function(ones)
        .independent_variable(x)
        .build()
        .expect("double exponential model builder should produce a valid model")
}

/// a helper function that allows us to check if two matrix have equal
/// elements. If the matrix do not have the same dimensions, the relative
/// equality comparison returns false anyways.
/// panics if the relative equality comparison fails
#[deprecated = "use this only to get stuff to compile until all matrices are dynamic"]
pub fn check_relative_matrix_eq<T, R1, C1, S1, R2, C2, S2>(
    left: &Matrix<T, R1, C1, S1>,
    right: &Matrix<T, R2, C2, S2>,
    epsilon: <T as AbsDiffEq>::Epsilon,
) where
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    T: RelativeEq,
    S1: RawStorage<T, R1, C1> + std::fmt::Debug,
    S2: RawStorage<T, R2, C2> + std::fmt::Debug,
    <T as AbsDiffEq>::Epsilon: Copy,
{
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        panic!(
            "Matrices do not have the same dimensions. Left is {}x{} and right is {}x{}",
            left.nrows(),
            left.ncols(),
            right.nrows(),
            right.ncols()
        );
    }

    let equal = left
        .iter()
        .zip(right.iter())
        .all(|(l, r)| l.relative_eq(r, epsilon, T::default_max_relative()));
    if !equal {
        panic!(
            "Matrices are not equal:\nLeft is: {:#?}\nRight is: {:#?}",
            left, right
        );
    }
}
