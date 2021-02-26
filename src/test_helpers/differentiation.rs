use num_traits::Float;

use approx::assert_relative_eq;

/// helper function to calculate a numerical derivative of a function real valued function
/// `$f:\math{R} \rightarrow \mathbb{R}$`
/// so both the function `$f$` and its value `$f(p)$` are scalar valued and this function
/// calculates a numerical approximation of `$\partial f /\partial p (p_0)$`
/// uses central finite differences with accuracy O(h⁶)
pub fn numerical_derivative<Func, ScalarType>(mut f: Func, p0: ScalarType) -> ScalarType
where
    ScalarType: Float,
    Func: FnMut(ScalarType) -> ScalarType,
{
    // stepsize for numerical differentiation
    // don't pick the stepsize too small. So I (kind of arbitrarily) pick it to be
    // sqrt of the machine precision
    let h = ScalarType::epsilon().sqrt();

    let forty_five = ScalarType::from(45).expect("Cannot convert scalar from integer");
    let nine = ScalarType::from(9).expect("Cannot convert scalar from integer");
    let two = ScalarType::from(2).expect("Cannot convert scalar from integer");
    let three = ScalarType::from(3).expect("Cannot convert scalar from integer");
    let sixty = ScalarType::from(60).expect("Cannot convert scalar from integer");

    (-f(p0 - three * h) + nine * f(p0 - two * h) - forty_five * f(p0 - h) + forty_five * f(p0 + h)
        - nine * f(p0 + two * h)
        + f(p0 + three * h))
        / (h * sixty)
}

#[test]
fn numeric_differentiation_produces_correct_results() {
    // function x*sin(x)+x^2*cos(x)
    let f = |x: f64| x * x.sin() + x.powi(2) * x.cos();
    // derivative: d/dx(x sin(x) + x^2 cos(x)) = x^2 (-sin(x)) + sin(x) + 3 x cos(x)
    let df = |x: f64| -x.powi(2) * x.sin() + x.sin() + 3. * x * x.cos();

    assert_relative_eq!(numerical_derivative(f, 1.), df(1.), epsilon = 1e-6);
    assert_relative_eq!(numerical_derivative(f, 2.), df(2.), epsilon = 1e-6);
    assert_relative_eq!(numerical_derivative(f, 3.), df(3.), epsilon = 1e-6);
    assert_relative_eq!(numerical_derivative(f, 5.), df(5.), epsilon = 1e-6);
    assert_relative_eq!(numerical_derivative(f, 10.), df(10.), epsilon = 1e-6);
}
