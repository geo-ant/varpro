use super::*;

/// a function taking a location vector and 1 parameter argument
fn callable1(x: &DVector<f64>, alpha: f64) -> DVector<f64> {
    alpha * x.clone()
}

/// a function taking a location vector and 2 parameter arguments
fn callable2(x: &DVector<f64>, alpha1: f64, alpha2: f64) -> DVector<f64> {
    alpha1 * alpha2 * x.clone()
}

/// a helper function to evaluate callables of different arguments lists. Also doubles
/// as a syntax check that the trait I implemented really can be used to pass functions with
/// variadic arguments using this common interface.
fn eval_callable<ScalarType: Scalar, ArgList, BasisFuncType: BasisFunction<ScalarType, ArgList>>(
    f: BasisFuncType,
    x: &DVector<ScalarType>,
    params: &[ScalarType],
) -> DVector<ScalarType> {
    f.eval(x, params)
}

#[test]
fn callable_evaluation_using_the_generic_interface() {
    let x = DVector::from_element(10, 1.);
    assert_eq!(eval_callable(callable1, &x, &[2.]), 2. * x.clone());
    assert_eq!(eval_callable(callable2, &x, &[2., 3.]), 2. * 3. * x.clone());
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c| { a * b * c * x.clone() },
            &x,
            &[2., 3., 4.]
        ),
        2. * 3. * 4. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d| { a * b * c * d * x.clone() },
            &x,
            &[2., 3., 4., 5.]
        ),
        2. * 3. * 4. * 5. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e| { a * b * c * d * e * x.clone() },
            &x,
            &[2., 3., 4., 5., 6.]
        ),
        2. * 3. * 4. * 5. * 6. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e, f| { a * b * c * d * e * f * x.clone() },
            &x,
            &[2., 3., 4., 5., 6., 7.]
        ),
        2. * 3. * 4. * 5. * 6. * 7. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e, f, g| { a * b * c * d * e * f * g * x.clone() },
            &x,
            &[2., 3., 4., 5., 6., 7., 8.]
        ),
        2. * 3. * 4. * 5. * 6. * 7. * 8. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e, f, g, h| {
                a * b * c * d * e * f * g * h * x.clone()
            },
            &x,
            &[2., 3., 4., 5., 6., 7., 8., 9.]
        ),
        2. * 3. * 4. * 5. * 6. * 7. * 8. * 9. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e, f, g, h, i| {
                a * b * c * d * e * f * g * h * i * x.clone()
            },
            &x,
            &[2., 3., 4., 5., 6., 7., 8., 9., 10.]
        ),
        2. * 3. * 4. * 5. * 6. * 7. * 8. * 9. * 10. * x.clone()
    );
    assert_eq!(
        eval_callable(
            |x: &DVector<f64>, a, b, c, d, e, f, g, h, i, j| {
                a * b * c * d * e * f * g * h * i * j * x.clone()
            },
            &x,
            &[2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
        ),
        2. * 3. * 4. * 5. * 6. * 7. * 8. * 9. * 10. * 11. * x.clone()
    );
}

#[test]
#[allow(unused_variables)]
fn callable_argument_length_is_correct() {
    let x = DVector::from_element(10, 1.);
    assert_eq!(callable1.argument_count(), 1);
    assert_eq!(callable2.argument_count(), 2);
    assert_eq!(
        (|x: &DVector<f64>, a, b, c| { x.clone() }).argument_count(),
        3
    );
    assert_eq!(
        (|x: &DVector<f64>, a, b, c, d| { x.clone() }).argument_count(),
        4
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e| { x.clone() }).argument_count(),
        5
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e, f| { x.clone() }).argument_count(),
        6
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e, f, g| { x.clone() }).argument_count(),
        7
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e, f, g, h| { x.clone() }).argument_count(),
        8
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e, f, g, h, i| { x.clone() }).argument_count(),
        9
    );
    assert_eq!(
        (|x: &DVector<f32>, a, b, c, d, e, f, g, h, i, j| { x.clone() }).argument_count(),
        10
    );
}

#[test]
#[should_panic]
fn callable_evaluation_panics_if_too_few_elements_in_parameter_slice() {
    let x = DVector::from_element(10, 1.);
    let _ = eval_callable(callable2, &x, &[1.]);
}

#[test]
#[should_panic]
fn callable_evaluation_panics_if_too_many_elements_in_parameter_slice() {
    let x = DVector::from_element(10, 1.);
    let _ = eval_callable(callable1, &x, &[1., 2.]);
}
