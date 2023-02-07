use crate::linalg_helpers::DiagDMatrix;
use approx::assert_relative_eq;
use nalgebra::{ComplexField, DMatrix, DVector};

#[test]
#[allow(non_snake_case)]
fn diagonal_matrix_matrix_and_matrix_vector_product_produces_correct_results() {
    let diagonal = DVector::from(vec![1., 2., 3., 4., 5.]);
    let ndiag = diagonal.len();
    // nalgebra dense diagonal matrix
    let D1 = DMatrix::from_diagonal(&diagonal);
    // my sparse diagonal matrix
    let D2 = DiagDMatrix::from(diagonal);

    let v = DVector::from(vec![5., 8., 6., 2., 5.]);

    let mut A = DMatrix::zeros(ndiag, 2);
    A.set_column(0, &DVector::from(vec![2., 6., 4., 9., 5.]));
    A.set_column(1, &DVector::from(vec![3., 9., 2., 1., 0.]));

    assert_eq!(
        &D1 * &A,
        &D2 * &A,
        "Diagonal matrix-matrix multiplication must produce correct result."
    );
    assert_eq!(
        &D1 * &v,
        &D2 * &v,
        "Diagonal matrix-vector multiplication must produce correct result."
    );

    let mut B = DMatrix::from_element(ndiag, 5, 0.);
    B.set_column(0, &DVector::from(vec![2., 6., 4., 9., 5.]));
    B.set_column(1, &DVector::from(vec![3., 9., 2., 1., 0.]));
    B.set_column(2, &DVector::from(vec![1., 5., 7., 4., 2.]));
    B.set_column(3, &DVector::from(vec![13., 7., 71., 46., 22.]));
    B.set_column(4, &DVector::from(vec![3., 77., 11., 26., 234.]));

    assert_eq!(
        &D1 * &B,
        &D2 * &B,
        "Diagonal matrix multiplication must produce correct result."
    );
}

#[test]
#[should_panic]
#[allow(non_snake_case)]
fn diagonal_matrix_vector_multiplication_must_panic_for_incorrect_rhs_dimension() {
    let diagonal = DVector::from(vec![1., 2., 3., 4., 5.]);
    let D = DiagDMatrix::from(diagonal);
    let v = DVector::from(vec![5., 5.]);
    let _ = &D * &v;
}

#[test]
#[should_panic]
#[allow(non_snake_case)]
fn diagonal_matrix_multiplication_must_panic_for_rhs_dimensions_too_small() {
    let diagonal = DVector::from(vec![1., 2., 3., 4., 5.]);
    let ndiag = diagonal.len();
    let D = DiagDMatrix::from(diagonal);
    let A = DMatrix::from_element(ndiag + 1, 1, 1.);
    let _ = &D * &A;
}

#[test]
#[should_panic]
#[allow(non_snake_case)]
fn diagonal_matrix_multiplication_must_panic_for_rhs_dimensions_too_large() {
    let diagonal = DVector::from(vec![1., 2., 3., 4., 5.]);
    let ndiag = diagonal.len();
    // my sparse diagonal matrix
    let D = DiagDMatrix::from(diagonal);
    let A = DMatrix::from_element(ndiag - 1, 1, 1.);
    let _ = &D * &A;
}

#[test]
#[allow(non_snake_case)]
fn diagonal_matrix_test_from_real_field() {
    use num_complex::Complex;
    let diagonal = DVector::from(vec![1., 2., 3.]);
    let D = DiagDMatrix::<Complex<f64>>::from_real_field(&diagonal);
    let ones = DVector::from(vec![
        Complex::new(1., 5.),
        Complex::new(1., 5.),
        Complex::new(1., 5.),
    ]);
    let result = &D * &ones;

    assert_relative_eq!(
        result.map(ComplexField::real),
        DVector::from(vec![1., 2., 3.])
    );
    assert_relative_eq!(
        result.map(ComplexField::imaginary),
        DVector::from(vec![5., 10., 15.])
    );
}
