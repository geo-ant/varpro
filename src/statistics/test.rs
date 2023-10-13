use super::{calc_correlation_matrix, concat_colwise};
use approx::assert_relative_eq;
use nalgebra::{DMatrix, Dyn, OMatrix, U2, U3, U5};
#[test]
fn matrix_concatenation_for_dynamic_matrices() {
    // two DMatrix instances with the same number of rows but
    // different number of columns
    let lhs = DMatrix::from_column_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
    let rhs = DMatrix::from_column_slice(2, 2, &[7., 8., 9., 10.]);
    let concat = DMatrix::from_column_slice(2, 5, &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    assert_relative_eq!(concat, concat_colwise(lhs, rhs));
}

#[test]
// the same test as above but lhs is a fixed column size matrix
fn matrix_concatenation_for_fixed_and_dyn_size_matrices() {
    // two DMatrix instances with the same number of rows but
    // different number of columns
    let lhs = OMatrix::<f64, Dyn, U3>::from_column_slice(&[1., 2., 3., 4., 5., 6.]);
    let rhs = DMatrix::from_column_slice(2, 2, &[7., 8., 9., 10.]);
    let concat = DMatrix::from_column_slice(2, 5, &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    assert_relative_eq!(concat, concat_colwise(lhs, rhs));
}

#[test]
// same test as above, but both matrices are fixed column size
fn matrix_concatenation_for_fixed_size_matrices() {
    // two DMatrix instances with the same number of rows but
    // different number of columns
    let lhs = OMatrix::<f64, U2, U3>::from_column_slice(&[1., 2., 3., 4., 5., 6.]);
    let rhs = OMatrix::<f64, U2, U2>::from_column_slice(&[7., 8., 9., 10.]);
    let concat =
        OMatrix::<f64, U2, U5>::from_column_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    assert_relative_eq!(concat, concat_colwise(lhs, rhs));
}

#[test]
fn correlation_matrix_is_calculated_correctly_from_a_covariance_matrix() {
    // covariance matrix
    let cov = DMatrix::from_row_slice(2, 2, &[2., 3., 4., 5.]);
    println!("cov: {}", cov);
    // correlation matrix
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 3. / f64::sqrt(10.), 4. / f64::sqrt(10.), 1.0]);
    let calc = calc_correlation_matrix(&cov);
    println!("corr: {}", corr);
    println!("calc: {}", calc);
    assert_relative_eq!(corr, calc);
}
