use std::convert::Infallible;

#[cfg(test)]
use crate::util::to_matrix;
use crate::{
    prelude::SeparableNonlinearModel,
    statistics::{extract_range, model_function_jacobian},
};

use super::{calc_correlation_matrix, concat_colwise};
use approx::assert_relative_eq;
use nalgebra::{vector, DMatrix, DVector, Dyn, OMatrix, U0, U1, U2, U3, U4, U5, U6, U7};
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
    // correlation matrix
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 3. / f64::sqrt(10.), 4. / f64::sqrt(10.), 1.0]);
    let calc = calc_correlation_matrix(&cov);
    assert_relative_eq!(corr, calc);
}

/// this thing exists only to be evaluated and give us a jacobian
struct DummyModel {}

impl SeparableNonlinearModel for DummyModel {
    type ScalarType = f64;
    type Error = Infallible;

    fn parameter_count(&self) -> usize {
        2
    }

    fn base_function_count(&self) -> usize {
        3
    }

    fn output_len(&self) -> usize {
        3
    }

    fn set_params(
        &mut self,
        _parameters: nalgebra::OVector<Self::ScalarType, Dyn>,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn params(&self) -> nalgebra::OVector<Self::ScalarType, Dyn> {
        todo!()
    }

    fn eval(&self) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error> {
        Ok(DMatrix::from_row_slice(
            3,
            3,
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        ))
    }

    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error> {
        let mut jacobian = DMatrix::zeros(3, 3);
        jacobian.column_mut(derivative_index).fill(1.0);
        jacobian
            .column_mut(self.base_function_count() - 1)
            .fill(1.0);
        Ok(jacobian)
    }
}

#[test]
fn model_function_jacobian_is_calculated_correctly() {
    // this test makes sense once you look at the DummyModel implementation
    // and print the matrices for the eval and the partial derivatives
    let model = DummyModel {};
    let c = DVector::from_column_slice(&[5., 6., 7.]);
    let expected_jac = concat_colwise(
        model.eval().unwrap(),
        DMatrix::from_column_slice(3, 2, &[12., 12., 12., 13., 13., 13.]),
    );
    let calculated_jac = model_function_jacobian(&model, c.as_view());
    assert_relative_eq!(expected_jac, calculated_jac.unwrap());
}

#[test]
fn extract_range_for_dynamic_vector() {
    // test range extraction for fixed size and dynamic vectors with different
    // parameters
    let vec = DVector::from_column_slice(&[1., 2., 3., 4., 5., 6.]);
    let expected = DVector::from_column_slice(&[2., 3., 4.]);
    assert_relative_eq!(expected, extract_range(&vec, Dyn(1), Dyn(4)));
    assert_relative_eq!(vec, extract_range(&vec, Dyn(0), Dyn(6)));
    let expected = vector![2., 3., 4.];
    assert_relative_eq!(expected, extract_range(&vec, U1, U4));
    let expected = vector![1., 2., 3., 4., 5., 6.];
    assert_relative_eq!(expected, extract_range(&vec, U0, U6));
}

#[test]
#[should_panic]
fn extract_range_for_dynamic_vector_fails_for_out_of_bounds_dynamic() {
    let vec = DVector::from_column_slice(&[1., 2., 3., 4., 5., 6.]);
    _ = extract_range(&vec, Dyn(0), Dyn(7));
}

#[test]
#[should_panic]
fn extract_range_for_dynamic_vector_fails_for_out_of_bounds_static() {
    let vec = DVector::from_column_slice(&[1., 2., 3., 4., 5., 6.]);
    _ = extract_range(&vec, U0, U7);
}

// same tests as above for statically sized vector
#[test]
fn extract_range_for_static_vector() {
    let vec = vector![1., 2., 3., 4., 5., 6.];
    let expected = vector![2., 3., 4.];
    assert_relative_eq!(expected, extract_range(&vec, U1, U4));
    let expected = vector![1., 2., 3., 4., 5., 6.];
    assert_relative_eq!(expected, extract_range(&vec, U0, U6));
    let expected = DVector::from_column_slice(&[2., 3., 4.]);
    assert_relative_eq!(expected, extract_range(&vec, Dyn(1), U4));
}

#[test]
#[should_panic]
fn extract_range_for_static_vector_fails_for_out_of_bounds_static() {
    let vec = vector![1., 2., 3., 4., 5., 6.];
    _ = extract_range(&vec, Dyn(0), U7);
}

#[test]
#[should_panic]
fn extract_range_for_static_vector_fails_for_out_of_bounds_dynamic() {
    let vec = vector![1., 2., 3., 4., 5., 6.];
    _ = extract_range(&vec, U0, Dyn(7));
}

#[test]
#[should_panic]
fn extract_range_for_static_vector_fails_for_out_of_bounds_dynamic_2() {
    let vec = vector![1., 2., 3., 4., 5., 6.];
    _ = extract_range(&vec, U0, U7);
}
