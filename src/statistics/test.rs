use std::convert::Infallible;

use crate::{
    model::SeparableModel, prelude::SeparableNonlinearModel, statistics::model_function_jacobian,
};

use super::{calc_correlation_matrix, concat_colwise};
use approx::assert_relative_eq;
use nalgebra::{matrix, DMatrix, DVector, Dim, Dyn, OMatrix, U2, U3, U5};
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
    type ParameterDim = Dyn;
    type ModelDim = Dyn;
    type OutputDim = Dyn;

    fn parameter_count(&self) -> Self::ParameterDim {
        Dyn(2)
    }

    fn base_function_count(&self) -> Self::ModelDim {
        Dyn(3)
    }

    fn output_len(&self) -> Self::OutputDim {
        Dyn(3)
    }

    fn set_params(
        &mut self,
        _parameters: nalgebra::OVector<Self::ScalarType, Self::ParameterDim>,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn params(&self) -> nalgebra::OVector<Self::ScalarType, Self::ParameterDim> {
        todo!()
    }

    fn eval(
        &self,
    ) -> Result<OMatrix<Self::ScalarType, Self::OutputDim, Self::ModelDim>, Self::Error> {
        Ok(DMatrix::from_row_slice(
            3,
            3,
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        ))
    }

    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<OMatrix<Self::ScalarType, Self::OutputDim, Self::ModelDim>, Self::Error> {
        let mut jacobian = DMatrix::zeros(3, 3);
        jacobian.column_mut(derivative_index).fill(1.0);
        jacobian
            .column_mut(self.base_function_count().value() - 1)
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
    let calculated_jac = model_function_jacobian(&model, &c);
    assert_relative_eq!(expected_jac, calculated_jac.unwrap());
}
