use levenberg_marquardt::LeastSquaresProblem as LevenbergMarquardtLeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::{Dynamic, Scalar, ComplexField, Vector, DVector, Matrix, DMatrix, SVD, RealField};
use crate::model::SeparableModel;

#[cfg(test)]
mod test;
mod builder;

pub use builder::LevMarLeastSquaresProblemBuilder;
use std::ops::Mul;


/// TODO add weight matrix
/// TODO Document
pub struct LevMarProblem<'a,ScalarType>
    where ScalarType : Scalar + RealField{
    /// the independent variable `\vec{x}` (location parameter)
    x: DVector<ScalarType>,
    /// the data vector to which to fit the model `$\vec{y}$`
    y: DVector<ScalarType>,
    /// current parameters that the optimizer is operating on
    model_parameters: Vec<ScalarType>,
    /// The current residual of model function values belonging to the current parameters
    current_residuals: DVector<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    current_svd : SVD<ScalarType,Dynamic,Dynamic>,
    /// the linear coefficients `$\vec{c}$` providing the current best fit
    linear_coefficients: DVector<ScalarType>,
    /// a reference to the separable model we are trying to fit to the data
    model : &'a SeparableModel<ScalarType>,
    /// truncation epsilon for SVD below which all singular values are assumed zero
    svd_epsilon : ScalarType::RealField,
}


/// TODO document and document panics!
impl<'a,ScalarType> LevenbergMarquardtLeastSquaresProblem<ScalarType, Dynamic, Dynamic> for LevMarProblem<'a,ScalarType>
    where ScalarType : Scalar+RealField,
          ScalarType::RealField : Mul<ScalarType>{
    type ResidualStorage = Owned<ScalarType, Dynamic>;
    type JacobianStorage = Owned<ScalarType, Dynamic, Dynamic>;
    type ParameterStorage = Owned<ScalarType, Dynamic>;

    fn set_params(&mut self, params: &Vector<ScalarType, Dynamic, Self::ParameterStorage>) {
        self.model_parameters = params.iter().cloned().collect();
        let model_value_matrix = self.model.eval(&self.x, self.model_parameters.as_slice()).expect("Error evaluating function value matrix");
        self.current_svd = model_value_matrix.clone().svd(true, true);
        self.linear_coefficients = self.current_svd.solve(&self.y, self.svd_epsilon).expect("Error calculating solution with SVD.");
        self.current_residuals = &self.y - model_value_matrix * &self.linear_coefficients;
    }

    fn params(&self) -> Vector<ScalarType, Dynamic, Self::ParameterStorage> {
        DVector::from(self.model_parameters.clone())
    }
    fn residuals(&self) -> Option<Vector<ScalarType, Dynamic, Self::ResidualStorage>> {
        Some(self.current_residuals.clone())
    }

    #[allow(non_snake_case)]
    fn jacobian(&self) -> Option<Matrix<ScalarType, Dynamic, Dynamic, Self::JacobianStorage>> {
        //todo: make this more efficient by parallelizing
        let mut jacobian_matrix = unsafe {DMatrix::<ScalarType>::new_uninitialized(self.y.len(), self.model.parameter_count())};

        let U = self.current_svd.u.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");
        // U transposed
        let U_t = U.transpose();

        let Sigma_inverse : DMatrix<ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));

        let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

        for (k, mut jacobian_col) in jacobian_matrix.column_iter_mut().enumerate() {
            let Dk = self.model.eval_deriv(&self.x,self.model_parameters.as_slice()).at(k).expect("Error evaluating model derivatives.");
            let Dk_c = &Dk * &self.linear_coefficients;
            let minus_ak : DVector<ScalarType> = U *(&U_t *(&Dk_c))- Dk_c;
            let Dk_t_rw : DVector<ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
            let minus_bk : DVector<ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));
            jacobian_col.copy_from(&(minus_ak));
        }
        Some(jacobian_matrix)
    }
}
