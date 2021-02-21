use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::{Dynamic, Scalar, ComplexField, Vector, DVector, Matrix, DMatrix, SVD};
use crate::model::SeparableModel;

/// TODO add weight matrix
pub struct LevMarSolver<'a,ScalarType>
    where ScalarType : Scalar + ComplexField{
    /// the independent variable `\vec{x}` (location parameter)
    location : DVector<ScalarType>,
    /// the data vector to which to fit the model `$\vec{y}$`
    data : DVector<ScalarType>,

    /// current parameters that the optimizer is operating on
    current_parameters : Vec<ScalarType>,
    /// The current residual of model function values belonging to the current parameters
    current_residuals: DVector<ScalarType>,
    /// the current matrix U of the SVD (
    current_svd : SVD<ScalarType,Dynamic,Dynamic>,

    /// the linear coefficients providing the current best fit
    current_linear_coeffs : DVector<ScalarType>,
    /// a reference to the separable model we are trying to fit to the data
    model : &'a SeparableModel<ScalarType>,

    /// truncation epsilon for SVD below which all singular values are assumed zero
    svd_epsilon : ScalarType::RealField,
}


/// TODO: document
/// TODO document panics!
impl<'a,ScalarType> LeastSquaresProblem<ScalarType, Dynamic, Dynamic> for LevMarSolver<'a,ScalarType>
    where ScalarType : Scalar+ComplexField{
    type ResidualStorage = Owned<ScalarType, Dynamic>;
    type JacobianStorage = Owned<ScalarType, Dynamic, Dynamic>;
    type ParameterStorage = Owned<ScalarType, Dynamic>;

    fn set_params(&mut self, params: &Vector<ScalarType, Dynamic, Self::ParameterStorage>) {
        self.current_parameters = params.iter().cloned().collect();
        let model_value_matrix = self.model.eval(&self.location, self.current_parameters.as_slice()).expect("Error evaluating function value matrix");
        self.current_svd = model_value_matrix.clone().svd(true, true);
        self.current_linear_coeffs = self.current_svd.solve(&self.data, self.svd_epsilon).expect("Error calculating SVD.");
        self.current_residuals = &self.data - model_value_matrix * &self.current_linear_coeffs;
    }

    fn params(&self) -> Vector<ScalarType, Dynamic, Self::ParameterStorage> {
        DVector::from(self.current_parameters.clone())
    }
    fn residuals(&self) -> Option<Vector<ScalarType, Dynamic, Self::ResidualStorage>> {
        Some(self.current_residuals.clone())
    }

    #[allow(non_snake_case)]
    fn jacobian(&self) -> Option<Matrix<ScalarType, Dynamic, Dynamic, Self::JacobianStorage>> {
        //todo: make this more efficient
        let mut jacobian_matrix = unsafe {DMatrix::<ScalarType>::new_uninitialized(self.data.len(), self.model.basis_function_count())};

        let U = self.current_svd.u.as_ref().expect("Did not calculate U of SVD");
        // U transposed
        let U_t = U.transpose();

        for (idx , mut jacobian_col) in jacobian_matrix.column_iter_mut().enumerate() {
            let Dk = self.model.eval_deriv(&self.location,self.current_parameters.as_slice()).at(idx).expect("Error evaluating model derivatives.");
            let Dk_c = &Dk * &self.current_linear_coeffs;
            let minus_ak : DVector<ScalarType> = U *(&U_t *(&Dk_c))- Dk_c;
            jacobian_col.copy_from(&minus_ak);
        }
        Some(jacobian_matrix)
    }
}
