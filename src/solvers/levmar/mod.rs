use crate::model::SeparableModel;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::{ComplexField, DMatrix, DVector, Dynamic, Matrix, Scalar, Vector, SVD};

mod builder;
#[cfg(test)]
mod test;
mod weights;

use crate::solvers::levmar::weights::Weights;
pub use builder::LevMarProblemBuilder;
pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use num_traits::Float;
use std::ops::Mul;

/// helper structure that stores the cached calculations,
/// which are carried out by the LevMarProblem on setting the parameters
#[derive(Debug, Clone)]
struct CachedCalculations<ScalarType: Scalar + ComplexField> {
    /// The current residual of model function values belonging to the current parameters
    current_residuals: DVector<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    current_svd: SVD<ScalarType, Dynamic, Dynamic>,
    /// the linear coefficients `$\vec{c}$` providing the current best fit
    linear_coefficients: DVector<ScalarType>,
}

/// This is a the problem of fitting the separable model to data in a form that the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate can use it to
/// perform the least squares fit.
/// # Construction
/// Use the [LevMarProblemBuilder](self::builder::LevMarProblemBuilder) to create an instance of a
/// levmar problem.
/// # Usage
/// After obtaining an instance of `LevMarProblem` we can pass it to the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// structure of the levenberg_marquardt crate for minimization. Refer to the documentation of the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) for an overview. A usage example
/// is provided in this crate documentation as well. The [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// solver is reexported by this module as [LevMarSolver](self::LevMarSolver) for naming consistency.
#[derive(Clone)]
pub struct LevMarProblem<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField,
{
    /// the independent variable `\vec{x}` (location parameter)
    x: DVector<ScalarType>,
    /// the *weighted* data vector to which to fit the model `$\vec{y}_w$`
    /// **Attention** the data vector is weighted with the weights if some weights
    /// where provided (otherwise it is unweighted)
    y_w: DVector<ScalarType>,
    /// current parameters that the optimizer is operating on
    model_parameters: Vec<ScalarType>,
    /// a reference to the separable model we are trying to fit to the data
    model: &'a SeparableModel<ScalarType>,
    /// truncation epsilon for SVD below which all singular values are assumed zero
    svd_epsilon: ScalarType::RealField,
    /// the weights of the data. If none are given, the data is not weighted
    /// If weights were provided, the builder has checked that the weights have the
    /// correct dimension for the data
    weights: Weights<ScalarType>,
    /// the currently cached calculations belonging to the currently set model parameters
    /// those are updated on set_params. If this is None, then it indicates some error that
    /// is propagated on to the levenberg-marquardt crate by also returning None results
    /// by residuals() and/or jacobian()
    cached: Option<CachedCalculations<ScalarType>>,
}

impl<'a, ScalarType: Scalar + ComplexField> LevMarProblem<'a, ScalarType> {
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    /// # Returns
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    pub fn linear_coefficients(&self) -> Option<DVector<ScalarType>> {
        self.cached
            .as_ref()
            .map(|cache| cache.linear_coefficients.clone())
    }
}

impl<'a, ScalarType> LeastSquaresProblem<ScalarType, Dynamic, Dynamic>
    for LevMarProblem<'a, ScalarType>
where
    ScalarType: Scalar + ComplexField,
    ScalarType::RealField: Mul<ScalarType, Output = ScalarType> + Float,
{
    type ResidualStorage = Owned<ScalarType, Dynamic>;
    type JacobianStorage = Owned<ScalarType, Dynamic, Dynamic>;
    type ParameterStorage = Owned<ScalarType, Dynamic>;

    #[allow(non_snake_case)]
    /// Set the (nonlinear) model parameters `$\vec{\alpha}$` and update the internal state of the
    /// problem accordingly. The parameters are expected in the same order that the parameter
    /// names were provided in at model creation. So if we gave `&["tau","beta"]` as parameters at
    /// model creation, the function expects the layout of the parameter vector to be `$\vec{\alpha}=(\tau,\beta)^T$`.
    fn set_params(&mut self, params: &Vector<ScalarType, Dynamic, Self::ParameterStorage>) {
        self.model_parameters = params.iter().cloned().collect();
        // matrix of weighted model function values
        let Phi_w = self
            .model
            .eval(&self.x, self.model_parameters.as_slice())
            .ok()
            .map(|Phi| &self.weights * Phi);

        // calculate the svd
        let svd_epsilon = self.svd_epsilon;
        let current_svd = Phi_w.as_ref().map(|Phi_w| Phi_w.clone().svd(true, true));
        let linear_coefficients = current_svd
            .as_ref()
            .and_then(|svd| svd.solve(&self.y_w, svd_epsilon).ok());

        // calculate the residuals
        let current_residuals = Phi_w
            .zip(linear_coefficients.as_ref())
            .map(|(Phi_w, coeff)| &self.y_w - Phi_w * coeff);

        // if everything was successful, update the cached calculations, otherwise set the cache to none
        if let (Some(current_residuals), Some(current_svd), Some(linear_coefficients)) =
            (current_residuals, current_svd, linear_coefficients)
        {
            self.cached = Some(CachedCalculations {
                current_residuals,
                current_svd,
                linear_coefficients,
            })
        } else {
            self.cached = None;
        }
    }

    /// Retrieve the (nonlinear) model parameters as a vector `$\vec{\alpha}$`.
    /// The order of the parameters in the vector is the same as the order of the parameter
    /// names given on model creation. E.g. if the parameters at model creation where given as
    /// `&["tau","beta"]`, then the returned vector is `$\vec{\alpha} = (\tau,\beta)^T$`, i.e.
    /// the value of parameter `$\tau$` is at index `0` and the value of `$\beta$` at index `1`.
    fn params(&self) -> Vector<ScalarType, Dynamic, Self::ParameterStorage> {
        DVector::from(self.model_parameters.clone())
    }

    /// Calculate the residual vector `$\vec{r}_w$` of *weighted* residuals at every location `$\vec{x}$`.
    /// The residual is calculated from the data `\vec{y}` as `$\vec{r}_w(\vec{\alpha}) = W\cdot(\vec{y}-\vec{f}(\vec{x},\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
    /// where `$\vec{f}(\vec{x},\vec{\alpha},\vec{c})$` is the model function evaluated at the currently
    /// set nonlinear parameters `$\vec{\alpha}$` and the linear coefficients `$\vec{c}(\vec{\alpha})$`. The VarPro
    /// algorithm calculates `$\vec{c}(\vec{\alpha})$` as the coefficients that provide the best linear least squares
    /// fit, given the current `$\vec{\alpha}$`. For more info on the math of VarPro, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn residuals(&self) -> Option<Vector<ScalarType, Dynamic, Self::ResidualStorage>> {
        self.cached
            .as_ref()
            .map(|cached| cached.current_residuals.clone())
    }

    #[allow(non_snake_case)]
    /// Calculate the Jacobian matrix of the *weighted* residuals `$\vec{r}_w(\vec{\alpha})$`.
    /// For more info on how the Jacobian is calculated in the VarPro algorithm, see
    /// e.g. [here](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/).
    fn jacobian(&self) -> Option<Matrix<ScalarType, Dynamic, Dynamic, Self::JacobianStorage>> {
        // TODO (Performance): make this more efficient by parallelizing

        if let Some(CachedCalculations {
            current_residuals: _,
            current_svd,
            linear_coefficients,
        }) = self.cached.as_ref()
        {
            let mut jacobian_matrix = unsafe {
                DMatrix::<ScalarType>::new_uninitialized(
                    self.y_w.len(),
                    self.model.parameter_count(),
                )
                .assume_init()
            };

            let U = current_svd.u.as_ref()?; // will return None if this was not calculated
            let U_t = U.transpose();

            //let Sigma_inverse : DMatrix<ScalarType::RealField> = DMatrix::from_diagonal(&self.current_svd.singular_values.map(|val|val.powi(-1)));
            //let V_t = self.current_svd.v_t.as_ref().expect("Did not calculate U of SVD. This should not happen and indicates a logic error in the library.");

            for (k, mut jacobian_col) in jacobian_matrix.column_iter_mut().enumerate() {
                // weighted derivative matrix
                let Dk = &self.weights
                    * self
                        .model
                        .eval_deriv(&self.x, self.model_parameters.as_slice())
                        .at(k)
                        .ok()?; // will return none if this could not be calculated
                let Dk_c = &Dk * linear_coefficients;
                let minus_ak: DVector<ScalarType> = U * (&U_t * (&Dk_c)) - Dk_c;
                //for non-approximate jacobian we require our scalar type to be a real field (or maybe we can finagle it with clever trait bounds)
                //let Dk_t_rw : DVector<ScalarType> = &Dk.transpose()*self.residuals().as_ref().expect("Residuals must produce result");
                //let _minus_bk : DVector<ScalarType> = U*(&Sigma_inverse*(V_t*(&Dk_t_rw)));
                jacobian_col.copy_from(&(minus_ak));
            }
            Some(jacobian_matrix)
        } else {
            None
        }
    }
}
