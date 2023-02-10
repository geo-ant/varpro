use nalgebra::{DMatrix, DVector, Dyn};
use varpro::{model::errors::ModelError, prelude::*};

#[derive(Default, Clone)]
/// A separable model for double exponential decay
/// with a constant offset
/// f_j = c1*exp(-x_j/tau1) + c2*exp(-x_j/tau2) + c3
pub struct DoubleExpModelWithConstantOffsetSepModel {}

impl SeparableNonlinearModel<f64> for DoubleExpModelWithConstantOffsetSepModel {
    type Error = ModelError;

    #[inline]
    fn parameter_count(&self) -> usize {
        2
    }

    #[inline]
    fn base_function_count(&self) -> usize {
        3
    }

    fn eval(
        &self,
        location: &nalgebra::DVector<f64>,
        parameters: &[f64],
    ) -> Result<nalgebra::DMatrix<f64>, Self::Error> {
        // parameters expected in this order
        // use unsafe to avoid bounds checks
        let tau1 = unsafe { parameters.get_unchecked(0) };
        let tau2 = unsafe { parameters.get_unchecked(1) };

        let f1 = location.map(|x| f64::exp(-x / tau1));
        let f2 = location.map(|x| f64::exp(-x / tau2));

        let mut basefuncs = unsafe {
            nalgebra::DMatrix::uninit(Dyn(location.len()), nalgebra::Dyn(3)).assume_init()
        };

        basefuncs.set_column(0, &f1);
        basefuncs.set_column(1, &f2);
        basefuncs.set_column(2, &DVector::from_element(location.len(), 1.));
        Ok(basefuncs)
    }

    fn eval_partial_deriv(
        &self,
        location: &nalgebra::DVector<f64>,
        parameters: &[f64],
        derivative_index: usize,
    ) -> Result<nalgebra::DMatrix<f64>, Self::Error> {
        // derivative index can be either 0,1 (corresponding to the linear parameters
        // tau1, tau2). Since only one of the basis functions depends on
        // tau_i, we can simplify calculations here

        let tau = parameters[derivative_index];
        let df = location.map(|x| x / (tau * tau) * f64::exp(-x / tau));

        let mut basefuncs = DMatrix::zeros(location.len(), 3);

        basefuncs.set_column(derivative_index, &df);
        Ok(basefuncs)
    }
}

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::storage::Owned;
use nalgebra::{Matrix, Vector, Vector5, U5};

/// Implementation of double exponential decay with constant offset
/// f(x) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
/// using the levenberg_marquardt crate
pub struct DoubleExponentialDecayFittingWithOffsetLevmar {
    /// current problem paramters with layout (tau1,tau2,c1,c2,c3)
    params: Vector5<f64>,
    /// current independent variable
    x: DVector<f64>,
    /// the data (must have same length as x)
    y: DVector<f64>,
    /// precached calculations for the exponential terms so that the jacobian and residuals can
    /// 1) exp(-x/tau1)
    precalc_exp_tau1: DVector<f64>,
    /// 2) exp(-x/tau2)
    precalc_exp_tau2: DVector<f64>,
}

impl DoubleExponentialDecayFittingWithOffsetLevmar {
    /// create new fitting problem with data and initial guesses
    pub fn new(initial_guesses: &[f64], x: &DVector<f64>, y: &DVector<f64>) -> Self {
        assert_eq!(
            initial_guesses.len(),
            5,
            "Wrong parameter count. The 5 Parameters are: tau1, tau2, c1, c2, c3"
        );
        assert_eq!(y.len(), x.len(), "x and y must have same length");
        let parameters = Vector5::from_iterator(initial_guesses.iter().copied());
        let mut problem = Self {
            params: parameters,
            x: x.clone(),
            y: y.clone(),
            precalc_exp_tau1: DVector::from_element(x.len(), 0.), //will both be overwritten with correct vals
            precalc_exp_tau2: DVector::from_element(x.len(), 0.), //by the following call to set params
        };
        problem.set_params(&parameters);
        problem
    }
}

impl LeastSquaresProblem<f64, Dyn, U5> for DoubleExponentialDecayFittingWithOffsetLevmar {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U5>;

    fn set_params(&mut self, params: &Vector<f64, U5, Self::ParameterStorage>) {
        self.params = *params;
        let tau1 = params[0];
        let tau2 = params[1];

        // and do precalculations
        self.precalc_exp_tau1 = self.x.map(|x: f64| (-x / tau1).exp());
        self.precalc_exp_tau2 = self.x.map(|x: f64| (-x / tau2).exp());
    }

    fn params(&self) -> Vector<f64, U5, Self::ParameterStorage> {
        self.params
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        // get parameters from internal param storage
        //let tau1 = self.params[0];
        //let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        let c3 = self.params[4];

        // function values at the parameters
        let f = c1 * &self.precalc_exp_tau1
            + c2 * &self.precalc_exp_tau2
            + &DVector::from_element(self.x.len(), c3);

        // residuals: f-y
        Some(&f - &self.y)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U5, Self::JacobianStorage>> {
        // get parameters from internal param storage
        let tau1 = self.params[0];
        let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        // populate jacobian
        //let ncols = 5;
        let nrows = self.x.len();
        let mut jacobian = Matrix::<f64, Dyn, U5, Self::JacobianStorage>::zeros(nrows);

        jacobian.set_column(
            0,
            &(c1 / (tau1.powi(2)) * &(self.precalc_exp_tau1.component_mul(&self.x))),
        );
        jacobian.set_column(
            1,
            &(c2 / (tau2.powi(2)) * &(self.precalc_exp_tau2.component_mul(&self.x))),
        );
        jacobian.set_column(2, &self.precalc_exp_tau1);
        jacobian.set_column(3, &self.precalc_exp_tau2);
        jacobian.set_column(4, &DVector::from_element(self.x.len(), 1.));

        Some(jacobian)
    }
}