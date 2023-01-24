use crate::common::linspace;
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem};
use nalgebra::storage::Owned;
use nalgebra::{DVector, Dynamic, Matrix, Vector, Vector5, U5};

use approx::assert_relative_eq;

/// Implementation of double exponential decay with constant offset
/// f(x) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
/// using the levenberg_marquardt crate
pub struct DoubleExponentialDecayFittingWithOffset {
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

impl DoubleExponentialDecayFittingWithOffset {
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

impl LeastSquaresProblem<f64, Dynamic, U5> for DoubleExponentialDecayFittingWithOffset {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

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

    fn jacobian(&self) -> Option<Matrix<f64, Dynamic, U5, Self::JacobianStorage>> {
        // get parameters from internal param storage
        let tau1 = self.params[0];
        let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        // populate jacobian
        //let ncols = 5;
        let nrows = self.x.len();
        let mut jacobian = Matrix::<f64, Dynamic, U5, Self::JacobianStorage>::zeros(nrows);

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

#[test]
// sanity check my calculations above
fn sanity_check_jacobian_of_levenberg_marquardt_problem_is_correct() {
    let x = linspace(0., 12.5, 20);
    let tau1 = 2.;
    let tau2 = 4.;
    let c1 = 2.;
    let c2 = 4.;
    let c3 = 0.2;
    let f = x.map(|x: f64| c1 * (-x / tau1).exp() + c2 * (-x / tau2).exp() + c3);

    let mut problem = DoubleExponentialDecayFittingWithOffset::new(
        &[0.5 * tau1, 1.5 * tau2, 2. * c1, 3. * c2, 3. * c3],
        &x,
        &f,
    );

    // Let `problem` be an instance of `LeastSquaresProblem`
    let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-5);
}
