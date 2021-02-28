use levenberg_marquardt::{LeastSquaresProblem, differentiate_numerically};
use nalgebra::{ Dynamic, Scalar, Matrix, Vector, DVector, DMatrix, RowDVector};
use nalgebra::storage::Owned;
use crate::common::linspace;

use approx::assert_relative_eq;

/// Implementation of double exponential decay with constant offset
/// f(x) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
/// using the levenberg_marquardt crate
pub struct DoubleExponentialDecayFittingWithOffset
{
    /// current problem paramters with layout (tau1,tau2,c1,c2,c3)
    params : DVector<f64>,
    /// current independent variable
    x : DVector<f64>,
    /// the data (must have same length as x)
    y : DVector<f64>,
    /// precached calculations for the exponential terms so that the jacobian and residuals can
    /// share these calculations
    /// 1) a vector of ones
    vector_of_ones : DVector<f64>,
    /// 2) exp(-x/tau1)
    precalc_exp_tau1 : DVector<f64>,
    /// 3) exp(-x/tau2)
    precalc_exp_tau2 : DVector<f64>,
}

impl DoubleExponentialDecayFittingWithOffset {
    /// create new fitting problem with data and initial guesses
    pub fn new(initial_guesses : &[f64], x : &DVector<f64>, y:&DVector<f64>) -> Self{
        assert_eq!(initial_guesses.len(),5,"Wrong parameter count. The 5 Parameters are: tau1, tau2, c1, c2, c3");
        assert_eq!(y.len(),x.len(),"x and y must have same length");
        let parameters =  DVector::from(initial_guesses.to_vec());
        let mut problem = Self {
            params : parameters.clone(),
            x : x.clone(),
            y : y.clone(),
            vector_of_ones : DVector::from_element(x.len(),1.),
            precalc_exp_tau1 : DVector::from_element(x.len(),0.), //will both be overwritten with correct vals
            precalc_exp_tau2 : DVector::from_element(x.len(),0.), //by the following call to set params
        };
        problem.set_params(&parameters);
        problem
    }
}

impl LeastSquaresProblem<f64,Dynamic,Dynamic> for DoubleExponentialDecayFittingWithOffset {
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
    type ParameterStorage = Owned<f64, Dynamic>;

    fn set_params(&mut self, params: &DVector<f64>) {
        self.params = params.clone();
        let tau1 = params[0];
        let tau2 = params[1];

        // and do precalculations
        self.precalc_exp_tau1 = self.x.map(|x:f64|(-x/tau1).exp());
        self.precalc_exp_tau2 = self.x.map(|x:f64|(-x/tau2).exp());
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        // get parameters from internal param storage
        let tau1 = self.params[0];
        let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        let c3 = self.params[4];

        // function values at the parameters
        //let f = self.x.map(|x|c1*(-x/tau1).exp()+c2*(-x/tau2).exp()+c3);
        let f = c1*&self.precalc_exp_tau1 +c2*&self.precalc_exp_tau2 + c3*&self.vector_of_ones;

        // residuals: f-y
        Some(&f- &self.y)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        // get parameters from internal param storage
        let tau1 = self.params[0];
        let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        let c3 = self.params[4];
        // populate jacobian
        let ncols = 5;
        let nrows = self.x.len();
        let mut jacobian :DMatrix<f64> = unsafe {DMatrix::new_uninitialized(nrows,ncols)};

        // for (k, x) in self.x.iter().enumerate() {
        //     jacobian.set_row(k, &RowDVector::from_iterator(5,vec!{
        //         c1*x*(-x/tau1).exp()/(tau1.powi(2)), //df/dtau1
        //         c2*x*(-x/tau2).exp()/(tau2.powi(2)), //df/dtau1
        //         (-x/tau1).exp(),    //df/c1
        //         (-x/tau2).exp(),     //df/dc2
        //             1.  //df/dc3
        //     }.into_iter()));
        // }
        jacobian.set_column(0,&(c1/(tau1.powi(2))*&(self.precalc_exp_tau1.component_mul(&self.x))));
        jacobian.set_column(1,&(c2/(tau2.powi(2))*&(self.precalc_exp_tau2.component_mul(&self.x))));
        jacobian.set_column(2,&self.precalc_exp_tau1);
        jacobian.set_column(3,&self.precalc_exp_tau2);
        jacobian.set_column(4,&self.vector_of_ones);

        Some(jacobian)
    }
}

#[test]
fn jacobian_of_levenberg_marquardt_problem_is_correct() {
    let x = linspace(0.,12.5,20);
    let tau1 = 2.;
    let tau2  = 4.;
    let c1 = 2.;
    let c2 = 4.;
    let c3 = 0.2;
    let f = x.map(|x: f64|c1*(-x/tau1).exp()+c2*(-x/tau2).exp()+c3);

    let mut problem= DoubleExponentialDecayFittingWithOffset::new(&[0.5*tau1,1.5*tau2,2.*c1,3.*c2,3.*c3],&x,&f);

    // Let `problem` be an instance of `LeastSquaresProblem`
    let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-5);
}