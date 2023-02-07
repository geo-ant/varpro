use varpro::{prelude::*, model::errors::ModelError};
use nalgebra::{DMatrix,Dyn, DVector};


#[derive(Default,Clone)]
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
    fn basis_function_count(&self) -> usize {
        3
    }

    fn eval(&self, location : &nalgebra::DVector<f64>, parameters : &[f64])-> Result<nalgebra::DMatrix<f64>, Self::Error> {
        // parameters expected in this order
        // use unsafe to avoid bounds checks
        let tau1 = unsafe {parameters.get_unchecked(0)};
        let tau2 = unsafe {parameters.get_unchecked(1)};
        
        let f1 = location.map(|x| f64::exp(-x/tau1));
        let f2 = location.map(|x| f64::exp(-x/tau2));
        
        let mut basefuncs = unsafe {nalgebra::DMatrix::uninit(
            Dyn(location.len()),nalgebra::Dyn(3)).assume_init()};
        
        basefuncs.set_column(0, &f1);
        basefuncs.set_column(1, &f2);
        basefuncs.set_column(2, &DVector::from_element(location.len(), 1.));
        Ok(basefuncs)
    }

    fn eval_partial_deriv(&self, location: &nalgebra::DVector<f64>, parameters : &[f64],derivative_index : usize) -> Result<nalgebra::DMatrix<f64>, Self::Error> {
        // derivative index can be either 0,1 (corresponding to the linear parameters
        // tau1, tau2). Since only one of the basis functions depends on 
        // tau_i, we can simplify calculations here

        let tau = unsafe {parameters.get_unchecked(derivative_index)};
        let df = location.map(|x| -x/(tau*tau)*f64::exp(-x/tau));
        
        let mut basefuncs = DMatrix::zeros(location.len(), 3);

        basefuncs.set_column(derivative_index, &df);
        Ok(basefuncs)
    }
}
