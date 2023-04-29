use nalgebra::{DVector, Dyn, OMatrix, OVector, Vector2, U2, U3};
use varpro::{model::errors::ModelError, prelude::*};

#[derive(Clone)]
/// A separable model for double exponential decay
/// with a constant offset
/// f_j = c1*exp(-x_j/tau1) + c2*exp(-x_j/tau2) + c3
/// this is a handcrafted model which uses caching for
/// maximum performance.
///
/// This is an example of how to implement a separable model
/// using the trait directly without using the builder.
/// This allows us to use caching of the intermediate results
/// to calculate the derivatives more efficiently.
pub struct DoubleExpModelWithConstantOffsetSepModel {
    /// the x vector associated with this model.
    /// We make this configurable to enable models to
    /// work with different x coordinates, but this is
    /// not an inherent requirement. We don't have to have a
    /// field like this. The only requirement given by the trait
    /// is that the model can be queried about the length
    /// of its output
    x_vector: DVector<f64>,
    /// current parameters [tau1,tau2] of the exponential
    /// functions
    params: Vector2<f64>,
    /// cached evaluation of the model
    /// the matrix columns contain the complete evaluation
    /// of the model. That is the first column contains the
    /// exponential exp(-x/tau), the second column contains
    /// exp(-x/tau) both evaluated on the x vector. The third
    /// column contains a column of straight ones for the constant
    /// offset.
    ///
    /// This value is calculated in the set_params method, which is
    /// the only method with mutable access to the model state.
    eval: OMatrix<f64, Dyn, U3>,
}

impl DoubleExpModelWithConstantOffsetSepModel {
    /// create a new model with the given x vector and initial guesses
    /// for the exponential decay constants tau_1 and tau_2
    pub fn new(x_vector: DVector<f64>, (tau1_guess, tau2_guess): (f64, f64)) -> Self {
        let x_len = x_vector.len();
        let mut ret = Self {
            x_vector,
            params: Vector2::zeros(), //<-- will be overwritten by set_params
            eval: OMatrix::<f64, Dyn, U3>::zeros_generic(Dyn(x_len), U3),
        };
        ret.set_params(Vector2::new(tau1_guess, tau2_guess))
            .unwrap();
        ret
    }
}

impl SeparableNonlinearModel for DoubleExpModelWithConstantOffsetSepModel {
    /// we give our own mddel error here, but we
    /// could also have indicated that our calculations cannot
    /// fail by using [`std::convert::Infallible`].
    type Error = ModelError;
    /// We use a compile time constant (2) to indicate the
    /// number of parameters at compile time
    type ParameterDim = U2;
    /// the model dimension is the number of base functions.
    /// We also use a type to indicate its size at compile time
    type ModelDim = U3;
    /// the ouput dim is the number of elements that each base function
    /// produces. We have made this dynamic here since it has
    /// the same length as the x vector given to the model. We made it
    /// so that the length of the x vector is only runtime known.
    /// We could just as well have made it compile time known.
    type OutputDim = Dyn;
    /// the actual scalar type that our model uses for calculations
    type ScalarType = f64;

    #[inline]
    fn parameter_count(&self) -> U2 {
        // regardless of the fact that we know at compile time
        // that the length is 2, we still have to return an instance
        // of that type
        U2 {}
    }

    #[inline]
    fn base_function_count(&self) -> U3 {
        // same as above
        U3 {}
    }

    // we use this method not only to set the parameters inside the
    // model but we also cache some calculations. The advantage is that
    // we don't have to recalculate the exponential terms for either
    // the evaluations or the derivatives for the same parameters.
    fn set_params(&mut self, parameters: Vector2<f64>) -> Result<(), Self::Error> {
        // even if it is not the only thing we do, we still
        // have to update the internal parameters of the model
        self.params = parameters;

        // parameters expected in this order
        // use unsafe to avoid bounds checks
        let tau1 = unsafe { self.params.get_unchecked(0) };
        let tau2 = unsafe { self.params.get_unchecked(1) };

        // the exponential exp(-x/tau1)
        let f1 = self.x_vector.map(|x| f64::exp(-x / tau1));
        // the exponential exp(-x/tau2)
        let f2 = self.x_vector.map(|x| f64::exp(-x / tau2));

        self.eval.set_column(0, &f1);
        self.eval.set_column(1, &f2);
        self.eval
            .set_column(2, &DVector::from_element(location.len(), 1.));
        Ok(())
    }

    fn params(&self) -> OVector<f64, Self::ParameterDim> {
        self.params.clone()
    }

    // since we cached the model evaluation, we can just return
    // it here
    fn eval(&self) -> Result<OMatrix<f64, Dyn, Self::ModelDim>, Self::Error> {
        Ok(self.eval.clone())
    }

    // here we take advantage of the cached calculations
    // so that we do not have to recalculate the exponential
    // to calculate the derivative.
    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<nalgebra::OMatrix<f64, Dyn, Self::ModelDim>, Self::Error> {
        let location = &self.x_vector;
        let parameters = &self.params;
        // derivative index can be either 0,1 (corresponding to the linear parameters
        // tau1, tau2). Since only one of the basis functions depends on
        // tau_i, we can simplify calculations here

        let tau = parameters[derivative_index];
        // the only nonzero derivative is the derivative of the exp(-x/tau) for
        // the corresponding tau at derivative_index
        // we can use the precalculated results so we don't have to use the
        // exponential function again
        let df = location
            .map(|x| x / (tau * tau))
            .component_mul(&self.eval.column(derivative_index));

        // two of the columns are always zero when we differentiate
        // with respect to tau_1 or tau_2. Remember the constant term
        // also occupies one column and will always be zero when differentiated
        // with respect to the nonlinear params of the model
        let mut derivatives = OMatrix::zeros_generic(Dyn(location.len()), U3);

        derivatives.set_column(derivative_index, &df);
        Ok(derivatives)
    }

    fn output_len(&self) -> Self::OutputDim {
        // this is how we give a length that is only known at runtime.
        // We wrap it in a `Dyn` instance.
        Dyn(self.x_vector.len())
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
            + DVector::from_element(self.x.len(), c3);

        // residuals: f-y
        Some(f - &self.y)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U5, Self::JacobianStorage>> {
        // get parameters from internal param storage
        let tau1 = self.params[0];
        let tau2 = self.params[1];
        let c1 = self.params[2];
        let c2 = self.params[3];
        // populate jacobian
        let nrows = self.x.len();
        let mut jacobian = Matrix::<f64, Dyn, U5, Self::JacobianStorage>::zeros(nrows);

        jacobian.set_column(
            0,
            &(c1 / (tau1 * tau1) * &(self.precalc_exp_tau1.component_mul(&self.x))),
        );
        jacobian.set_column(
            1,
            &(c2 / (tau2 * tau2) * &(self.precalc_exp_tau2.component_mul(&self.x))),
        );
        jacobian.set_column(2, &self.precalc_exp_tau1);
        jacobian.set_column(3, &self.precalc_exp_tau2);
        jacobian.set_column(4, &DVector::from_element(self.x.len(), 1.));

        Some(jacobian)
    }
}
