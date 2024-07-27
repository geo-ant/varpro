use nalgebra::{DVector, Dyn, OMatrix, OVector, U1};
use varpro::model::SeparableModel;
use varpro::{model::errors::ModelError, prelude::*};

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
#[derive(Clone, Debug, PartialEq)]
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
    params: OVector<f64, Dyn>,
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
    eval: OMatrix<f64, Dyn, Dyn>,
}

impl DoubleExpModelWithConstantOffsetSepModel {
    /// create a new model with the given x vector and initial guesses
    /// for the exponential decay constants tau_1 and tau_2
    pub fn new(x_vector: DVector<f64>, (tau1_guess, tau2_guess): (f64, f64)) -> Self {
        let x_len = x_vector.len();
        let mut ret = Self {
            x_vector,
            params: OVector::zeros_generic(Dyn(2), U1), //<-- will be overwritten by set_params
            eval: OMatrix::<f64, Dyn, Dyn>::zeros_generic(Dyn(x_len), Dyn(3)),
        };
        ret.set_params(OVector::from_column_slice_generic(
            Dyn(2),
            U1,
            &[tau1_guess, tau2_guess],
        ))
        .unwrap();
        ret
    }
}

impl SeparableNonlinearModel for DoubleExpModelWithConstantOffsetSepModel {
    /// we give our own mddel error here, but we
    /// could also have indicated that our calculations cannot
    /// fail by using [`std::convert::Infallible`].
    type Error = ModelError;
    /// the actual scalar type that our model uses for calculations
    type ScalarType = f64;

    #[inline]
    fn parameter_count(&self) -> usize {
        // regardless of the fact that we know at compile time
        // that the length is 2, we still have to return an instance
        // of that type
        2
    }

    #[inline]
    fn base_function_count(&self) -> usize {
        // same as above
        3
    }

    // we use this method not only to set the parameters inside the
    // model but we also cache some calculations. The advantage is that
    // we don't have to recalculate the exponential terms for either
    // the evaluations or the derivatives for the same parameters.
    fn set_params(&mut self, parameters: OVector<f64, Dyn>) -> Result<(), Self::Error> {
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
            .set_column(2, &DVector::from_element(self.x_vector.len(), 1.));
        Ok(())
    }

    fn params(&self) -> OVector<f64, Dyn> {
        self.params.clone()
    }

    // since we cached the model evaluation, we can just return
    // it here
    fn eval(&self) -> Result<OMatrix<f64, Dyn, Dyn>, Self::Error> {
        Ok(self.eval.clone())
    }

    // here we take advantage of the cached calculations
    // so that we do not have to recalculate the exponential
    // to calculate the derivative.
    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<nalgebra::OMatrix<f64, Dyn, Dyn>, Self::Error> {
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
        let mut derivatives = OMatrix::zeros_generic(Dyn(location.len()), Dyn(3));

        derivatives.set_column(derivative_index, &df);
        Ok(derivatives)
    }

    fn output_len(&self) -> usize {
        // this is how we give a length that is only known at runtime.
        // We wrap it in a `Dyn` instance.
        self.x_vector.len()
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

/// example function from the matlab code that was published as part of the
/// O'Leary paper here: https://www.cs.umd.edu/~oleary/software/varpro/
/// The model function is
///
///   f(t) = c1 exp(-alpha2 t)*cos(alpha3 t)
///         + c2 exp(-alpha1 t)*cos(alpha2 t)
/// # performance
/// this model does reuse calculations but I think it could be optimized
/// further
#[derive(Clone, Debug, PartialEq)]
pub struct OLearyExampleModel {
    /// the t vector
    t: DVector<f64>,
    /// the current parameters (alpha1,alpha2,alpha3,alpha4)
    alpha: OVector<f64, Dyn>,
    /// the current evaluation of the model
    phi: OMatrix<f64, Dyn, Dyn>,
}

impl OLearyExampleModel {
    /// create a new model with the given t vector and initial guesses
    pub fn new(t: DVector<f64>, initial_guesses: OVector<f64, Dyn>) -> Self {
        let t_len = t.len();
        let mut ret = Self {
            t,
            alpha: initial_guesses.clone(),
            phi: OMatrix::<f64, Dyn, Dyn>::zeros_generic(Dyn(t_len), Dyn(2)),
        };
        ret.set_params(initial_guesses).unwrap();
        ret
    }
}

impl SeparableNonlinearModel for OLearyExampleModel {
    type ScalarType = f64;
    type Error = ModelError;

    fn parameter_count(&self) -> usize {
        3
    }

    fn base_function_count(&self) -> usize {
        2
    }

    fn output_len(&self) -> usize {
        self.t.len()
    }

    fn set_params(
        &mut self,
        parameters: OVector<Self::ScalarType, Dyn>,
    ) -> Result<(), Self::Error> {
        self.alpha = parameters;
        let alpha1 = self.alpha[0];
        let alpha2 = self.alpha[1];
        let alpha3 = self.alpha[2];

        let f1 = self.t.map(|t| f64::exp(-alpha2 * t) * f64::cos(alpha3 * t));
        let f2 = self.t.map(|t| f64::exp(-alpha1 * t) * f64::cos(alpha2 * t));

        self.phi = OMatrix::<f64, Dyn, Dyn>::from_columns(&[f1, f2]);
        Ok(())
    }

    fn params(&self) -> OVector<Self::ScalarType, Dyn> {
        self.alpha.clone()
    }

    fn eval(&self) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error> {
        Ok(self.phi.clone())
    }

    fn eval_partial_deriv(
        &self,
        derivative_index: usize,
    ) -> Result<OMatrix<Self::ScalarType, Dyn, Dyn>, Self::Error> {
        let mut derivs = OMatrix::<f64, Dyn, Dyn>::zeros_generic(Dyn(self.t.len()), Dyn(2));
        let alpha1 = self.alpha[0];
        let alpha2 = self.alpha[1];
        let alpha3 = self.alpha[2];

        // from the original matlab impl:
        // % The nonzero partial derivatives of Phi with respect to alpha are
        // %              d Phi_1 / d alpha_2 ,
        // %              d Phi_1 / d alpha_3 ,
        // %              d Phi_2 / d alpha_1 ,
        // %              d Phi_2 / d alpha_2 ,
        // % and this determines Ind.
        // % The ordering of the columns of Ind is arbitrary but must match dPhi.

        // % Evaluate the four nonzero partial derivatives of Phi at each of
        // % the data points and store them in dPhi.

        // dPhi 1 / dalpha 2 = -t .* Phi(:,1);
        // dPhi 1 / dalpha 3 = -t .* exp(-alpha(2)*t).*sin(alpha(3)*t);
        // dPhi 2 / dalpha 1 = -t .* Phi(:,2);
        // dPhi 2 / dalpha 2 = -t .* exp(-alpha(1)*t).*sin(alpha(2)*t);

        match derivative_index {
            0 => {
                // d/d alpha1
                derivs.set_column(1, &self.phi.column(1).component_mul(&(-1. * &self.t)));
            }
            1 => {
                // d/d alpha2
                derivs.set_column(0, &self.phi.column(0).component_mul(&(-1. * &self.t)));
                derivs.set_column(
                    1,
                    &self
                        .t
                        .map(|t| -t * (-alpha1 * t).exp() * (alpha2 * t).sin()),
                );
            }
            2 => {
                // d/d alpha3
                derivs.set_column(
                    0,
                    &self
                        .t
                        .map(|t| -t * (-alpha2 * t).exp() * (alpha3 * t).sin()),
                );
            }
            _ => {
                unreachable!("derivative index must be 0,1,2");
            }
        }

        Ok(derivs)
    }
}

/// the oleary example model as above but this time it is generated using
/// the separable model builder
pub fn o_leary_example_model(t: DVector<f64>, initial_guesses: Vec<f64>) -> SeparableModel<f64> {
    fn phi1(t: &DVector<f64>, alpha2: f64, alpha3: f64) -> DVector<f64> {
        t.map(|t| f64::exp(-alpha2 * t) * f64::cos(alpha3 * t))
    }
    fn phi2(t: &DVector<f64>, alpha1: f64, alpha2: f64) -> DVector<f64> {
        t.map(|t| f64::exp(-alpha1 * t) * f64::cos(alpha2 * t))
    }

    SeparableModelBuilder::new(["alpha1", "alpha2", "alpha3"])
        .initial_parameters(initial_guesses)
        .independent_variable(t)
        // phi 1
        .function(["alpha2", "alpha3"], phi1)
        .partial_deriv("alpha2", |t: &DVector<f64>, alpha2: f64, alpha3: f64| {
            phi1(t, alpha2, alpha3).component_mul(&(-1. * t))
        })
        .partial_deriv("alpha3", |t: &DVector<f64>, alpha2: f64, alpha3: f64| {
            t.map(|t| -t * (-alpha2 * t).exp() * (alpha3 * t).sin())
        })
        .function(["alpha1", "alpha2"], phi2)
        .partial_deriv("alpha1", |t: &DVector<f64>, alpha1: f64, alpha2: f64| {
            phi2(t, alpha1, alpha2).component_mul(&(-1. * t))
        })
        .partial_deriv("alpha2", |t: &DVector<f64>, alpha1: f64, alpha2: f64| {
            t.map(|t| -t * (-alpha1 * t).exp() * (alpha2 * t).sin())
        })
        .build()
        .unwrap()
}
