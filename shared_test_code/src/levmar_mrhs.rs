use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::DVectorSlice;
use nalgebra::Dyn;
use nalgebra::Matrix;
use nalgebra::OMatrix;
use nalgebra::Owned;
use nalgebra::Vector;
use nalgebra::U3;

/// double exponential model with constant offset where the nonlinear
/// parameters are shared across the dataset and the linear parameters are not
/// f_s(x) = c_{s,1} * exp(-x/alpha_1) + c_{s,2} * exp(-x/alpha_2) + c_{s,3},
/// where s is the index of the element in the dataset
pub struct DoubleExponentialModelWithConstantOffsetLevmarMrhs {
    /// the independent variable. Also defines the output length of the model.
    x: DVector<f64>,
    /// the nonlinear parameters [alpha1,alpha2]
    alpha: [f64; 2],
    /// the matrix of datasets organized as columns
    ///       |       |
    /// Y = (y_1,...,y_s,...)
    ///       |       |
    Y: DMatrix<f64>,
    /// the matrix of linear coefficients organized as columns
    ///       |       |
    /// C = (c_1,...,c_s,...)
    ///       |       |
    C: OMatrix<f64, Dyn, U3>,
    /// the precalculated function matrix
    /// where the first exponential comes first (alpha1),
    /// then the second exponential and finally
    /// a column of all ones for the constant offset
    Phi: OMatrix<f64, Dyn, U3>,
}

impl DoubleExponentialModelWithConstantOffsetLevmarMrhs {
    pub fn new(x: DVector<f64>, data: DMatrix<f64>, initial_params: DVector<f64>) -> Self {
        let mut Phi = OMatrix::zeros_generic(Dyn(x.len()), U3);
        // the third column of only ones. We never touch this one again
        Phi.column_mut(2).copy_from_slice(&vec![1f64; x.len()]);
        let mut me = Self {
            C: OMatrix::zeros_generic(Dyn(data.len() * x.len()), U3), //<- this will be overwritten by set params
            x,
            alpha: [initial_params[0], initial_params[1]],
            Y: data,
            Phi,
        };
        me.set_params(&initial_params);
        me
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for DoubleExponentialModelWithConstantOffsetLevmarMrhs {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    /// parameters are organized as
    /// alpha_1,alpha_2, c_{1,1},c_{1,2},c_{1,3},...,c_{s,1},c_{s,2},c_{s,3},...
    fn set_params(&mut self, params: &Vector<f64, Dyn, Self::ParameterStorage>) {
        //tau1
        self.alpha[0] = params[0];
        //tau2
        self.alpha[1] = params[1];

        // and do precalculations
        self.Phi
            .column_mut(0)
            .copy_from(&self.x.map(|x: f64| (-x / self.alpha[0]).exp()));
        self.Phi
            .column_mut(0)
            .copy_from(&self.x.map(|x: f64| (-x / self.alpha[1]).exp()));
        // and load the matrix of linear coefficients
        // I have a test down below that shows this should actually work
        self.C.copy_from_slice(&params.as_slice()[2..]);
    }

    fn params(&self) -> Vector<f64, Dyn, Self::ParameterStorage> {
        let param_count = self.C.data.len() + 2;
        let params = DVector::from_iterator(
            param_count,
            self.alpha
                .iter()
                .cloned()
                .chain(self.C.as_slice().into_iter().cloned()),
        );
        params
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        todo!()
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        todo!()
    }
}

#[test]
// make sure that the way I copy the parameters into a matrix works as
// I think it does
fn matrix_is_colum_major_layout() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let mut mat = OMatrix::zeros_generic(Dyn(2), U3);
    mat.copy_from_slice(data);
    assert_eq!(mat.column(0).as_slice(), &[1., 2.]);
    assert_eq!(mat.column(1).as_slice(), &[3., 4.]);
    assert_eq!(mat.column(2).as_slice(), &[5., 6.]);
    // also assert that the reverse is true
    assert_eq!(mat.as_slice(), data);
}
