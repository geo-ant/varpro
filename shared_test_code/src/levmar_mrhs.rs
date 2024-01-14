use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::DVectorSlice;
use nalgebra::Dyn;
use nalgebra::Matrix;
use nalgebra::OMatrix;
use nalgebra::Owned;
use nalgebra::Vector;
use nalgebra::U1;
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
        assert_eq!(
            x.len(),
            data.nrows(),
            "x vector must have same number of rows as data"
        );
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
        debug_assert_eq!(
            params.len(),
            self.Y.ncols() * 3 + 2,
            "number of parameters does not match the dataset and the decay model"
        );
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
        let param_count = self.C.len() + 2;
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
        let R = &self.Y - &self.Phi * &self.C;
        let new_nrows = Dyn(R.nrows() * R.ncols());
        Some(R.reshape_generic(new_nrows, U1))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        // let mut jacobian = DMatrix::<f64>::zeros(self.
        let mut dPhi_dalpha1 = OMatrix::<_, Dyn, U3>::zeros_generic(Dyn(self.Phi.nrows()), U3);
        dPhi_dalpha1.set_column(
            0,
            &(1. / (self.alpha[0] * self.alpha[0]) * &self.Phi.column(0).component_mul(&self.x)),
        );
        let mut dPhi_dalpha2 = OMatrix::<_, Dyn, U3>::zeros_generic(Dyn(self.Phi.nrows()), U3);
        dPhi_dalpha2.set_column(
            1,
            &(1. / (self.alpha[1] * self.alpha[1]) * &self.Phi.column(1).component_mul(&self.x)),
        );
        let mut jac_block = DMatrix::from_element(self.Phi.nrows(), 3, 1.);
        jac_block.set_column(0, &self.Phi.column(0));
        jac_block.set_column(1, &self.Phi.column(1));

        let x_len = self.Y.nrows();
        let total_data_len = x_len * self.Y.ncols();
        let mut jacobian = DMatrix::zeros(self.Y.nrows() * self.Y.ncols(), self.C.len() + 2);
        debug_assert_eq!(
            self.Y.ncols() * 3,
            self.C.len(),
            "we need exactly 3 linear parameters per dataset"
        );

        for (row, col) in (3..jacobian.nrows())
            .into_iter()
            .step_by(3)
            .zip((0..total_data_len).step_by(x_len))
        {
            jacobian
                .view_mut((row, col), (x_len, 3))
                .copy_from(&jac_block);
        }
        Some(jacobian)
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
