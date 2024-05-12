use crate::{
    prelude::SeparableNonlinearModel,
    util::{to_vector, Weights},
};
use nalgebra::{
    allocator::Allocator, ComplexField, DVector, DefaultAllocator, Dim, DimAdd, DimMin, DimSub,
    Dyn, Matrix, OMatrix, OVector, RealField, Scalar, VectorView, U0, U1,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use thiserror::Error as ThisError;

use self::numeric_traits::CastF64;

#[cfg(any(test, doctest))]
mod test;

/// contains some helper traits for numeric conversions that
/// are too specialized for the num_traits crate
pub mod numeric_traits;

/// Information about an error that can occur during calculation of the
/// of the fit statistics.
#[derive(Debug, Clone, ThisError)]
pub(crate) enum Error<ModelError: std::error::Error> {
    /// Model returned error when it was evaluated
    #[error("Model returned error when it was evaluated:{}", .0)]
    ModelEvaluation(#[from] ModelError),
    /// Fit is underdetermined
    #[error("Fit is underdetermined")]
    Underdetermined,
    #[error("Floating point unable to capture integral value {}", .0)]
    /// the floating point type was unable to capture an integral value
    IntegerToFloatConversion(usize),
    /// Failed to calculate the inverse of a matrix
    #[error("Matrix inversion error")]
    MatrixInversion,
    #[error("Fit statistics for multiple right hand sides are currently not supported")]
    MrhsUnsupported,
}

/// This structure contains some additional statistical information
/// about the fit, such as errors on the parameters and other useful
/// information to assess the quality of the fit.
///
/// # Where is `$R^2$`?
///
/// We don't calculate `$R^2$` because "it is an inadequate measure for the
/// goodness of the fit in nonlinear models" ([Spiess and Neumeyer 2010](https://doi.org/10.1186/1471-2210-10-6)).
/// See also
/// [here](https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/),
/// [here](https://blog.minitab.com/en/adventures-in-statistics-2/why-is-there-no-r-squared-for-nonlinear-regression),
/// and [here](https://statisticsbyjim.com/regression/standard-error-regression-vs-r-squared/),
/// where the recommendation is to use the standard error of the regression instead.
/// See also the next section.
///
/// # Model Selection
///
/// If you want to want to use goodness of fit metrics to decide which of two
/// models to use, please look into the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion),
/// or the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion).
/// Both of these can be calculated from standard error of the regression. Note
/// that this fit uses least squares minimization so it assumes the errors are
/// zero mean and normally distributed.
#[derive(Debug, Clone)]
pub struct FitStatistics<Model>
where
    Model: SeparableNonlinearModel,
    DefaultAllocator: Allocator<Model::ScalarType, Dyn, Dyn>,
    DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
{
    /// The covariance matrix of the parameter estimates. The linear
    /// parameters are ordered first, followed by the non-linear parameters
    /// as if we had one big parameter vector `$(\vec{c}, \vec{\alpha})^T$`.
    /// # Correlation
    /// Note that we can calculate the estimated correlation matrix from
    /// the covariance matrix by dividing each element `$c_{ij}$` by
    /// `$\sqrt{c_{ii} c_{jj}}$`.
    /// # References
    /// See [O'Leary and Rust 2012](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)
    /// for reference.
    covariance_matrix: OMatrix<Model::ScalarType, Dyn, Dyn>,

    /// the weighted residuals `$\vec{r_w}$ = W * (\vec{y} - \vec{f}(vec{\alpha},\vec{c}))$`,
    /// where `$\vec{y}$` is the data, `$\vec{f}$` is the model function and `$W$` is the
    /// weights
    weighted_residuals: OVector<Model::ScalarType, Dyn>,

    /// the _weighted residual mean square_ or _regression standard error_.
    sigma: Model::ScalarType,

    /// the number of linear coefficients
    linear_coefficient_count: usize,

    /// the number of degrees of freedom
    degrees_of_freedom: usize,

    /// the number of nonlinear parameters
    nonlinear_parameter_count: usize,

    /// a helper variable that stores the calculated unscaled
    /// standard deviation that we can use in calculation for the
    /// confidence band.
    /// calculated according to eq (97) in
    /// https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
    /// with the difference that the square root has already been
    /// applied
    unscaled_confidence_sigma: OVector<Model::ScalarType, Dyn>,
}

impl<Model> FitStatistics<Model>
where
    Model: SeparableNonlinearModel,
    DefaultAllocator: Allocator<Model::ScalarType, Dyn, Dyn>,
    DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
{
    /// The covariance matrix of the parameter estimates. Here the parameters
    /// are both the linear as well as the nonlinear parameters of the model.
    /// The linear parameters are ordered first, followed by the non-linear parameters
    /// as if we had one combined parameter vector `$(\vec{c}, \vec{\alpha})$`.
    /// See [O'Leary and Rust 2012](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)
    /// for reference.
    ///
    /// # Example
    ///
    /// Say our model has two linear coefficients `$\vec{c}=(c_1,c_2)^T$` and three nonlinear parameters
    /// `$\vec{\alpha}=(\alpha_1,\alpha_2,\alpha_3)^T$`. Then the covariance matrix
    /// is odered for the parameter vector `$(c_1,c_2,\alpha_1,\alpha_2,\alpha_3)^T$`.
    /// The covariance matrix `$C$` (upper case C) is a square matrix of size `$5 \times 5$`.
    /// Matrix element `$C_{ij}$` is the covariance between the parameters at indices `$i$` and `$j$`, so in this
    /// example:
    /// * `$C_{11}$` is the variance of `$c_1$`,
    /// * `$C_{12}$` is the covariance between `$c_1$`and `$c_2$`,
    /// * `$C_{13}$` is the covariance between `$c_1$` and `$\alpha_1$`,
    /// * and so on.
    pub fn covariance_matrix(&self) -> &OMatrix<Model::ScalarType, Dyn, Dyn> {
        &self.covariance_matrix
    }

    #[deprecated(note = "Use the method calc_correlation_matrix.", since = "0.9.0")]
    /// calculate the correlation matrix. **Deprecated**, use the `calc_correlation_matrix``
    /// function instead.
    pub fn correlation_matrix(&self) -> OMatrix<Model::ScalarType, Dyn, Dyn>
    where
        Model::ScalarType: Float,
    {
        self.calc_correlation_matrix().clone()
    }

    /// The correlation matrix, ordered the same way as the covariance matrix.
    ///
    /// **Note** The correlation matrix is calculated on the fly from the
    /// covariance matrix when this function is called. It is suggested to
    /// store this matrix somewhere to avoid having to recalculate it.
    pub fn calc_correlation_matrix(&self) -> OMatrix<Model::ScalarType, Dyn, Dyn>
    where
        Model::ScalarType: Float,
    {
        calc_correlation_matrix(&self.covariance_matrix)
    }

    /// the weighted residuals
    /// ```math
    /// \vec{r_w} = W * (\vec{y} - \vec{f}(\vec{\alpha},\vec{c}))
    /// ```
    /// at the best fit parameters `$\vec{alpha}$` and `$\vec{c}$`.
    ///
    /// In case of a dataset with multiple members, the residuals are
    /// written one after the other into the vector
    pub fn weighted_residuals(&self) -> VectorView<Model::ScalarType, Dyn> {
        self.weighted_residuals.as_view()
    }

    /// the _regression standard error_ (also called _weighted residual mean square_, or _sigma_).
    /// Calculated as
    /// ```math
    /// \sigma = \frac{||\vec{r_w}||}{\sqrt{N_{data}-N_{params}-N_{basis}}},
    /// ```
    /// where `$N_{data}$` is the number of data points (observations), `$N_{params}$` is the number of nonlinear
    /// parameters, and `$N_{basis}$` is the number of linear parameters (i.e. the number
    /// of basis functions).
    pub fn regression_standard_error(&self) -> Model::ScalarType {
        self.sigma.clone()
    }

    /// helper function to extract the estimated _variance_
    /// of the nonlinear model parameters. Those could also be
    /// manually extracted from the diagonal of the covariance matrix.
    pub fn nonlinear_parameters_variance(&self) -> OVector<Model::ScalarType, Dyn>
    where
        Model::ScalarType: Scalar + Zero,
        DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
    {
        let total_parameter_count = self.linear_coefficient_count + self.nonlinear_parameter_count;
        let diagonal = self.covariance_matrix.diagonal();
        extract_range(
            &diagonal,
            Dyn(self.linear_coefficient_count),
            Dyn(total_parameter_count),
        )
    }

    /// helper function to extract the estimated _variance_
    /// of the linear model parameters.
    pub fn linear_coefficients_variance(&self) -> OVector<Model::ScalarType, Dyn>
    where
        Model::ScalarType: Scalar + Zero,
        DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
    {
        let diagonal = self.covariance_matrix.diagonal();
        extract_range(&diagonal, U0, Dyn(self.linear_coefficient_count))
    }

    //@todo comment:
    // panics: if probability is not in the open interval $$(0,1)$$
    pub fn confidence_radius(
        &self,
        probability: Model::ScalarType,
    ) -> OVector<Model::ScalarType, Dyn>
    where
        Model::ScalarType: num_traits::Float + One + Zero + CastF64,
    {
        assert!(
            probability.is_finite()
                && probability > Model::ScalarType::ZERO
                && probability < Model::ScalarType::ONE,
            "probability must be in open interval (0.,1.)"
        );

        let t_scale = distrs::StudentsT::ppf(
            probability.into_f64(),
            f64::from_usize(self.degrees_of_freedom).expect("failed int to float conversion"),
        );

        // this is a bit cumbersome due to the method being generic
        // and t_scale only being available as f64. We are just
        // multiplying t_scale to the unscaled confidence sigma element wise
        let mut confidence_radius =
            OVector::<Model::ScalarType, Dyn>::zeros(self.unscaled_confidence_sigma.nrows());

        confidence_radius
            .iter_mut()
            .zip(self.unscaled_confidence_sigma.iter())
            .for_each(|(cb, sigma)| {
                *cb = CastF64::from_f64(t_scale * sigma.into_f64());
            });

        confidence_radius
    }
}

/// extrant the half  open range `[Start,End)` from the given vector.
fn extract_range<ScalarType, D, Start, End>(
    vector: &OVector<ScalarType, D>,
    start: Start,
    end: End,
) -> OVector<ScalarType, <End as DimSub<Start>>::Output>
where
    ScalarType: Scalar + Zero,
    D: Dim,
    Start: Dim,
    End: DimMin<Start>,
    End: DimSub<Start>,
    nalgebra::DefaultAllocator: Allocator<ScalarType, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<ScalarType, <End as nalgebra::DimSub<Start>>::Output>,
{
    assert!(end.value() >= start.value());
    assert!(end.value() <= vector.nrows());
    //@todo this could be more efficient
    let mut range = OVector::<ScalarType, <End as DimSub<Start>>::Output>::zeros_generic(
        <End as DimSub<Start>>::Output::from_usize(end.value() - start.value()),
        U1,
    );
    for (idx, val) in range.iter_mut().enumerate() {
        *val = vector[(idx + start.value(), 0)].clone();
    }
    range
}

impl<Model> FitStatistics<Model>
where
    Model: SeparableNonlinearModel,
    DefaultAllocator: Allocator<Model::ScalarType, Dyn, Dyn>,
    DefaultAllocator: Allocator<<Model as SeparableNonlinearModel>::ScalarType, Dyn>,
{
    /// Calculate the fit statistics from the model, the WEIGHTED data, the weights, and the linear coefficients.
    /// Note the nonlinear coefficients are part of state of the model.
    /// The given parameters must be the ones after the the fit has completed.
    ///
    /// # Errors
    ///
    /// This gives an error if the fit is underdetermined or if
    /// the model returns an error when it is evaluated. Both of those
    /// are practically impossible after a successful fit, but better safe than sorry.
    #[allow(non_snake_case)]
    pub(crate) fn try_calculate(
        model: &Model,
        weighted_data: VectorView<Model::ScalarType, Dyn>,
        weights: &Weights<Model::ScalarType, Dyn>,
        linear_coefficients: VectorView<Model::ScalarType, Dyn>,
    ) -> Result<Self, Error<Model::Error>>
    where
        Model::ScalarType: Scalar + ComplexField + Float + Zero + FromPrimitive,
        Model: SeparableNonlinearModel,
        DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
        DefaultAllocator: Allocator<Model::ScalarType, Dyn, Dyn>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float,
    {
        // this is a sanity check and should never actually fail
        // it just makes sure I have been using this correctly inside of this crate
        // if this fails, this indicates a programming error
        debug_assert_eq!(
            weighted_data.ncols(),
            linear_coefficients.ncols(),
            "Data dims and linear coefficient dims don't match. Indicates logic error in library!"
        );
        debug_assert_eq!(weighted_data.nrows(),
            model.output_len(),
            "model output dimensions and data dimensions do not match. Indicates a programming error in this library!"
        );
        let output_len = model.output_len();
        //@todo(georgios) lift this restriction. As of now, all calculations assume
        // the problem has only a single RHS
        if linear_coefficients.ncols() > 1 {
            return Err(Error::MrhsUnsupported);
        }

        let J = model_function_jacobian(model, linear_coefficients)?;

        // see the OLeary and Rust Paper for reference
        // the names are taken from the paper
        let H = weights * J.clone();
        let weighted_residuals = weighted_data - weights * model.eval()? * linear_coefficients;
        let degrees_of_freedom = model.parameter_count() + model.base_function_count();
        if output_len <= degrees_of_freedom {
            return Err(Error::Underdetermined);
        }

        let sigma: Model::ScalarType = weighted_residuals.norm()
            / Float::sqrt(
                Model::ScalarType::from_usize(output_len - degrees_of_freedom).ok_or(
                    Error::IntegerToFloatConversion(output_len - degrees_of_freedom),
                )?,
            );

        let HTH_inv = (H.transpose() * H)
            .try_inverse()
            .ok_or(Error::MatrixInversion)?;
        let covariance_matrix = HTH_inv * sigma * sigma;

        // we don't calculate R^2, see the notes on the documentation
        // of this struct

        // what follows is the calculation for the confidence bands.
        // this is the correspoding github issue #29: https://github.com/geo-ant/varpro/issues/29
        // also see documentation for the python lmfit library here
        // https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult.eval_uncertainty
        // which references the formula here:
        // https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals

        //@todo(georgios) this logic assumes that unscaled_sigma
        // is a column vector. That means iter_mut will just iterate
        // over the elements.
        let mut unscaled_confidence_sigma = OVector::<Model::ScalarType, Dyn>::zeros(output_len);

        unscaled_confidence_sigma
            .iter_mut()
            // we have to iterate over the columns of J^T (transpose)
            // so we iterate over the rows of J.
            // @todo this could be made much more maybe by calculating
            // J in a suitable format from the beginning (or maybe we
            // can use a better dot product down below that does not
            // require us to transpose the vector like this
            .zip(J.row_iter())
            .for_each(|(sig, j)| {
                let j = j.transpose();
                *sig = j.dot(&(&covariance_matrix * &j));
                *sig = Float::sqrt(*sig);
            });

        Ok(Self {
            covariance_matrix,
            weighted_residuals,
            sigma,
            linear_coefficient_count: model.base_function_count(),
            degrees_of_freedom,
            nonlinear_parameter_count: model.parameter_count(),
            unscaled_confidence_sigma,
        })
    }
}

/// helper function to calculate a correlation matrix from a covariance matrix.
/// See the O'Leary and Rust paper for reference.
fn calc_correlation_matrix<ScalarType, D>(
    covariance_matrix: &OMatrix<ScalarType, D, D>,
) -> OMatrix<ScalarType, D, D>
where
    ScalarType: Float + Scalar + Zero,
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, D, D>,
{
    assert_eq!(
        covariance_matrix.nrows(),
        covariance_matrix.ncols(),
        "covariance matrix must be square"
    );
    let mut correlation_matrix = OMatrix::zeros_generic(
        D::from_usize(covariance_matrix.nrows()),
        D::from_usize(covariance_matrix.ncols()),
    );
    for i in 0..correlation_matrix.nrows() {
        let c_ii = covariance_matrix[(i, i)];
        for j in 0..correlation_matrix.ncols() {
            let c_jj = covariance_matrix[(j, j)];
            let sqrt_c_ii_c_jj = Float::sqrt(c_ii * c_jj);
            correlation_matrix[(i, j)] = covariance_matrix[(i, j)] / sqrt_c_ii_c_jj;
        }
    }
    correlation_matrix
}

// a helper function that calculates the jacobian of the
// model function `$\vec{f}(\vec{\alpha},\vec{c})$` evaluated at the parameters `$(\vec{c},\vec{\alpha})$`
// This is not the same as the jacobian of the
// fitting problem, which is the jacobian `$\vec{f}(\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
// where `$\vec{c}(\vec{\alpha})$` is the linear coefficients that solve the linear problem.
// see also the O'Leary matlab code.
#[allow(non_snake_case)]
fn model_function_jacobian<Model>(
    model: &Model,
    c: VectorView<Model::ScalarType, Dyn>,
) -> Result<OMatrix<Model::ScalarType, Dyn, Dyn>, Model::Error>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Float + Zero + Scalar + ComplexField,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Dyn>,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Dyn, Dyn>,
{
    // the part of the jacobian that contains the derivatives
    // with respect to the nonlinear parameters
    let mut jacobian_matrix_for_nonlinear_params =
        OMatrix::<Model::ScalarType, Dyn, Dyn>::zeros_generic(
            Dyn(model.output_len()),
            Dyn(model.parameter_count()),
        );
    for (idx, mut col) in jacobian_matrix_for_nonlinear_params
        .column_iter_mut()
        .enumerate()
    {
        //@todo this is not very efficient, make this better
        //but this does not happen repeatedly, so it might not be as bad
        col.copy_from(&(model.eval_partial_deriv(idx)? * c));
    }

    Ok(concat_colwise(
        model.eval()?,
        jacobian_matrix_for_nonlinear_params,
    ))
}

/// helper function to concatenate two matrices by pasting the
/// columns after each other: [left,right]. The matrices must have
/// the same number of rows.
fn concat_colwise<T, R, C1, C2, S1, S2>(
    left: Matrix<T, R, C1, S1>,
    right: Matrix<T, R, C2, S2>,
) -> OMatrix<T, R, <C1 as DimAdd<C2>>::Output>
where
    R: Dim,
    C1: Dim + DimAdd<C2>,
    C2: Dim,
    T: Scalar + Zero,
    nalgebra::DefaultAllocator: Allocator<T, R, <C1 as DimAdd<C2>>::Output>,
    nalgebra::DefaultAllocator: Allocator<T, R, C1>,
    nalgebra::DefaultAllocator: Allocator<T, R, C2>,
    S2: nalgebra::RawStorage<T, R, C2>,
    S1: nalgebra::RawStorage<T, R, C1>,
{
    assert_eq!(
        left.nrows(),
        right.nrows(),
        "left and right matrix must have the same number of rows"
    );
    let mut result = OMatrix::<T, R, <C1 as DimAdd<C2>>::Output>::zeros_generic(
        R::from_usize(left.nrows()),
        <C1 as DimAdd<C2>>::Output::from_usize(left.ncols() + right.ncols()),
    );

    for idx in 0..left.ncols() {
        result.column_mut(idx).copy_from(&left.column(idx));
    }

    for idx in 0..right.ncols() {
        result
            .column_mut(idx + left.ncols())
            .copy_from(&right.column(idx));
    }

    result
}
