use crate::{prelude::SeparableNonlinearModel, util::Weights};
use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, DimAdd, Matrix, OMatrix, OVector,
    RealField, Scalar,
};
use num_traits::{Float, FromPrimitive, Zero};
use thiserror::Error as ThisError;

#[cfg(any(test, doctest))]
mod test;

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
    FloatToIntError(usize),
    /// Failed to calculate the inverse of a matrix
    #[error("Matrix inversion error")]
    MatrixInversionError,
}

/// This structure contains some additional statistical information
/// about the fit, such as errors on the parameters and other useful
/// information to assess the quality of the fit.
///
/// # Where is R squared?
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
    Model::ModelDim: DimAdd<Model::ParameterDim>,
    Model::ParameterDim: Dim,
    DefaultAllocator: Allocator<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
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
    #[allow(clippy::type_complexity)]
    covariance_matrix: OMatrix<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,

    /// the correlation matrix, ordered the same way as the covariance matrix.
    #[allow(clippy::type_complexity)]
    correlation_matrix: OMatrix<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,

    /// the weighted residuals `$\vec{r_w}$ = W * (\vec{y} - \vec{f}(vec{\alpha},\vec{c}))$`,
    /// where `$\vec{y}$` is the data, `$\vec{f}$` is the model function and `$W$` is the
    /// weights
    weighted_residuals: OVector<Model::ScalarType, Model::OutputDim>,

    /// the _weighted residual mean square_ or _regression standard error_.
    sigma: Model::ScalarType,
    // /// The parameter `$R^2$`, also known as the coefficient of determination,
    // /// or the square of the multiple correlation coefficient. A commonly
    // /// used (and misused) measure of the quality of a regression.
    // pub r_squared: ScalarType,
}

impl<Model> FitStatistics<Model>
where
    Model: SeparableNonlinearModel,
    Model::ModelDim: DimAdd<Model::ParameterDim>,
    Model::ParameterDim: Dim,
    DefaultAllocator: Allocator<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
{
    /// The covariance matrix of the parameter estimates. Here the parameters
    /// are both the linear as well as the nonlinear parameters of the model.
    /// parameters are ordered first, followed by the non-linear parameters
    /// as if we had one big parameter vector `$(\vec{c}, \vec{\alpha})^T$`.
    /// See [O'Leary and Rust 2012](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)
    /// for reference.
    ///
    /// # Example
    ///
    /// Say our model has two linear coefficients `$\vec{c}=(c_1,c_2)^T$` and three nonlinear parameters
    /// `$\vec{\alpha}=(\alpha_1,\alpha_2,\alpha_3)^T$`. Then the covariance matrix
    /// is odered for the parameter vector `$(c_1,c_2,\alpha_1,\alpha_2,\alpha_3)^T$`.
    /// The covariance matrix `$C$` (upper case C) is a square matrix of size `$5 \times 5$`.
    /// Element `$C_{ij}$` is the covariance between the parameters `$i$` and `$j$`, so in this
    /// example `$C_{11}$` is the variance of `$c_1$`, `$C_{12}$` is the covariance between `$c_1$`
    /// and `$c_2$`, `$C_{13}$` is the covariance between `$c_1$` and `$\alpha_1$`, and so on.
    #[allow(clippy::type_complexity)]
    pub fn covariance_matrix(
        &self,
    ) -> &OMatrix<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    > {
        &self.covariance_matrix
    }

    /// the correlation matrix, ordered the same way as the covariance matrix.
    #[allow(clippy::type_complexity)]
    pub fn correlation_matrix(
        &self,
    ) -> &OMatrix<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    > {
        &self.correlation_matrix
    }

    /// the weighted residuals
    /// ```math
    /// \vec{r_w}$ = W * (\vec{y} - \vec{f}(vec{\alpha},\vec{c}))
    /// ```
    /// at the best fit parameters.
    pub fn weighted_residuals(&self) -> &OVector<Model::ScalarType, Model::OutputDim> {
        &self.weighted_residuals
    }

    /// the _regression standard error_ (also called _weighted residual mean square_, or _sigma_).
    /// Calculated as
    /// ```math
    /// \sigma = \frac{||\vec{r_w}||}{\sqrt{N_{data}}-N_{params}-N_{basis}},
    /// ```
    /// where `$N_{data}$` is the number of data points, `$N_{params}$` is the number of nonlinear
    /// parameters, and `$N_{basis}$` is the number of linear parameters (i.e. the number
    /// of basis functions).
    pub fn regression_standard_error(&self) -> Model::ScalarType {
        self.sigma.clone()
    }
}

impl<Model> FitStatistics<Model>
where
    Model: SeparableNonlinearModel,
    Model::ModelDim: DimAdd<Model::ParameterDim>,
    DefaultAllocator: Allocator<
        Model::ScalarType,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    nalgebra::DefaultAllocator: Allocator<
        <Model as SeparableNonlinearModel>::ScalarType,
        <Model as SeparableNonlinearModel>::OutputDim,
        <Model as SeparableNonlinearModel>::ModelDim,
    >,
    nalgebra::DefaultAllocator: Allocator<
        <Model as SeparableNonlinearModel>::ScalarType,
        <Model as SeparableNonlinearModel>::ParameterDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim>,
{
    /// Calculate the fit statistics from the model, the data, the weights, and the linear coefficients.
    /// Note the nonlinear coefficients are part of state of the model.
    /// The given parameters must be the ones after the the fit has completed.
    ///
    /// # Errors
    /// This gives an error if the fit is underdetermined or if
    /// the model returns an error when it is evaluated. Both of those
    /// are practically impossible after a successful fit, but better safe than sorry.
    #[allow(non_snake_case)]
    pub(crate) fn try_calculate(
        model: &Model,
        data: &OVector<Model::ScalarType, Model::OutputDim>,
        weights: &Weights<Model::ScalarType, Model::OutputDim>,
        linear_coefficients: &OVector<Model::ScalarType, Model::ModelDim>,
    ) -> Result<Self, Error<Model::Error>>
    where
        Model::ScalarType: Scalar + ComplexField + Float + Zero + FromPrimitive,
        Model: SeparableNonlinearModel,
        DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::ModelDim>,
        DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
        DefaultAllocator: Allocator<
            Model::ScalarType,
            Model::OutputDim,
            <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
        >,
        DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim>,
        Model::ScalarType: Scalar + ComplexField + Copy + RealField + Float,
        DefaultAllocator: Allocator<
            Model::ScalarType,
            <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
            <Model as SeparableNonlinearModel>::OutputDim,
        >,
    {
        // see the OLeary and Rust Paper for reference
        // the names are taken from the paper

        let H = weights * model_function_jacobian(model, linear_coefficients)?;
        let output_len = model.output_len().value();
        let weighted_residuals = weights * (data - model.eval()? * linear_coefficients);
        let degrees_of_freedom =
            model.parameter_count().value() + model.base_function_count().value();
        if output_len <= degrees_of_freedom {
            return Err(Error::Underdetermined);
        }

        let sigma: Model::ScalarType = weighted_residuals.norm()
            / Float::sqrt(
                Model::ScalarType::from_usize(output_len - degrees_of_freedom)
                    .ok_or(Error::FloatToIntError(output_len - degrees_of_freedom))?,
            );

        let HTH_inv = (H.transpose() * H)
            .try_inverse()
            .ok_or(Error::MatrixInversionError)?;
        let covariance_matrix = HTH_inv * sigma * sigma;
        let correlation_matrix = calc_correlation_matrix(&covariance_matrix);

        // we don't calculate R^2, see the notes on the documentation
        // of this struct

        Ok(Self {
            covariance_matrix,
            correlation_matrix,
            weighted_residuals,
            sigma,
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
#[allow(clippy::type_complexity)]
fn model_function_jacobian<Model>(
    model: &Model,
    c: &OVector<Model::ScalarType, Model::ModelDim>,
) -> Result<
    OMatrix<
        Model::ScalarType,
        Model::OutputDim,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    Model::Error,
>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Float + Zero + Scalar + ComplexField,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::ParameterDim>,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    nalgebra::DefaultAllocator: Allocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim>,
    nalgebra::DefaultAllocator: Allocator<
        Model::ScalarType,
        Model::OutputDim,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    <Model as SeparableNonlinearModel>::ModelDim:
        nalgebra::DimAdd<<Model as SeparableNonlinearModel>::ParameterDim>,
    nalgebra::DefaultAllocator: Allocator<
        <Model as SeparableNonlinearModel>::ScalarType,
        <Model as SeparableNonlinearModel>::ModelDim,
    >,
    nalgebra::DefaultAllocator: Allocator<
        <Model as SeparableNonlinearModel>::ScalarType,
        <Model as SeparableNonlinearModel>::OutputDim,
    >,
{
    // the part of the jacobian that contains the derivatives
    // with respect to the nonlinear parameters
    let mut jacobian_matrix_for_nonlinear_params =
        OMatrix::<Model::ScalarType, Model::OutputDim, Model::ParameterDim>::zeros_generic(
            model.output_len(),
            model.parameter_count(),
        );
    for (idx, mut col) in jacobian_matrix_for_nonlinear_params
        .column_iter_mut()
        .enumerate()
    {
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
