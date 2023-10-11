use crate::{prelude::SeparableNonlinearModel, util::Weights};
use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, DimAdd, Matrix, OMatrix, OVector,
    RealField, Scalar,
};
use num_traits::{Float, FromPrimitive, Zero};

#[cfg(test)]
mod test;

/// Information about an error that occurred during calculation
/// of the fit statistics.
pub enum Error<ModelError: std::error::Error> {
    /// Model returned error when it was evaluated
    ModelError(ModelError),
    /// Fit is underdetermined
    Underdetermined,
}

/// this structure contains some additional statistical information
/// about the fit, such as errors on the parameters and other useful
/// information to assess the quality of the fit.
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
    /// Calculate the fit statistics from the model, the data and the linear coefficients.
    /// The given parameters must be the ones after the the fit has completed.
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
        let H = weights * model_function_jacobian(model, linear_coefficients).unwrap();
        let output_len = model.output_len().value();
        let weighted_residuals = weights * (data - model.eval().unwrap() * linear_coefficients);
        let degrees_of_freedom =
            model.parameter_count().value() + model.base_function_count().value();
        if output_len <= degrees_of_freedom {
            return Err(Error::Underdetermined);
        }
        let sigma: Model::ScalarType = weighted_residuals.norm()
            / Float::sqrt(Model::ScalarType::from_usize(output_len - degrees_of_freedom).unwrap());

        let HTH_inv = (H.transpose() * H).try_inverse().unwrap();
        let covariance_matrix = HTH_inv * sigma * sigma;

        Ok(Self {
            covariance_matrix,
            weighted_residuals,
            sigma,
        })
    }
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
