use crate::{prelude::SeparableNonlinearModel, util::Weights};
use nalgebra::{ComplexField, DefaultAllocator, Dim, DimAdd, Matrix, OMatrix, OVector, Scalar};
use num_traits::{Float, Zero};

#[cfg(test)]
mod test;

/// this structure contains information about the goodness of
/// a fit and some statistical properties like estimates uncerainties
/// of the parameters
#[derive(Debug, Clone)]
pub struct FitStatistics<ScalarType, ModelDim, ParameterDim>
where
    ModelDim: Dim + DimAdd<ParameterDim>,
    ParameterDim: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<
        ScalarType,
        <ModelDim as DimAdd<ParameterDim>>::Output,
        <ModelDim as DimAdd<ParameterDim>>::Output,
    >,
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
    pub covariance_matrix: OMatrix<
        ScalarType,
        <ModelDim as DimAdd<ParameterDim>>::Output,
        <ModelDim as DimAdd<ParameterDim>>::Output,
    >,
    // /// The parameter `$R^2$`, also known as the coefficient of determination,
    // /// or the square of the multiple correlation coefficient. A commonly
    // /// used (and misused) measure of the quality of a regression.
    // pub r_squared: ScalarType,
}

impl<ScalarType, ModelDim, ParameterDim> FitStatistics<ScalarType, ModelDim, ParameterDim>
where
    ModelDim: Dim + DimAdd<ParameterDim>,
    ParameterDim: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<
        ScalarType,
        <ModelDim as DimAdd<ParameterDim>>::Output,
        <ModelDim as DimAdd<ParameterDim>>::Output,
    >,
{
    pub(crate) fn try_calculate<Model>(
        model: &Model,
        weights: &Weights<ScalarType, Model::OutputDim>,
        linear_coefficients: OVector<ScalarType, ModelDim>,
    ) -> Option<Self>
    where
        ScalarType: Scalar + ComplexField,
        Model: SeparableNonlinearModel<
            ScalarType = ScalarType,
            ModelDim = ModelDim,
            ParameterDim = ParameterDim,
        >,
        DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, Model::OutputDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, ParameterDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, ModelDim>,
        DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, Model::OutputDim, ModelDim>,
    {
        // let hmat = weights
        //     * model_function_jacobian(model, linear_coefficients.unwrap()).unwrap();
        todo!()
    }
}

// a helper function that calculates the jacobian of the
// model function `$\vec{f}(\vec{\alpha},\vec{c})$` evaluated at the parameters `$(\vec{c},\vec{\alpha})$`
// This is not the same as the jacobian of the
// fitting problem, which is the jacobian `$\vec{f}(\vec{\alpha},\vec{c}(\vec{\alpha}))$`,
// where `$\vec{c}(\vec{\alpha})$` is the linear coefficients that solve the linear problem.
// see also the O'Leary matlab code.
fn model_function_jacobian<Model>(
    model: &Model,
    c: OVector<Model::ScalarType, Model::ModelDim>,
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
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::ParameterDim>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ModelDim>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<Model::ScalarType, Model::OutputDim, Model::ParameterDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        Model::ScalarType,
        Model::OutputDim,
        <Model::ModelDim as DimAdd<Model::ParameterDim>>::Output,
    >,
    <Model as SeparableNonlinearModel>::ModelDim:
        nalgebra::DimAdd<<Model as SeparableNonlinearModel>::ParameterDim>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model as SeparableNonlinearModel>::ScalarType,
        <Model as SeparableNonlinearModel>::ModelDim,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
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
        col.copy_from(&(model.eval_partial_deriv(idx)? * &c));
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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, R, <C1 as DimAdd<C2>>::Output>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, R, C1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, R, C2>,
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
