use nalgebra::{ComplexField, DefaultAllocator, Dim, DimAdd, OMatrix, OVector, Scalar};

use crate::{prelude::SeparableNonlinearModel, util::weights::Weights};

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
        todo!()
    }
}
