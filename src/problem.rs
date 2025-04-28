use crate::prelude::*;
use crate::util::Weights;
use nalgebra::{
    ComplexField, DMatrix, DVector, DefaultAllocator, DimMin, MatrixView, SVD, Scalar, VectorView,
};
use nalgebra::{Dim, Dyn};

mod builder;

pub use builder::LevMarBuilderError;
pub use builder::SeparableProblemBuilder;

/// trait describing the type of right hand side for the problem, meaning either
/// a single right hand side or multiple right hand sides. The latter implies
/// global fitting.
pub trait RhsType {}

/// This type indicates that the associated problem has a single (vector) right hand side.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SingleRhs;

/// This type indicates that the associated problem has multiple right hand sides
/// and thus performs global fitting for the nonlinear parameters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MultiRhs;

impl RhsType for MultiRhs {}
impl RhsType for SingleRhs {}

/// This is a the problem of fitting the separable model to data in a form that the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate can use it to
/// perform the least squares fit.
///
/// # Construction
///
/// Use the [SeparableProblemBuilder](self::builder::SeparableProblemBuilder) to create an instance of a
/// levmar problem.
///
/// # Usage
///
/// After obtaining an instance of `SeparableProblem` we can pass it to the [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// structure of the levenberg_marquardt crate for minimization. Refer to the documentation of the
/// [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) for an overview. A usage example
/// is provided in this crate documentation as well. The [LevenbergMarquardt](levenberg_marquardt::LevenbergMarquardt)
/// solver is reexported by this module as [LevMarSolver](self::LevMarSolver) for naming consistency.
///
/// # `MRHS`: Multiple Right Hand Sides
///
/// The problem generic on the boolean `MRHS` which indicates whether the
/// problem fits a single (`MRHS == false`) or multiple (`MRHS == true`) right
/// hand sides. This is decided during the building process. The underlying
/// math does not change, but the interface changes to use vectors for coefficients
/// and data in case of a single right hand side. For multiple right hand sides,
/// the coefficients and the data are matrices corresponding to columns of
/// coefficient vectors and data vectors respectively.
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SeparableProblem<Model, Rhs: RhsType>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// the *weighted* data matrix to which to fit the model `$\boldsymbol{Y}_w$`.
    /// It is a matrix so it can accomodate multiple right hand sides. If
    /// the problem has only a single right hand side (MRHS = false), this is just
    /// a matrix with one column. The underlying math does not change in either case.
    /// **Attention** the data matrix is weighted with the weights if some weights
    /// where provided (otherwise it is unweighted)
    pub(crate) Y_w: DMatrix<Model::ScalarType>,
    /// a reference to the separable model we are trying to fit to the data
    pub(crate) model: Model,
    /// truncation epsilon for SVD below which all singular values are assumed zero
    pub(crate) svd_epsilon: <Model::ScalarType as ComplexField>::RealField,
    /// the weights of the data. If none are given, the data is not weighted
    /// If weights were provided, the builder has checked that the weights have the
    /// correct dimension for the data
    pub(crate) weights: Weights<Model::ScalarType, Dyn>,
    /// the currently cached calculations belonging to the currently set model parameters
    /// those are updated on set_params. If this is None, then it indicates some error that
    /// is propagated on to the levenberg-marquardt crate by also returning None results
    /// by residuals() and/or jacobian()
    pub(crate) cached: Option<CachedCalculations<Model::ScalarType, Dyn, Dyn>>,
    phantom: std::marker::PhantomData<Rhs>,
}

/// helper structure that stores the cached calculations,
/// which are carried out by the SeparableProblem on setting the parameters
#[derive(Debug, Clone)]
pub(crate) struct CachedCalculations<ScalarType, ModelDim, OutputDim>
where
    ScalarType: Scalar + ComplexField,
    ModelDim: Dim,
    OutputDim: Dim + nalgebra::DimMin<ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<OutputDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<<OutputDim as DimMin<ModelDim>>::Output, ModelDim>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<OutputDim, <OutputDim as DimMin<ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<<OutputDim as DimMin<ModelDim>>::Output>,
{
    /// The current residual matrix of model function values belonging to the current parameters
    pub(crate) current_residuals: DMatrix<ScalarType>,
    /// Singular value decomposition of the current function value matrix
    pub(crate) current_svd: SVD<ScalarType, OutputDim, ModelDim>,
    /// the linear coefficients `$\boldsymbol C$` providing the current best fit
    pub(crate) linear_coefficients: DMatrix<ScalarType>,
}

impl<Model, Rhs: RhsType> std::fmt::Debug for SeparableProblem<Model, Rhs>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeparableProblem")
            .field("y_w", &self.Y_w)
            .field("model", &"/* omitted */")
            .field("svd_epsilon", &self.svd_epsilon)
            .field("weights", &self.weights)
            .field("cached", &self.cached)
            .finish()
    }
}

impl<Model> SeparableProblem<Model, MultiRhs>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    ///
    /// # Returns
    ///
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    ///
    /// Since this method is for fitting a single right hand side, the coefficients
    /// are a single column vector.
    pub fn linear_coefficients(&self) -> Option<MatrixView<Model::ScalarType, Dyn, Dyn>> {
        self.cached
            .as_ref()
            .map(|cache| cache.linear_coefficients.as_view())
    }

    /// the weighted data matrix`$\boldsymbol{Y}_w$` to which to fit the model. Note
    /// that the weights are already applied to the data matrix and this
    /// is not the original data vector.
    ///
    /// This method is for fitting a multiple right hand sides, hence the data
    /// matrix is a matrix that contains the right hand sides as columns.
    pub fn weighted_data(&self) -> MatrixView<Model::ScalarType, Dyn, Dyn> {
        self.Y_w.as_view()
    }
}

impl<Model> SeparableProblem<Model, SingleRhs>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// Get the linear coefficients for the current problem. After a successful pass of the solver,
    /// this contains a value with the best fitting linear coefficients
    /// # Returns
    /// Either the current best estimate coefficients or None, if none were calculated or the solver
    /// encountered an error. After the solver finished, this is the least squares best estimate
    /// for the linear coefficients of the base functions.
    ///
    /// The linear coefficients are column vectors that are ordered
    /// into a matrix, where the column at index $$s$$ are the best linear
    /// coefficients for the member at index $$s$$ of the dataset.
    pub fn linear_coefficients(&self) -> Option<VectorView<Model::ScalarType, Dyn>> {
        self.cached
            .as_ref()
            .map(|cache|{
                debug_assert_eq!(cache.linear_coefficients.ncols(),1,
                    "coefficient matrix must have exactly one column for single right hand side. This indicates a programming error in the library.");
                cache.linear_coefficients.as_view()
            })
    }

    /// the weighted data vector `$\vec{y}_w$` to which to fit the model. Note
    /// that the weights are already applied to the data vector and this
    /// is not the original data vector.
    ///
    /// This method is for fitting a single right hand side, hence the data
    /// is a single column vector.
    pub fn weighted_data(&self) -> VectorView<Model::ScalarType, Dyn> {
        debug_assert_eq!(
            self.Y_w.ncols(),
            1,
            "data matrix must have exactly one column for single right hand side. This indicates a programming error in the library."
        );
        self.Y_w.as_view()
    }
}

impl<Model, Rhs: RhsType> SeparableProblem<Model, Rhs>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: Scalar + ComplexField + Copy,
{
    /// access the contained model immutably
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// get the weights of the data for the fitting problem
    pub fn weights(&self) -> &Weights<Model::ScalarType, Dyn> {
        &self.weights
    }
}
