#[cfg(test)]
mod test;

pub mod detail;

use nalgebra::{DVector, Scalar};
use crate::model::errors::ModelBuildError;

/// This trait allows us to pass vector valued functions `$f(\vec{x},\alpha_1,...\alpha_N$)` in a generic fashion.
/// The functions must have at least one parameter argument besides the location, i.e. a function
/// `$f(\vec{x})$` does not satisfy that trait, but e.g. `$f(\vec{x},\alpha_1)$` does. The function
/// must be vector valued, i.e. produce a vector. The basis function must map the location vector `$\vec{x}$`
/// to a result vector of *the same size and scalar type*. Unfortunately, requirement on the length of the output
/// vector cannot be enforced by the type system. If it is violated, then calculations using the basis
/// function will fail in the [SeparableModel].
///
/// Since Rust
/// does not have variadic generics, this trait is implemented for all functions up to
/// a maximum number of arguments. This maximum number of arguments can be found out by checking
/// the blanket implementations.
/// # Generic Parameters
/// ## Scalar Type
/// The functions must have an interface `Fn(&DVector<ScalarType>,ScalarType)-> DVector<ScalarType>`,
/// `Fn(&DVector<ScalarType>,ScalarType,ScalarType)-> DVector<ScalarType>` and so on.
/// All numeric types, including the return value must be of the same scalar type.
/// ## ArgList : The argument list
/// This type is of no consequence for the user because it will be correctly inferred when
/// passing a function. It is a [nifty trick](https://geo-ant.github.io/blog/2021/rust-traits-and-variadic-functions/)
/// that allows us to implement this trait for functions taking different arguments. Just FYI: The
/// type reflects the list of parameters `\alpha_j$`, so that for a function `Fn(&DVector<ScalarType>,ScalarType) -> DVector<ScalarType>`
/// it follows that `ArgList=ScalarType`, while for `Fn(&DVector<ScalarType>,ScalarType)-> DVector<ScalarType>` it is `ArgList=(ScalarType,ScalarType)`.
pub trait BasisFunction<ScalarType, ArgList>
where
    ScalarType: Scalar + Clone,
{
    /// A common calling interface to evaluate this function by passing a vector of scalar types
    /// that is dispatched to the arguments in order. The dispatches vector must have the same
    /// length as the parameter argument list.
    /// # Effect and Panics
    /// If the vector has fewer elements than the parameter argument list. If the vector has more element,
    /// only the first `$N$` elements are used and dispatched to the parameters in order, i.e.
    /// `$\alpha_1$`=`param[0]`, ..., `$\alpha_N$`=`param[N-1]`. Calling eval will result in evaluating
    /// the implementing callable at the location and arguments given.
    ///
    /// **Attention** The library takes care that no panics can be caused by calls to `eval` from
    /// within this library by making sure the parameter vector has the correct length. Therefore
    /// it is not recommended (and not necessary) to use this function in other code than inside this library.
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType>;

    /// This gives the number of parameter arguments to the callable. So for a function `$f(\vec{x},\alpha_1,...\alpha_N)$`
    /// this will give `N`.
    fn argument_count(&self) -> usize;
}


