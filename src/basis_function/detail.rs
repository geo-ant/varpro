//! This module provides blanket implementations for the BasisFunction trait for all callables
//! `Fn(&DVector<T>,T)->DVector<T>`
//! `Fn(&DVector<T>,T,T)->DVector<T>`
//! `Fn(&DVector<T>,T,T,T)->DVector<T>`
//! ...
//! and so on up to a maximum number of parameters. T is of nalgebra::Scalar type.
//! Currently the maximum number of scalar parameters is 10.

use crate::basis_function::BasisFunction;
use nalgebra::{DVector, Scalar};

// this is just hand implemented so we can see the pattern from the first implementation.
// After that we'll use a macro to get rid of some of the boilerplate
impl<ScalarType, Func> BasisFunction<ScalarType, ScalarType> for Func
where
    ScalarType: Scalar,
    Func: Fn(&DVector<ScalarType>, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, x: &DVector<ScalarType>, params: &[ScalarType]) -> DVector<ScalarType> {
        if params.len() != Self::ARGUMENT_COUNT {
            panic!("Basisfunction expected {} arguments but the provided parameter slice has {} elements.",Self::ARGUMENT_COUNT,params.len());
        }
        (self)(x, params[0].clone())
    }

    const ARGUMENT_COUNT: usize = 1;
}

// a helper macro set for counting
// cribbed (and slightly modified) from
// https://danielkeep.github.io/tlborm/book/blk-counting.html
// and here
// https://stackoverflow.com/questions/43143327/how-to-allow-optional-trailing-commas-in-macros
// the macro can be called with or without trailing comma
// i.e. these example expressions evaluate to true:
// count_args(1,2,3)==count_args(1,2,3,) == 3usize, as well as
// count_args(a,b)==count_args(a,b,)==2usize
//TODO (Performance): maybe see also daniel keep post (see above): recursion is not the most efficient way to implement this!
// however, the alternative solutions can't be used in constant expressions, iirc.
macro_rules! count_args {
    () => {0usize};
    ($_head:tt, $($tail:tt),*) => {1usize + count_args!($($tail,)*)}; //these overloads are there to allow with and w/o trailing comma syntax
    ($_head:tt, $($tail:tt,)*) => {1usize + count_args!($($tail,)*)};
}

// a macro to implement the BasisFunction trait on functions taking a number of arguments
// without having to copy paste a lot of boilerplate.
// shamelessly cribbed and modified from:
// https://github.com/actix/actix-web/blob/web-v3.3.2/src/handler.rs
macro_rules! basefunction_impl_helper ({$ScalarType:ident, $(($n:tt, $T:ident)),+} => {
    impl<$ScalarType, Func> BasisFunction<$ScalarType,($($T,)+)> for Func
    where Func: Fn(&DVector<$ScalarType>,$($T,)+) -> DVector<$ScalarType>,
    $ScalarType : Scalar
    {
        fn eval(&self, x : &DVector<$ScalarType>,params : &[$ScalarType]) -> DVector<$ScalarType> {
            if params.len() != Self::ARGUMENT_COUNT {
                panic!("Basisfunction expected {} arguments but the provided parameter slice has {} elements.",Self::ARGUMENT_COUNT,params.len());
            }
            (&self)(
            x,
            $(params[$n].clone(),)+)
        }

        const ARGUMENT_COUNT :usize = count_args!($($T,)+);

    }
});

// this is the usage of the above macro to generate the required blanket implementations
// The trait bound on T is T: Scalar. It is important to continue the increasing sequence
// of numbers when extending the macro, because these numbers are used for indexing into
// the parameter slice
basefunction_impl_helper!(T, (0, T), (1, T));
basefunction_impl_helper!(T, (0, T), (1, T), (2, T));
basefunction_impl_helper!(T, (0, T), (1, T), (2, T), (3, T));
basefunction_impl_helper!(T, (0, T), (1, T), (2, T), (3, T), (4, T));
basefunction_impl_helper!(T, (0, T), (1, T), (2, T), (3, T), (4, T), (5, T));
basefunction_impl_helper!(T, (0, T), (1, T), (2, T), (3, T), (4, T), (5, T), (6, T));
basefunction_impl_helper!(
    T,
    (0, T),
    (1, T),
    (2, T),
    (3, T),
    (4, T),
    (5, T),
    (6, T),
    (7, T)
);
basefunction_impl_helper!(
    T,
    (0, T),
    (1, T),
    (2, T),
    (3, T),
    (4, T),
    (5, T),
    (6, T),
    (7, T),
    (8, T)
);
basefunction_impl_helper!(
    T,
    (0, T),
    (1, T),
    (2, T),
    (3, T),
    (4, T),
    (5, T),
    (6, T),
    (7, T),
    (8, T),
    (9, T)
); //10 parameter arguments
   //if more are implemented, add tests as well
