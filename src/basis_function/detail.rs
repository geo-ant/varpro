//! This module provides blanket implementations for the BasisFunction trait for all callables
//! Fn(&DVector<T>,T)->DVector<T>
//! Fn(&DVector<T>,T,T)->DVector<T>
//! Fn(&DVector<T>,T,T,T)->DVector<T>
//! ...
//! Fn(&DVector<T>,T,T,T,T,T,T,T,T,T,T)->DVector<T>
//! where T : Scalar

use crate::basis_function::BasisFunction;
use nalgebra::{Scalar, DVector};

impl<ScalarType, Func> BasisFunction<ScalarType, ScalarType> for Func
    where
        ScalarType: Scalar ,
        Func: Fn(&DVector<ScalarType>, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: &[ScalarType]) -> DVector<ScalarType> {
        if params.len() != self.argument_count() {
            panic!("Basisfunction expected {} arguments but the provided parameter slice has {} elements.",self.argument_count(),params.len());
        }
        (&self)(location, params[0].clone())
    }

    fn argument_count(&self) -> usize {
        1
    }
}

// this is just hand implemented so we can see the pattern from the first to the second implementation.
// after that we'll use a macro to get rid of some of the boilerplate
impl<ScalarType, Func> BasisFunction<ScalarType, (ScalarType, ScalarType)> for Func
    where
        ScalarType: Scalar,
        Func: Fn(&DVector<ScalarType>, ScalarType, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: &[ScalarType]) -> DVector<ScalarType> {
        if params.len() != self.argument_count() {
            panic!("Basisfunction expected {} arguments but the provided parameter slice has {} elements.",self.argument_count(),params.len());
        }
        (&self)(location, params[0].clone(),params[1].clone())
    }

    fn argument_count(&self) -> usize {
        2
    }
}

// a helper macro set for counting
// cribbed (and sligthly modified) from
// https://danielkeep.github.io/tlborm/book/blk-counting.html
// and here
// https://stackoverflow.com/questions/43143327/how-to-allow-optional-trailing-commas-in-macros
// it must be called with a trailing comma and counts number of elements between commas,
// i.e. count_tts(1,2,3,) == 3usize, count_tts(a,b,)==2usize
//TODO: this is ugly because of the trailing comma requirements, but I could not figure
// out how to make that optional. Also dan keep (see above) has better syntax that requires
// less recursion, but I couldn't get it to work either. So try improving this counting thing
macro_rules! count_args {
    () => {0usize};
    ($_head:tt, $($tail:tt)*) => {1usize + count_args!($($tail)*)};
    ($($all:tt)* ,) => {1usize + count_args!($(all)*)};
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
            if params.len() != self.argument_count() {
                panic!("Basisfunction expected {} arguments but the provided parameter slice has {} elements.",self.argument_count(),params.len());
            }

            (&self)(
            x,
            $(params[$n].clone(),)+)
        }

        fn argument_count(&self) -> usize {
            const COUNT :usize = count_args!($($T,)+);
            COUNT
        }

    }
});

// this is the usage of the above macro to generate the required blanket implementations
// The trait bound on T is T: Scalar
basefunction_impl_helper!(T,(0,T),(1,T),(2,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T),(5,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T),(5,T),(6,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T),(5,T),(6,T),(7,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T),(5,T),(6,T),(7,T),(8,T));
basefunction_impl_helper!(T,(0,T),(1,T),(2,T),(3,T),(4,T),(5,T),(6,T),(7,T),(8,T),(9,T)); //10 parameter arguments