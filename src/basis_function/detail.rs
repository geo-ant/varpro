// TODO: Automate this using a macro and implement for functions of more arguments!!

use crate::basis_function::BasisFunction;
use nalgebra::{Scalar, DVector};

impl<ScalarType, Func> BasisFunction<ScalarType, ScalarType> for Func
    where
        ScalarType: Scalar,
        Func: Fn(&DVector<ScalarType>, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType> {
        let params_length = params.len();
        let panic_out_of_bounds = || {
            panic!(
                "Basis function takes {} arguments, but the number of provided parameters was {}.",
                self.argument_count(),
                params_length
            )
        };
        let mut elems = params.into_iter();
        (&self)(location, elems.next().unwrap_or_else(panic_out_of_bounds))
    }

    fn argument_count(&self) -> usize {
        1
    }
}

impl<ScalarType, Func> BasisFunction<ScalarType, (ScalarType, ScalarType)> for Func
    where
        ScalarType: Scalar,
        Func: Fn(&DVector<ScalarType>, ScalarType, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType> {
        let params_length = params.len();
        let panic_out_of_bounds = || {
            panic!(
                "Basis function takes {} arguments, but the number of provided parameters was {}.",
                self.argument_count(),
                params_length
            )
        };
        let mut elems = params.into_iter();
        (&self)(
            location,
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
        )
    }

    fn argument_count(&self) -> usize {
        2
    }
}

impl<ScalarType, Func> BasisFunction<ScalarType, (ScalarType, ScalarType, ScalarType)> for Func
    where
        ScalarType: Scalar,
        Func: Fn(&DVector<ScalarType>, ScalarType, ScalarType, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType> {
        let params_length = params.len();
        let panic_out_of_bounds = || {
            panic!(
                "Basis function takes {} arguments, but the number of provided parameters was {}.",
                self.argument_count(),
                params_length
            )
        };
        let mut elems = params.into_iter();
        (&self)(
            location,
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
        )
    }

    fn argument_count(&self) -> usize {
        3
    }
}

impl<ScalarType, Func> BasisFunction<ScalarType, (ScalarType, ScalarType, ScalarType, ScalarType)>
for Func
    where
        ScalarType: Scalar,
        Func: Fn(&DVector<ScalarType>, ScalarType, ScalarType, ScalarType, ScalarType) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType> {
        let params_length = params.len();
        let panic_out_of_bounds = || {
            panic!(
                "Basis function takes {} arguments, but the number of provided parameters was {}.",
                self.argument_count(),
                params_length
            )
        };
        let mut elems = params.into_iter();
        (&self)(
            location,
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
        )
    }

    fn argument_count(&self) -> usize {
        4
    }
}

impl<ScalarType, Func>
BasisFunction<ScalarType, (ScalarType, ScalarType, ScalarType, ScalarType, ScalarType)> for Func
    where
        ScalarType: Scalar,
        Func: Fn(
            &DVector<ScalarType>,
            ScalarType,
            ScalarType,
            ScalarType,
            ScalarType,
            ScalarType,
        ) -> DVector<ScalarType>,
{
    fn eval(&self, location: &DVector<ScalarType>, params: Vec<ScalarType>) -> DVector<ScalarType> {
        let params_length = params.len();
        let panic_out_of_bounds = || {
            panic!(
                "Basis function takes {} arguments, but the number of provided parameters was {}.",
                self.argument_count(),
                params_length
            )
        };
        let mut elems = params.into_iter();
        (&self)(
            location,
            elems.next().unwrap_or_else(panic_out_of_bounds), // trick to use for repetition (elems.next().unwrap_or_else(panic_out_of_bounds),n).0
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
            elems.next().unwrap_or_else(panic_out_of_bounds),
        )
    }

    fn argument_count(&self) -> usize {
        5
    }
}