use super::*;

/// a function taking a location vector and 1 parameter argument
fn callable1(x: &DVector<f64>, alpha: f64) -> DVector<f64> {
    alpha * x.clone()
}

/// a function taking a location vector and 2 parameter arguments
fn callable2(x: &DVector<f64>, alpha1: f64, alpha2: f64) -> DVector<f64> {
    alpha1 * alpha2 * x.clone()
}

/// a helper function to evaluate callables of different arguments lists. Also doubles
/// as a syntax check that the trait I implemented really can be used to pass functions with
/// variadic arguments using this common interface.
fn eval_callable<ScalarType: Scalar, ArgList, BasisFuncType: BasisFunction<ScalarType, ArgList>>(
    f: BasisFuncType,
    x : &DVector<ScalarType>,
    params: Vec<ScalarType>,
) -> DVector<ScalarType> {
    f.eval(x,params)
}


#[test]
fn callable_evaluation_using_the_generic_interface() {
    let x = DVector::from_element(10, 1.);
    assert_eq!(eval_callable(callable1,&x,vec!{2.}),2.*x.clone());
    assert_eq!(eval_callable(callable2,&x,vec!{2.,3.}),2.*3.*x.clone());
    assert_eq!(eval_callable(|x:&DVector<f64>,a,b,c|{a*b*c*x.clone()},&x,vec!{2.,3.,4.}),2.*3.*4.*x.clone());
    assert_eq!(eval_callable(|x:&DVector<f64>,a,b,c,d|{a*b*c*d*x.clone()},&x,vec!{2.,3.,4.,5.}),2.*3.*4.*5.*x.clone());
    assert_eq!(eval_callable(|x:&DVector<f64>,a,b,c,d,e|{a*b*c*d*e*x.clone()},&x,vec!{2.,3.,4.,5.,6.}),2.*3.*4.*5.*6.*x.clone());
    //todo: implement other traits here as well
}

#[test]
fn callable_argument_length_is_correct() {
    let x = DVector::from_element(10, 1.);
    assert_eq!(callable1.argument_count(),1);
    assert_eq!(callable2.argument_count(),2);
    assert_eq!((|x:&DVector<f64>,a:f64,b:f64,c:f64|{x.clone()}).argument_count(),3);
    assert_eq!((|x:&DVector<f64>,a:f64,b:f64,c:f64,d:f64|{x.clone()}).argument_count(),4);
    assert_eq!((|x:&DVector<f32>,a:f32,b:f32,c:f32,d:f32,e:f32|{x.clone()}).argument_count(),5);
    //todo: implement other traits here as well
}

#[test]
#[should_panic]
fn callable_evaluation_panics_if_too_few_elements_in_vector() {
    let x = DVector::from_element(10, 1.);
    let _ = eval_callable(callable2,&x,vec!{1.});
}
