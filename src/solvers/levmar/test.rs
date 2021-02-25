use super::*;
use crate::test_helpers::get_double_exponential_model_with_constant_offset;
use levenberg_marquardt::{differentiate_numerically, LevenbergMarquardt};
use approx::assert_relative_eq;

// test that the jacobian of the least squares problem is correct
#[test]
fn jacobian_of_least_squares_prolem_is_correct() {
    let model = get_double_exponential_model_with_constant_offset();
    //octave: t = linspace(0,6,7);
    let tvec = DVector::from(vec![0., 1., 2., 3., 4., 5., 6.]);
    //octave y = 2*exp(-t/2)+exp(-t/4)+1
    let yvec = DVector::from(vec![
        4.0000, 2.9919, 2.3423, 1.9186, 1.6386, 1.4507, 1.3227
    ]);

    let mut problem = LevMarLeastSquaresProblemBuilder::new()
        .x(tvec)
        .y(yvec)
        .model(&model)
        .initial_guess(vec!{1.,7.})
        .build()
        .expect("Building a valid solver must not return an error.");

    // let jacobian_numerical = differentiate_numerically(&mut problem).expect("Numerical differentiation must succeed.");
    // let jacobian_calculated = problem.jacobian().expect("Jacobian must not be empty!");

    //assert_relative_eq!(jacobian_numerical,jacobian_calculated,epsilon=1e-12);

    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    println!("result : {:?}",result.parameters);
    println!("successful {}", report.termination.was_successful());

    todo!("get jacobian working....");
}
