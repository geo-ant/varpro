use shared_test_code::*;
use criterion::{criterion_group, criterion_main, Criterion};
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::ComplexField;
use nalgebra::DVector;
use nalgebra::RealField;
use nalgebra::Scalar;
use num_traits::Float;
use pprof::criterion::{Output, PProfProfiler};
use varpro::prelude::SeparableNonlinearModel;
use varpro::solvers::levmar::LevMarProblem;
use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

/// helper struct for the parameters of the double exponential
#[derive(Copy, Clone, PartialEq, Debug)]
struct DoubleExponentialParameters {
    tau1: f64,
    tau2: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

fn build_problem<Model:SeparableNonlinearModel<f64>>(
    true_parameters: DoubleExponentialParameters,
    (tau1_guess, tau2_guess): (f64, f64),
    model: &'_ Model,
) -> LevMarProblem<'_, f64,Model> {
    let DoubleExponentialParameters {
        tau1,
        tau2,
        c1,
        c2,
        c3,
    } = true_parameters;

    let x = linspace(0., 12.5, 1024);
    let y = evaluate_complete_model(model, &x, &[tau1, tau2], &DVector::from(vec![c1, c2, c3]));
    let problem = LevMarProblemBuilder::new(model)
        .x(x)
        .y(y)
        .initial_guess(&[tau1_guess, tau2_guess])
        .build()
        .expect("Building valid problem should not panic");
    problem
}

fn run_minimization<S,M>(problem: LevMarProblem<'_, S,M>) -> [S; 5]
where S: Scalar + Copy + ComplexField + Float + RealField, M:SeparableNonlinearModel<S> {
    let (problem, report) = LevMarSolver::new().minimize(problem);
    assert!(
        report.termination.was_successful(),
        "Termination not successful"
    );

    let params = problem.params();
    let coeff = problem.linear_coefficients().unwrap();
    [params[0], params[1], coeff[0], coeff[1], coeff[2]]
}

fn bench_double_exp_no_noise(c: &mut Criterion) {
    // let model = get_double_exponential_model_with_constant_offset();
    let model = DoubleExpModelWithConstantOffsetSepModel::default();
    let true_parameters = DoubleExponentialParameters {
        tau1: 1.,
        tau2: 3.,
        c1: 4.,
        c2: 2.5,
        c3: 1.,
    };

    c.bench_function("double exp w/o noise", move |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, (2., 6.5), &model),
            run_minimization,
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_double_exp_no_noise);
criterion_main!(benches);
