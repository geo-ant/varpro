use criterion::{criterion_group, criterion_main, Criterion};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::ComplexField;
use nalgebra::DVector;
use nalgebra::RealField;
use nalgebra::Scalar;
use num_traits::Float;
use pprof::criterion::{Output, PProfProfiler};
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::*;
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

fn build_problem<Model: SeparableNonlinearModel<f64>>(
    true_parameters: DoubleExponentialParameters,
    (tau1_guess, tau2_guess): (f64, f64),
    model: Model,
) -> LevMarProblem<f64, Model> {
    let DoubleExponentialParameters {
        tau1,
        tau2,
        c1,
        c2,
        c3,
    } = true_parameters;

    let y = evaluate_complete_model(&model,  &DVector::from(vec![c1, c2, c3]));
    let problem = LevMarProblemBuilder::new(model)
        .y(y)
        .build()
        .expect("Building valid problem should not panic");
    problem
}

fn run_minimization<S, M>(problem: LevMarProblem<S, M>) -> [S; 5]
where
    S: Scalar + Copy + ComplexField + Float + RealField,
    M: SeparableNonlinearModel<S>,
{
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
    let true_parameters = DoubleExponentialParameters {
        tau1: 1.,
        tau2: 3.,
        c1: 4.,
        c2: 2.5,
        c3: 1.,
    };

    // see here on comparing functions
    // https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html
    let mut group = c.benchmark_group("Double Exponential Without Noise");
    // the support points for the model (moved into the closures separately)
    let x = linspace(0., 12.5, 1024);
    // initial guess for tau 
    let tau_guess  = (2., 6.5);

    group.bench_function("Using Model Builder", |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, tau_guess, get_double_exponential_model_with_constant_offset(x.clone())),
            run_minimization,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Handcrafted Model", |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, tau_guess, DoubleExpModelWithConstantOffsetSepModel::new(x.clone(),tau_guess)),
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
