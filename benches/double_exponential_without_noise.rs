use criterion::{criterion_group, criterion_main, Criterion};
use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;
use nalgebra::DefaultAllocator;
use nalgebra::Dyn;
use nalgebra::OVector;
use nalgebra::RawStorageMut;
use nalgebra::Storage;
use nalgebra::U1;
use pprof::criterion::{Output, PProfProfiler};
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::models::DoubleExponentialDecayFittingWithOffsetLevmar;
use shared_test_code::*;
use varpro::prelude::SeparableNonlinearModel;
use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarProblemSvd;
use varpro::solvers::levmar::LevMarSolver;
use varpro::solvers::levmar::SingleRhs;

/// helper struct for the parameters of the double exponential
#[derive(Copy, Clone, PartialEq, Debug)]
struct DoubleExponentialParameters {
    tau1: f64,
    tau2: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

fn build_problem<Model>(
    true_parameters: DoubleExponentialParameters,
    mut model: Model,
) -> LevMarProblemSvd<Model, SingleRhs, false>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn>,
    DefaultAllocator: nalgebra::allocator::Allocator<Dyn, Dyn>,
    <DefaultAllocator as nalgebra::allocator::Allocator<Dyn>>::Buffer<f64>: Storage<f64, Dyn>,
    <DefaultAllocator as nalgebra::allocator::Allocator<Dyn>>::Buffer<f64>: RawStorageMut<f64, Dyn>,
{
    let DoubleExponentialParameters {
        tau1,
        tau2,
        c1,
        c2,
        c3,
    } = true_parameters;

    // save the initial guess so that we can reset the model to those
    let params = OVector::from_vec_generic(Dyn(model.parameter_count()), U1, vec![tau1, tau2]);

    let base_function_count = model.base_function_count();
    let y = evaluate_complete_model_at_params(
        &mut model,
        params,
        &OVector::from_vec_generic(Dyn(base_function_count), U1, vec![c1, c2, c3]),
    );
    LevMarProblemBuilder::new(model)
        .observations(y)
        .build()
        .expect("Building valid problem should not panic")
}

fn run_minimization<Model, const PAR: bool>(
    problem: LevMarProblemSvd<Model, SingleRhs, PAR>,
) -> (DVector<f64>, DVector<f64>)
where
    Model: SeparableNonlinearModel<ScalarType = f64> + std::fmt::Debug,
    LevMarProblemSvd<Model, SingleRhs, PAR>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
{
    let result = LevMarSolver::default()
        .fit(problem)
        .expect("fitting must exit successfully");
    let params = result.nonlinear_parameters();
    let coeff = result.linear_coefficients().unwrap();
    (params, coeff.into_owned())
}

/// solve the problem by using nonlinear least squares with levenberg marquardt
/// from the levenberg marquardt crate only
fn run_minimization_for_levenberg_marquardt_crate_problem(
    problem: DoubleExponentialDecayFittingWithOffsetLevmar,
) -> [f64; 5] {
    let (problem, report) = LevenbergMarquardt::new()
        // if I don't set this, the solver will not converge
        .with_stepbound(1.)
        .minimize(problem);
    assert!(
        report.termination.was_successful(),
        "Termination not successful"
    );
    let params = problem.params();
    [params[0], params[1], params[2], params[3], params[4]]
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
    let tau_guess = (2., 6.5);
    // function values
    let f = x.map(|x: f64| {
        true_parameters.c1 * (-x / true_parameters.tau1).exp()
            + true_parameters.c2 * (-x / true_parameters.tau2).exp()
            + true_parameters.c3
    });

    group.bench_function("Using Levenberg Marquardt Crate", |bencher| {
        bencher.iter_batched(
            || {
                DoubleExponentialDecayFittingWithOffsetLevmar::new(
                    // we make it easier for the solver by giving it the
                    // correct guesses for the coefficients from the start
                    // which is not gonna happen for realistic problems
                    &[
                        tau_guess.0,
                        tau_guess.1,
                        true_parameters.c1,
                        true_parameters.c2,
                        true_parameters.c3,
                    ],
                    &x,
                    &f,
                )
            },
            run_minimization_for_levenberg_marquardt_crate_problem,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Using Model Builder", |bencher| {
        bencher.iter_batched(
            || {
                build_problem(
                    true_parameters,
                    get_double_exponential_model_with_constant_offset(
                        x.clone(),
                        vec![tau_guess.0, tau_guess.1],
                    ),
                )
            },
            run_minimization,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Handcrafted Model", |bencher| {
        bencher.iter_batched(
            || {
                build_problem(
                    true_parameters,
                    DoubleExpModelWithConstantOffsetSepModel::new(x.clone(), tau_guess),
                )
            },
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
