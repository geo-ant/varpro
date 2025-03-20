use criterion::{criterion_group, criterion_main, Criterion};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::OVector;
use nalgebra::U1;
use pprof::criterion::{Output, PProfProfiler};
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::*;
use varpro::prelude::SeparableNonlinearModel;
use varpro::solvers::levmar::LevMarProblem;
use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;
use varpro::solvers::levmar::MultiRhs;
use varpro::solvers::levmar::Parallelism;
use varpro::solvers::levmar::Sequential;
use varpro::solvers::levmar::SingularValueDecomposition;

/// helper struct for the parameters of the double exponential
#[derive(Clone, PartialEq, Debug)]
struct DoubleExponentialParameters {
    tau1: f64,
    tau2: f64,
    coeffs: DMatrix<f64>,
}

fn build_problem_mrhs<Model>(
    true_parameters: DoubleExponentialParameters,
    mut model: Model,
) -> LevMarProblem<Model, MultiRhs, Sequential, SingularValueDecomposition<Model::ScalarType>>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    let DoubleExponentialParameters { tau1, tau2, coeffs } = true_parameters.clone();
    // save the initial guess so that we can reset the model to those
    let params = OVector::from_vec_generic(Dyn(model.parameter_count()), U1, vec![tau1, tau2]);
    let y = evaluate_complete_model_at_params_mrhs(&mut model, params, &coeffs);
    LevMarProblemBuilder::mrhs(model)
        .observations(y)
        .build()
        .expect("Building valid problem should not panic")
}

fn run_minimization_mrhs<Model, Par: Parallelism>(
    problem: LevMarProblem<Model, MultiRhs, Par, SingularValueDecomposition<Model::ScalarType>>,
) -> (DVector<f64>, DMatrix<f64>)
where
    Model: SeparableNonlinearModel<ScalarType = f64> + std::fmt::Debug,
    LevMarProblem<Model, MultiRhs, Par, SingularValueDecomposition<Model::ScalarType>>:
        LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
{
    let result = LevMarSolver::default()
        .fit(problem)
        .expect("fitting must exit successfully");
    let params = result.nonlinear_parameters();
    let coeff = result.linear_coefficients().unwrap();
    (params, coeff.into_owned())
}

fn bench_double_exp_no_noise_mrhs(c: &mut Criterion) {
    // see here on comparing functions
    // https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html
    let mut group = c.benchmark_group("Double Exponential Without Noise");
    // the support points for the model (moved into the closures separately)
    let x = linspace(0., 12.5, 1024);
    // initial guess for tau
    let tau_guess = (2., 6.5);

    let dataset_size = 1000;
    let linear_coeffs = create_random_dmatrix((3, dataset_size), 2314093240213841123, 0.0..100.0);

    let true_parameters = DoubleExponentialParameters {
        tau1: 1.,
        tau2: 3.,
        coeffs: linear_coeffs,
    };

    group.bench_function("Handcrafted Model (MRHS)", |bencher| {
        bencher.iter_batched(
            || {
                build_problem_mrhs(
                    true_parameters.clone(),
                    DoubleExpModelWithConstantOffsetSepModel::new(x.clone(), tau_guess),
                )
            },
            run_minimization_mrhs,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Using Model Builder (MRHS)", |bencher| {
        bencher.iter_batched(
            || {
                build_problem_mrhs(
                    true_parameters.clone(),
                    get_double_exponential_model_with_constant_offset(
                        x.clone(),
                        vec![tau_guess.0, tau_guess.1],
                    ),
                )
            },
            run_minimization_mrhs,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Handcrafted Model (MRHS) [multithreaded]", |bencher| {
        bencher.iter_batched(
            || {
                build_problem_mrhs(
                    true_parameters.clone(),
                    DoubleExpModelWithConstantOffsetSepModel::new(x.clone(), tau_guess),
                )
                .into_parallel()
            },
            run_minimization_mrhs,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Using Model Builder (MRHS) [multithreaded]", |bencher| {
        bencher.iter_batched(
            || {
                build_problem_mrhs(
                    true_parameters.clone(),
                    get_double_exponential_model_with_constant_offset(
                        x.clone(),
                        vec![tau_guess.0, tau_guess.1],
                    ),
                )
                .into_parallel()
            },
            run_minimization_mrhs,
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches_mrhs;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_double_exp_no_noise_mrhs);
criterion_main!(benches_mrhs);
