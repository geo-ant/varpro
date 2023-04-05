use criterion::{criterion_group, criterion_main, Criterion};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::ComplexField;

use nalgebra::Const;
use nalgebra::DefaultAllocator;

use nalgebra::DimMin;
use nalgebra::DimSub;

use nalgebra::OVector;
use nalgebra::RawStorageMut;


use nalgebra::Storage;
use nalgebra::U1;

use pprof::criterion::{Output, PProfProfiler};
use shared_test_code::models::DoubleExpModelWithConstantOffsetSepModel;
use shared_test_code::*;
use shared_test_code::models::DoubleExponentialDecayFittingWithOffsetLevmar;
use varpro::model::SeparableModel;
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

fn build_problem<Model>(
    true_parameters: DoubleExponentialParameters,
    mut model: Model,
) -> LevMarProblem<Model> 
where
    Model: SeparableNonlinearModel<ScalarType=f64>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::ParameterDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::ParameterDim,Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::OutputDim, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::OutputDim>,
    <DefaultAllocator as nalgebra::allocator::Allocator<f64, Model::OutputDim>>::Buffer: Storage<f64, Model::OutputDim>,
    <DefaultAllocator as nalgebra::allocator::Allocator<f64, Model::OutputDim>>::Buffer: RawStorageMut<f64, Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::OutputDim, Model::ParameterDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<(usize, usize), <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::OutputDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<<f64 as ComplexField> ::RealField, <<Model::OutputDim as DimMin<Model::ModelDim>>::Output as DimSub<Const<1>>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, <<Model::OutputDim as DimMin<Model::ModelDim>>::Output as DimSub<Const<1>>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<(<f64 as ComplexField>::RealField, usize), <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    <Model::OutputDim as DimMin<Model::ModelDim>>::Output: DimSub<nalgebra::dimension::Const<1>> ,
    Model::OutputDim: DimMin<Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, <Model::OutputDim as DimMin<Model::ModelDim>>::Output, Model::ModelDim>,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Model::OutputDim, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
    DefaultAllocator: nalgebra::allocator::Allocator<<f64 as ComplexField>::RealField, <Model::OutputDim as DimMin<Model::ModelDim>>::Output>,
{
    let DoubleExponentialParameters {
        tau1,
        tau2,
        c1,
        c2,
        c3,
    } = true_parameters;
    
    // save the initial guess so that we can reset the model to those
    let params = OVector::from_vec_generic(model.parameter_count(), U1, vec![tau1,tau2]);
    
    let base_function_count = model.base_function_count();
    let y = evaluate_complete_model_at_params(&mut model,  params,&OVector::from_vec_generic(base_function_count,U1,vec![c1, c2, c3]));
    let  problem = LevMarProblemBuilder::new(model)
        .observations(y)
        .build()
        .expect("Building valid problem should not panic");
    problem
}

/// solve the double exponential fitting problem using a handrolled model 
fn run_minimization_for_handrolled_separable_model(problem: LevMarProblem<DoubleExpModelWithConstantOffsetSepModel>) -> [f64; 5] {
    let (problem, report) = LevMarSolver::new().minimize(problem);
    assert!(
        report.termination.was_successful(),
        "Termination not successful"
    );
    let params = problem.params();
    let coeff = problem.linear_coefficients().unwrap();
    [params[0], params[1], coeff[0], coeff[1], coeff[2]]
}

/// solve the double exponential fitting problem using a separable model from the builder
/// I should be able to unify this with the handrolled model, but I can't figure out how to do it
/// because I cannot find the correct generic bounds to do it
fn run_minimization_for_builder_separable_model(problem: LevMarProblem<SeparableModel<f64>>) -> [f64; 5] {
    let (problem, report) = LevMarSolver::new().minimize(problem);
    assert!(
        report.termination.was_successful(),
        "Termination not successful"
    );
    let params = problem.params();
    let coeff = problem.linear_coefficients().unwrap();
    [params[0], params[1], coeff[0], coeff[1], coeff[2]]
}

/// solve the problem by using nonlinear least squares with levenberg marquardt
/// from the levenberg marquardt crate only
fn run_minimization_for_levenberg_marquardt_crate_problem(problem: DoubleExponentialDecayFittingWithOffsetLevmar) -> [f64; 5] {
    let (problem, report) = LevMarSolver::new()
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
    let tau_guess  = (2., 6.5);
    // function values
    let f = x.map(|x: f64| true_parameters.c1 * (-x / true_parameters.tau1).exp() + true_parameters.c2 * (-x / true_parameters.tau2).exp() + true_parameters.c3);
    
    group.bench_function("Using Levenberg Marquardt Crate", |bencher| {
        bencher.iter_batched(
            || DoubleExponentialDecayFittingWithOffsetLevmar::new(
                // we make it easier for the solver by giving it the 
                // correct guesses for the coefficients from the start
                // which is not gonna happen for realistic problems
                &[tau_guess.0, tau_guess.1, true_parameters.c1, true_parameters.c2, true_parameters.c3], 
                &x,&f
            ),
            run_minimization_for_levenberg_marquardt_crate_problem,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Using Model Builder", |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, get_double_exponential_model_with_constant_offset(x.clone(),vec![tau_guess.0,tau_guess.1])),
            run_minimization_for_builder_separable_model,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Handcrafted Model", |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, DoubleExpModelWithConstantOffsetSepModel::new(x.clone(),tau_guess)),
            run_minimization_for_handrolled_separable_model,
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_double_exp_no_noise);
criterion_main!(benches);