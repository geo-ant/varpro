use criterion::{criterion_group, criterion_main, Criterion};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::ComplexField;

use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimMax;
use nalgebra::DimMin;
use nalgebra::DimSub;
use nalgebra::Dyn;
use nalgebra::OVector;
use nalgebra::RawStorageMut;
use nalgebra::RealField;
use nalgebra::Scalar;
use nalgebra::Storage;
use nalgebra::U1;
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

fn build_problem<Model>(
    true_parameters: DoubleExponentialParameters,
    mut model: Model,
) -> LevMarProblem<f64, Model> 
where
    Model: SeparableNonlinearModel<f64>,
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
        .y(y)
        .build()
        .expect("Building valid problem should not panic");
    problem
}

fn run_minimization<Model>(problem: LevMarProblem<f64, Model>) -> [f64; 5]
where
    Model: SeparableNonlinearModel<f64>,
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
    <Model as SeparableNonlinearModel<f64>>::OutputDim: DimMin<<Model as SeparableNonlinearModel<f64>>::ParameterDim>,
    <Model as SeparableNonlinearModel<f64>>::OutputDim: DimMax<<Model as SeparableNonlinearModel<f64>>::ParameterDim>, 
    DefaultAllocator: nalgebra::allocator::Allocator<usize, <Model as SeparableNonlinearModel<f64>>::ParameterDim>
{
    let (problem, report) = LevMarSolver::new().minimize(problem);
    // assert!(
    //     report.termination.was_successful(),
    //     "Termination not successful"
    // );

    // let params = problem.params();
    // let coeff = problem.linear_coefficients().unwrap();
    // [params[0], params[1], coeff[0], coeff[1], coeff[2]]
    todo!()
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
            || build_problem(true_parameters, get_double_exponential_model_with_constant_offset(x.clone(),vec![tau_guess.0,tau_guess.1])),
            run_minimization,
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("Handcrafted Model", |bencher| {
        bencher.iter_batched(
            || build_problem(true_parameters, DoubleExpModelWithConstantOffsetSepModel::new(x.clone(),tau_guess)),
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
