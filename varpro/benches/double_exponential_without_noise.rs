use common::*;
use criterion::{criterion_group, criterion_main, Criterion};
use detail::DoubleExpModelWithConstantOffset;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::ComplexField;
use nalgebra::DVector;
use nalgebra::RealField;
use nalgebra::Scalar;
use num_traits::Float;
use pprof::criterion::{Output, PProfProfiler};
use varpro::model::SeparableModel;
use varpro::prelude::SeparableNonlinearModel;
use varpro::solvers::levmar::LevMarProblem;
use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

mod detail;

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
    let model = DoubleExpModelWithConstantOffset::default();
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

mod common {
    use nalgebra::{ComplexField, DVector, Scalar};
    use num_traits::Float;
    use varpro::model::builder::SeparableModelBuilder;
    use varpro::model::SeparableModel;
    use varpro::prelude::SeparableNonlinearModel;

    /// create holding `count` the elements from range [first,last] with linear spacing. (equivalent to matlabs linspace)
    pub fn linspace<ScalarType: Float + Scalar>(
        first: ScalarType,
        last: ScalarType,
        count: usize,
    ) -> DVector<ScalarType> {
        let n_minus_one = ScalarType::from(count - 1).expect("Could not convert usize to Float");
        let lin: Vec<ScalarType> = (0..count)
            .map(|n| {
                first
                    + (first - last) / (n_minus_one)
                        * ScalarType::from(n).expect("Could not convert usize to Float")
            })
            .collect();
        DVector::from(lin)
    }

    /// evaluete the vector valued function of a model by evaluating the model at the given location
    /// `x` with (nonlinear) parameters `params` and by calculating the linear superposition of the basisfunctions
    /// with the given linear coefficients `linear_coeffs`.
    pub fn evaluate_complete_model<ScalarType, Model>(
        model: &'_ Model,
        x: &DVector<ScalarType>,
        params: &[ScalarType],
        linear_coeffs: &DVector<ScalarType>,
    ) -> DVector<ScalarType>
    where
        ScalarType: Scalar + ComplexField,
        Model : SeparableNonlinearModel<ScalarType>
    {
        (&model
            .eval(x, params)
            .expect("Evaluating model must not produce error"))
            * linear_coeffs
    }

    /// exponential decay f(t,tau) = exp(-t/tau)
    pub fn exp_decay<ScalarType: Float + Scalar>(
        tvec: &DVector<ScalarType>,
        tau: ScalarType,
    ) -> DVector<ScalarType> {
        tvec.map(|t| (-t / tau).exp())
    }

    /// derivative of exp decay with respect to tau
    pub fn exp_decay_dtau<ScalarType: Scalar + Float>(
        tvec: &DVector<ScalarType>,
        tau: ScalarType,
    ) -> DVector<ScalarType> {
        tvec.map(|t| (-t / tau).exp() * t / (tau * tau))
    }

    /// A helper function that returns a double exponential decay model
    /// f(x,tau1,tau2) = c1*exp(-x/tau1)+c2*exp(-x/tau2)+c3
    /// Model parameters are: tau1, tau2
    pub fn get_double_exponential_model_with_constant_offset() -> SeparableModel<f64> {
        let ones = |t: &DVector<_>| DVector::from_element(t.len(), 1.);

        SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
            .function(&["tau1"], exp_decay)
            .partial_deriv("tau1", exp_decay_dtau)
            .function(&["tau2"], exp_decay)
            .partial_deriv("tau2", exp_decay_dtau)
            .invariant_function(ones)
            .build()
            .expect("double exponential model builder should produce a valid model")
    }
}
