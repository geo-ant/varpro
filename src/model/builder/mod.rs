use nalgebra::{DVector, Scalar};

use crate::basis_function::BasisFunction;
use crate::model::builder::modelfunction_builder::ModelBasisFunctionBuilder;
use crate::model::detail::check_parameter_names;
use crate::model::model_basis_function::ModelBasisFunction;
use crate::model::SeparableModel;
use error::ModelBuildError;

mod modelfunction_builder;

#[cfg(any(test, doctest))]
mod test;

/// contains the error for the model builder
pub mod error;

/// A builder that allows us to construct a valid [SeparableModel](crate::model::SeparableModel),
/// which is an implementor of the [SeparableNonlinearModel](crate::model::SeparableNonlinearModel)
/// trait.
///
/// # Introduction
///
/// In the main crate we defined a separable model as a vector valued function
/// `$\vec{f}(\vec{\alpha},\vec{c})$`, but we are going to deviate from this
/// definition slightly here. We want to provide an *independent variable* `$\vec{x}$`
/// that the function depends on, to make a model usable on different supports.
///
/// To make the dependence on the independent variable explitict,
/// we now writethe separable model as
/// ```math
///  \vec{f}(\vec{x},\vec{\alpha},\vec{c}) = \sum_{j=1}^{N_{basis}} c_j \vec{f}_j(\vec{x},S_j(\alpha))
/// ```
///
/// The basis functions `$\vec{f}_j(\vec{x},S_j(\alpha))$` depend on the independent variable `$\vec{x}$`
/// and *a subset* `$S_j(\alpha)$` of the *nonlinear* model parameters `$\vec{\alpha}$`
/// just as in the other notation.
///
/// # Usage
/// The SeparableModelBuilder is concerned with building a model from basis functions and their derivatives.
/// This is done as a step by step process.
///
/// ## Constructing an Empty Builder
/// The first step is to create an empty builder by specifying the complete set of *nonlinear* parameters that
/// the model will be depending on. This is done by calling [SeparableModelBuilder::new](SeparableModelBuilder::new)
/// and specifying the list of parameters of the model by name.
///
/// ## Adding Basis Functions to The Model
///
/// Basis functions come in two flavors. Those that depend on a subset of the nonlinear parameters
/// `$\vec{\alpha}$` and those that do not. Both function types have to obey certain rules to
/// be considered valid:
///
/// **Function Arguments and Output**
///
/// * The first argument of the function must be a reference to a `&DVector` type
/// that accepts the independent variable (the `$\vec{x}$` values) and the other
/// parameters must be scalars that are the nonlinear parameters that the basis
/// function depends on.
///
/// So if we want to model a basis function `$\vec{f_1}(\vec{x},\vec{\alpha})$`
/// where `$\vec{\alpha}=(\alpha_1,\alpha_2)$` we would write the function in Rust as
///
/// ```rust
/// # use nalgebra::DVector;
/// fn f1(x: &DVector<f32>, alpha1: f32, alpha2: f32) -> DVector<f32> {
///    // e.g. for sinusoidal function with frequency alpha1 and phase alpha2
///    // apply the function elementwise to the vector x
///    x.map(|x| f32::sin(alpha1*x+alpha2))
/// }
/// ```
/// using single precision (`f32`) floats.
///
/// **Linear Independence**
///
/// The basis functions must be linearly independent. That means adding `$\vec{f_1}(\vec{x})=\vec{x}$`
/// and `$\vec{f_1}(\vec{x})=2\,\vec{x}$` is forbidden. Adding functions that
/// are lineary dependent will possibly destabilize the fitting process.
/// the calculations. Adding linearly dependent functions is also a bad idea
/// because it adds no value due to the linear superposition of the basis functions.
///
/// For some models, e.g. sums of exponential decays it might happen that the basis functions become
/// linearly dependent *for some combinations* of nonlinear model parameters. This isn't great but it is
/// okay, since the VarPro algorithm in this crate exhibits a degree of robustness against basis functions
/// becoming collinear (see [LevMarProblemBuilder::epsilon](crate::solvers::levmar::LevMarProblemBuilder::epsilon)).
///
/// ### Invariant Basis Functions
///
/// Basis functions that do not depend on model parameters are treated specially. The library refers
/// to them as *invariant functions* and they are added to a builder by calling
/// [SeparableModelBuilder::invariant_function](SeparableModelBuilder::invariant_function). Since
/// the basis function depends only on `$\vec{x}$` it can be written as `$\vec{f}_j(\vec{x})$`. In Rust
/// this translates to a signature `Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static` for the callable.
///
/// **Example**: Calling [SeparableModelBuilder::invariant_function](SeparableModelBuilder::invariant_function)
/// adds the function to the model. These calls can be chained to add more functions.
///
/// ```rust
/// use nalgebra::DVector;
/// use varpro::prelude::SeparableModelBuilder;
/// fn squared(x: &DVector<f64>) -> DVector<f64> {
///     x.map(|x|x.powi(2))
/// }
///
/// let builder = SeparableModelBuilder::<f64>::new(&["alpha","beta"])
///                 // we can add an invariant function using a function pointer
///                 .invariant_function(squared)
///                 // or we can add it using a lambda
///                 .invariant_function(|x|x.map(|x|(x+1.).sin()));
///```
/// Caveat: we cannot successfully build a model containing only invariant functions. It would
/// make no sense to use the varpro library to fit such a model because that is purely a linear
/// least squares problem. See the next section for adding parameter dependent functions.
///
/// ### Nonlinear Basis Functions
///
/// The core functionality of the builder is to add basis functions to the model
/// that depend nonlinearly on some (or all) of the model parameters `$\vec{\alpha}$`.
/// We add a basis function to a builder by calling `builder.function`. Each call must
/// be immediately followed by calls to `partial_deriv` for each of the parameters that the basis
/// function depends on.
///
/// #### Rules for Model Functions
///
/// There are several rules for adding model basis functions. One of them is enforced by the compiler,
/// some of them are enforced at runtime (when trying to build the model) and others simply cannot
/// be enforced by the library.
///
/// ** Rules You Must Abide By **
///
/// * Basis functions must be **nonlinear** in the parameters they take. If they aren't, you can always
/// rewrite the problem so that the linear parameters go in the coefficient vector `$\vec{c}$`. This
/// means that each partial derivative also depend on all the parameters that the basis function depends
/// on.
///
/// * Derivatives must take the same parameter arguments *and in the same order* as the original
/// basis function. This means if basis function `$\vec{f}_j$` is given as `$\vec{f}_j(\vec{x},a,b)$`,
/// then the derivatives must also be given with the parameters `$a,b$` in the same order, i.e.
/// `$\partial/\partial a \vec{f}_j(\vec{x},a,b)$`, `$\partial/\partial b \vec{f}_j(\vec{x},a,b)$`.
///
/// **Rules Enforced at Compile Time**
///
/// * Partial derivatives cannot be added to invariant functions.
///
/// **Rules Enforced at Runtime**
///
/// * A partial derivative must be given for each parameter that the basis function depends on.
/// * Basis functions may only depend on the parameters that the model depends on.
///
///
/// The builder allows us to provide basis functions for a separable model as a step by step process.
///
/// ## Example
///
/// Let's build a model that is the sum of an exponential decay `$\exp(-t/\tau)$`
/// and a sine  function `$\sin(\omega t + \phi)$`. The model depends on the parameters `$\tau$`,
/// `$\omega$` and `$\phi$`. The exponential decay depends only on `$\tau$` and the sine function
/// depends on `$\omega$` and `$\phi$`. The model is given by
///
/// ```math
/// f(t,\tau,\omega,\phi) = \exp(-t/\tau) + \sin(\omega t + \phi)
/// ```
/// which is a reasonable nontrivial model to demonstrate the usage of the library.
///
/// ```rust
/// // exponential decay f(t,tau) = exp(-t/tau)
/// use nalgebra::{Scalar, DVector};
/// use num_traits::Float;
/// use varpro::prelude::SeparableModelBuilder;
/// pub fn exp_decay<ScalarType: Float + Scalar>(
///     tvec: &DVector<ScalarType>,
///     tau: ScalarType,
/// ) -> DVector<ScalarType> {
///     tvec.map(|t| (-t / tau).exp())
/// }
///
/// // derivative of exp decay with respect to tau
/// pub fn exp_decay_dtau<ScalarType: Scalar + Float>(
///     tvec: &DVector<ScalarType>,
///     tau: ScalarType,
/// ) -> DVector<ScalarType> {
///     tvec.map(|t| (-t / tau).exp() * t / (tau * tau))
/// }
///
/// // function sin (omega*t+phi)
/// pub fn sin_ometa_t_plus_phi<ScalarType: Scalar + Float>(
///     tvec: &DVector<ScalarType>,
///     omega: ScalarType,
///     phi: ScalarType,
/// ) -> DVector<ScalarType> {
///     tvec.map(|t| (omega * t + phi).sin())
/// }
///
/// // derivative d/d(omega) sin (omega*t+phi)
/// pub fn sin_ometa_t_plus_phi_domega<ScalarType: Scalar + Float>(
///     tvec: &DVector<ScalarType>,
///     omega: ScalarType,
///     phi: ScalarType,
/// ) -> DVector<ScalarType> {
///     tvec.map(|t| t * (omega * t + phi).cos())
/// }
///
/// // derivative d/d(phi) sin (omega*t+phi)
/// pub fn sin_ometa_t_plus_phi_dphi<ScalarType: Scalar + Float>(
///     tvec: &DVector<ScalarType>,
///     omega: ScalarType,
///     phi: ScalarType,
/// ) -> DVector<ScalarType> {
///     tvec.map(|t| (omega * t + phi).cos())
/// }
///
/// let x_coords = DVector::from_vec(vec![0.,1.,2.,3.,4.,5.]);
/// let initial_guess = vec![1.,1.,1.];
///
/// let model = SeparableModelBuilder::<f64>::new(&["tau","omega","phi"])
///               // the x coordintates that this model
///               // is evaluated on
///               .independent_variable(x_coords)
///               // add the exp decay and all derivatives
///               .function(&["tau"],exp_decay)
///               .partial_deriv("tau",exp_decay_dtau)
///               // a new call to function finalizes adding the previous function
///               .function(&["omega","phi"],sin_ometa_t_plus_phi)
///               .partial_deriv("phi", sin_ometa_t_plus_phi_dphi)
///               .partial_deriv("omega",sin_ometa_t_plus_phi_domega)
///               // we can also add invariant functions. Same as above, the
///               // call tells the model builder that the previous function has all
///               // the partial derivatives finished
///               .invariant_function(|x|x.clone())
///               // the initial nonlinear parameters
///               // of the model
///               .initial_parameters(initial_guess)
///               // we build the model calling build. This returns either a valid model
///               // or an error variant which is pretty helpful in understanding what went wrong
///               .build().unwrap();
/// ```
///
/// There is some [special macro magic](https://geo-ant.github.io/blog/2021/rust-traits-and-variadic-functions/)
/// that allows us to pass a function `$f(\vec{x},a_1,..,a_n)$`
/// as any item that implements the Rust trait `Fn(&DVector<ScalarType>, ScalarType,... ,ScalarType)->DVector<ScalarType> + 'static`.
/// This allows us to write the functions in an intuitive fashion in Rust code. All nonlinear parameters `$\alpha$`
/// are simply scalar arguments in the parameter list of the function. This works for functions
/// taking up to 10 nonlinear arguments, but can be extended easily by modifying this crates source.
///
///
/// ## Building a Model
///
/// The model is finalized and built using the [SeparableModelBuilder::build](SeparableModelBuilder::build)
/// method. This method returns a valid model or an error variant doing a pretty good job of
/// explaning why the model is invalid.
#[must_use = "The builder should be transformed into a model using the build() method"]
pub struct SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    model_result: Result<UnfinishedModel<ScalarType>, ModelBuildError>,
}

/// a helper structure that represents an unfinished separable model
#[derive(Default)]
struct UnfinishedModel<ScalarType: Scalar> {
    /// the parameter names
    parameter_names: Vec<String>,
    /// the base functions
    basefunctions: Vec<ModelBasisFunction<ScalarType>>,
    /// the x-vector (independent variable associated with the model)
    x_vector: Option<DVector<ScalarType>>,
    /// the initial guesses for the parameters
    initial_parameters: Option<Vec<ScalarType>>,
}

/// create a SeparableModelBuilder which contains an error variant
impl<ScalarType> From<ModelBuildError> for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelBuildError) -> Self {
        Self {
            model_result: Err(err),
        }
    }
}

/// create a SeparableModelBuilder with the given result variant
impl<ScalarType> From<Result<UnfinishedModel<ScalarType>, ModelBuildError>>
    for SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(model_result: Result<UnfinishedModel<ScalarType>, ModelBuildError>) -> Self {
        Self { model_result }
    }
}

impl<ScalarType> SeparableModelBuilder<ScalarType>
where
    ScalarType: Scalar,
{
    /// Create a new builder for a model that depends on this list of paramters
    /// The model parameters indices correspond to the order of appearance in here.
    /// Model parameter indices start at 0.
    /// # Arguments
    /// * `parameter_names` A collection containing all the nonlinear model parameters
    /// # Requirements on the Parameters
    /// * The list of parameters must only contain unique names
    /// * The list of parameter names must not be empty
    /// * Parameter names must not contain a comma. This is a precaution because
    /// `&["alpha,beta"]` most likely indicates a typo for `&["alpha","beta"]`. Any other form
    /// of punctuation is allowed.
    pub fn new<StrCollection>(parameter_names: StrCollection) -> Self
    where
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        let parameter_names: Vec<String> = parameter_names
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        if let Err(parameter_error) = check_parameter_names(&parameter_names) {
            Self {
                model_result: Err(parameter_error),
            }
        } else {
            let model_result = Ok(UnfinishedModel {
                parameter_names,
                basefunctions: Vec::new(),
                x_vector: None,
                initial_parameters: None,
            });
            Self { model_result }
        }
    }

    /// Add a function `$\vec{f}(\vec{x})$` to the model. In the `varpro` library this is called
    /// an *invariant* function because the model function is independent of the model parameters
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
    pub fn invariant_function<F>(mut self, function: F) -> Self
    where
        F: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        if let Ok(model) = self.model_result.as_mut() {
            model
                .basefunctions
                .push(ModelBasisFunction::parameter_independent(function));
        }
        self
    }

    /// Add a function `$\vec{f}(\vec{x},\alpha_1,...,\alpha_n)$` to the model that depends on the
    /// location parameter `\vec{x}` and nonlinear model parameters.
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
    pub fn function<F, StrCollection, ArgList>(
        self,
        function_params: StrCollection,
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        SeparableModelBuilderProxyWithDerivatives::new(self.model_result, function_params, function)
    }

    /// Set the independent variable `$x$` which will be used when evaluating the model.
    /// Also see the struct documentation of [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// for information on how to use this method.
    pub fn independent_variable(mut self, x: DVector<ScalarType>) -> Self {
        if let Ok(model) = self.model_result.as_mut() {
            model.x_vector = Some(x);
        }
        self
    }

    /// Set the initial values for the model parameters `$\vec{\alpha}$`.
    /// Also see the struct documentation of [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// for information on how to use this method.
    pub fn initial_parameters(mut self, initial_parameters: Vec<ScalarType>) -> Self {
        if let Ok(model) = self.model_result.as_mut() {
            let expected = model.parameter_names.len();
            if expected != initial_parameters.len() {
                self.model_result = Err(ModelBuildError::IncorrectParameterCount {
                    expected,
                    actual: initial_parameters.len(),
                });
            } else {
                model.initial_parameters = Some(initial_parameters);
            }
        }
        self
    }

    /// Build a separable model from the contents of this builder.
    /// # Result
    /// A valid separable model or an error indicating why a valid model could not be constructed.
    ///
    /// ## Valid Models
    /// A model is valid if the following conditions where upheld during construction
    /// * The list of parameters is valid (see [SeparableModelBuilder::new](SeparableModelBuilder::new))
    /// * Each basis function depends on valid parameters
    /// * Each basis function only depends on (a subset of) the parameters given on model construction
    /// * For each parameter in the model, there is at least one function that depends on it
    ///
    /// # Order of the Basis Functions in the Model
    /// **Note** The order of basis functions in the model is order in which the basis functions
    /// where provided during the builder stage. That means the first basis functions gets index `0` in
    /// the model, the second gets index `1` and so on.
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelBuildError> {
        self.model_result.and_then(TryInto::try_into)
    }
}

/// try to convert an unfinished model into a valid model
/// # Returns
/// If the model is valid, then the model is returned as an ok variant, otherwise an error variant
/// A model is valid, when
///  * the model has at least one modelfunction, and
///  * for each model parameter we have at least one function that depends on this
///    parameter.
impl<ScalarType: Scalar> TryInto<SeparableModel<ScalarType>> for UnfinishedModel<ScalarType> {
    type Error = ModelBuildError;
    fn try_into(self) -> Result<SeparableModel<ScalarType>, ModelBuildError> {
        if self.basefunctions.is_empty() {
            Err(ModelBuildError::EmptyModel)
        } else if self.parameter_names.is_empty() {
            Err(ModelBuildError::EmptyParameters)
        } else {
            // now check that all model parameters are referenced in at least one parameter of one
            // of the given model functions. We do this by checking the indices of the derivatives
            // because each model parameter must occur at least once as a key in at least one of the
            // modelfunctions derivatives
            for (param_index, parameter_name) in self.parameter_names.iter().enumerate() {
                if !self
                    .basefunctions
                    .iter()
                    .any(|function| function.derivatives.contains_key(&param_index))
                {
                    return Err(ModelBuildError::UnusedParameter {
                        parameter: parameter_name.clone(),
                    });
                }
            }
            let x_vector = self.x_vector.ok_or(ModelBuildError::MissingX)?;
            let initial_parameters = self
                .initial_parameters
                .ok_or(ModelBuildError::MissingInitialParameters)?;
            // otherwise return this model
            Ok(SeparableModel {
                parameter_names: self.parameter_names,
                basefunctions: self.basefunctions,
                x_vector,
                current_parameters: DVector::from_vec(initial_parameters),
            })
        }
    }
}
/// helper struct that contains a seperable model as well as a model function builder
/// used inside the SeparableModelBuilderProxyWithDerivatives.
struct ModelAndModelBasisFunctionBuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    model: UnfinishedModel<ScalarType>,
    builder: ModelBasisFunctionBuilder<ScalarType>,
}

impl<ScalarType> ModelAndModelBasisFunctionBuilderPair<ScalarType>
where
    ScalarType: Scalar,
{
    fn new(
        model: UnfinishedModel<ScalarType>,
        builder: ModelBasisFunctionBuilder<ScalarType>,
    ) -> Self {
        Self { model, builder }
    }
}

/// This is just a proxy that does need to be used directly. For constructing a model from
/// a builder see the documentation for [SeparableModelBuilder](self::SeparableModelBuilder).
/// **Sidenote** This structure will hopefully be made more elegant using some metaprogramming techniques
/// in the future. Right now this exists to make sure that partial derivatives cannot accidentally
/// be added to invariant functions. The compiler simply will forbid it. In future the library aims
/// to make more such checks at compile time and reduce the need for runtime errors caused by invalid model
/// construction.
#[must_use = "This is meant as a transient expression proxy. Use build() to build a model."]
pub struct SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    current_result: Result<ModelAndModelBasisFunctionBuilderPair<ScalarType>, ModelBuildError>,
}

impl<ScalarType> From<ModelBuildError> for SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    fn from(err: ModelBuildError) -> Self {
        Self {
            current_result: Err(err),
        }
    }
}

impl<ScalarType> SeparableModelBuilderProxyWithDerivatives<ScalarType>
where
    ScalarType: Scalar,
{
    /// Construct an instance. This is invoked when adding a function that depends on model
    /// parameters to the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// # Arguments
    /// * `model_result`: the current model or an error
    /// * `function_parameters`: the given list of nonlinear function parameters
    /// * `function`: the function
    fn new<F, StrCollection, ArgList>(
        model_result: Result<UnfinishedModel<ScalarType>, ModelBuildError>,
        function_parameters: StrCollection,
        function: F,
    ) -> Self
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        match model_result {
            Ok(model) => {
                let model_parameters = model.parameter_names.clone();
                Self {
                    current_result: Ok(ModelAndModelBasisFunctionBuilderPair::new(
                        model,
                        ModelBasisFunctionBuilder::new(
                            model_parameters,
                            function_parameters,
                            function,
                        ),
                    )),
                }
            }
            Err(err) => Self {
                current_result: Err(err),
            },
        }
    }

    /// Add a partial derivative to a function. This function is documented as part of the
    /// [SeparableModelBuilder](self::SeparableModelBuilder) documentation.
    /// *Info*: the reason it appears in this proxy only is to make sure that partial derivatives
    /// can only be added to functions that depend on model parameters.
    pub fn partial_deriv<StrType: AsRef<str>, F, ArgList>(
        self,
        parameter: StrType,
        derivative: F,
    ) -> Self
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
    {
        match self.current_result {
            Ok(result) => Self {
                current_result: Ok(ModelAndModelBasisFunctionBuilderPair {
                    model: result.model,
                    builder: result.builder.partial_deriv(parameter.as_ref(), derivative),
                }),
            },
            Err(err) => Self::from(err),
        }
    }

    /// Add a function `\vec{f}(\vec{x})` that does not depend on the model parameters. This is documented
    /// as part of the [SeparableModelBuilder](self::SeparableModelBuilder) documentation.
    pub fn invariant_function<F>(self, function: F) -> SeparableModelBuilder<ScalarType>
    where
        F: Fn(&DVector<ScalarType>) -> DVector<ScalarType> + 'static,
    {
        match self.current_result {
            Ok(pair) => {
                let model_result = extend_model(pair.model, pair.builder);
                SeparableModelBuilder::from(model_result).invariant_function(function)
            }
            Err(err) => SeparableModelBuilder::from(err),
        }
    }

    /// Add a function `\vec{f}(\vec{x},\alpha_j,...,\alpha_k)` that depends on a subset of the
    /// model parameters. This functionality is documented as part of the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// documentation.
    pub fn function<F, StrCollection, ArgList>(
        self,
        function_params: StrCollection,
        function: F,
    ) -> SeparableModelBuilderProxyWithDerivatives<ScalarType>
    where
        F: BasisFunction<ScalarType, ArgList> + 'static,
        StrCollection: IntoIterator,
        StrCollection::Item: AsRef<str>,
    {
        match self.current_result {
            Ok(pair) => {
                let model_result = extend_model(pair.model, pair.builder);
                Self::new(model_result, function_params, function)
            }
            Err(err) => SeparableModelBuilderProxyWithDerivatives::from(err),
        }
    }

    /// Set the independent variable `\vec{x}`
    ///
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
    pub fn independent_variable(self, x: DVector<ScalarType>) -> SeparableModelBuilder<ScalarType> {
        match self.current_result {
            Ok(pair) => {
                let model_result = extend_model(pair.model, pair.builder);
                SeparableModelBuilder::from(model_result).independent_variable(x)
            }
            Err(err) => SeparableModelBuilder::from(err),
        }
    }

    /// Set the initial value for the nonlinear parameters `\vec{\alpha}`
    ///
    /// # Usage
    /// For usage see the documentation of the [SeparableModelBuilder](crate::model::builder::SeparableModelBuilder)
    /// struct documentation.
    pub fn initial_parameters(
        self,
        initial_parameters: Vec<ScalarType>,
    ) -> SeparableModelBuilder<ScalarType> {
        match self.current_result {
            Ok(pair) => {
                let model_result = extend_model(pair.model, pair.builder);
                SeparableModelBuilder::from(model_result).initial_parameters(initial_parameters)
            }
            Err(err) => SeparableModelBuilder::from(err),
        }
    }

    /// Finalized the building process and build a separable model.
    /// This functionality is documented as part of the [SeparableModelBuilder](self::SeparableModelBuilder)
    /// documentation.
    pub fn build(self) -> Result<SeparableModel<ScalarType>, ModelBuildError> {
        // this method converts the internal results into a separable model and uses its
        // facilities to check for completion and the like
        match self.current_result {
            Ok(pair) => {
                let model_result = extend_model(pair.model, pair.builder);
                SeparableModelBuilder::from(model_result).build()
            }
            Err(err) => SeparableModelBuilder::from(err).build(),
        }
    }
}

/// try and extend a model with the given function in the builder
/// if building the function in the builder fails, an error is returned,
/// otherwise the extended model is returned
fn extend_model<ScalarType: Scalar>(
    mut model: UnfinishedModel<ScalarType>,
    builder: ModelBasisFunctionBuilder<ScalarType>,
) -> Result<UnfinishedModel<ScalarType>, ModelBuildError> {
    let function = builder.build()?;
    model.basefunctions.push(function);
    Ok(model)
}
