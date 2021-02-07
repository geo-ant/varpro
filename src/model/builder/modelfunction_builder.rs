use nalgebra::base::{Dim, Scalar};
use nalgebra::Dynamic;

use crate::model::detail::{check_parameter_names, create_index_mapping, create_wrapper_function};
use crate::model::errors::ModelError;
use crate::model::modelfunction::ModelFunction;
use crate::model::OwnedVector;

/// The modelfunction builder allows to create model functions that depend on
/// a subset or the whole model parameters. Functions that depend on model parameters
/// need to have partial derivatives provided for each parameter they depend on.
pub struct ModelFunctionBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// the model parameters for the model this function belongs to
    model_parameters: Vec<String>,
    /// the parameters that the function depends on. Must be a subset of the model parameters
    function_parameters: Vec<String>,
    /// the current result of the building process of the model function
    model_function_result: Result<ModelFunction<ScalarType, NData>, ModelError>,
}

impl<ScalarType, NData> ModelFunctionBuilder<ScalarType, NData>
where
    ScalarType: Scalar,
    NData: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<ScalarType, NData>, //see https://github.com/dimforge/nalgebra/issues/580
{
    /// begin constructing a modelfunction for a specific model. The modelfunction must take
    /// a subset of the model parameters. This is the first step in creating a function, because
    /// the modelfunction must have partial derivatives specified for each parameter it takes.
    /// # Arguments
    /// * `model`: the model for which this function should be generated. This is important
    /// so the builder understands how the parameters of the function relate to the parameters of the model
    /// * `function_parameters`: the parameters that the function takes. Those must be in the order
    /// of the parameter vector. The paramters must not be empty, nor may they contain duplicates
    /// * `function`: the actual function.
    /// # Result
    /// A model builder that can be used to add derivatives.
    pub fn new<FuncType>(
        model_parameters: Vec<String>,
        function_parameters: &[String],
        function: FuncType,
    ) -> Self
    where
        FuncType: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
    {
        // check that the function parameter list is valid, otherwise continue with an
        // internal error state
        if let Err(err) = check_parameter_names(function_parameters) {
            return Self {
                model_function_result: Err(err),
                model_parameters,
                function_parameters: function_parameters.to_vec(),
            };
        }

        let model_function_result =
            create_wrapper_function(&model_parameters, function_parameters, function).map(
                |function| ModelFunction {
                    function,
                    derivatives: Default::default(),
                },
            );

        Self {
            model_function_result,
            model_parameters,
            function_parameters: function_parameters.to_vec(),
        }
    }

    /// Add a derivative for the function with respect to the given parameter.
    /// # Arguments
    /// * `parameter`: the parameter with respect to which the derivative is taken.
    /// The parameter must be inside the set of model parameters. Furthermore the
    /// * `derivative`: the partial derivative of the function with which the
    /// builder was created.
    pub fn partial_deriv<FuncType>(mut self, parameter: &str, derivative: FuncType) -> Self
    where
        FuncType: Fn(
                &OwnedVector<ScalarType, NData>,
                &OwnedVector<ScalarType, Dynamic>,
            ) -> OwnedVector<ScalarType, NData>
            + 'static,
    {
        //check to see if parameter is in list
        if let Some(deriv_index) = self
            .model_parameters
            .iter()
            .position(|model_param| model_param == parameter)
        {
            if let Ok(model_function) = self.model_function_result.as_mut() {
                match create_wrapper_function(
                    &self.model_parameters,
                    &self.function_parameters,
                    derivative,
                ) {
                    Ok(deriv) => {
                        // push derivative and check that the derivative was not already in the set
                        if model_function
                            .derivatives
                            .insert(deriv_index, deriv)
                            .is_some()
                        {
                            self.model_function_result = Err(ModelError::DuplicateDerivative {
                                parameter: parameter.into(),
                            });
                        }
                    }
                    Err(error) => {
                        self.model_function_result = Err(error);
                    }
                }
            }
            self
        } else {
            Self {
                model_function_result: Err(ModelError::InvalidDerivative {
                    parameter: parameter.into(),
                    function_parameters: self.function_parameters.clone(),
                }),
                ..self
            }
        }
    }

    /// Build a modelfunction with derivatives from the contents of this builder
    /// # Result
    /// A modelfunction containing the given function and derivatives or an error
    /// variant if an error occurred during the building process
    pub fn build(self) -> Result<ModelFunction<ScalarType, NData>, ModelError> {
        self.check_completion()?;
        self.model_function_result
    }

    /// If the modelfunction builder is carrying an Error result, this function returns Ok(()).
    /// If it carries an ok variant, then this function checks that the modelfunction
    /// has all necessary derivatives provided and if so returns Ok(()), otherwise it returns
    /// an error indicating that there are missing derivatives
    fn check_completion(&self) -> Result<(), ModelError> {
        match self.model_function_result.as_ref() {
            Ok(modelfunction) => {
                // this should not go wrong, but to be safe
                check_parameter_names(self.model_parameters.as_slice())?;
                check_parameter_names(self.function_parameters.as_slice())?;
                // create the index mapping
                let index_mapping = create_index_mapping(
                    self.model_parameters.as_slice(),
                    self.function_parameters.as_slice(),
                )?;
                // now make sure that the derivatives are provided for all indices of that mapping
                for (index, parameter) in index_mapping.iter().zip(self.function_parameters.iter())
                {
                    if !modelfunction.derivatives.contains_key(index) {
                        return Err(ModelError::MissingDerivative {
                            missing_parameter: parameter.clone(),
                            function_parameters: self.function_parameters.clone(),
                        });
                    }
                }
                // this is a sanity check. if this came this far, there should not be an error here
                if index_mapping.len() != modelfunction.derivatives.len() {
                    // this also should never be the case and indicates a programming error in the library
                    return Err(ModelError::Fatal {
                        message: "Too many or too few derivatives in set!".to_string(),
                    });
                }
                // otherwise
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::model::builder::modelfunction_builder::ModelFunctionBuilder;
    use crate::model::OwnedVector;
    use nalgebra::{Dim, Dynamic};

    /// a function that calculates exp( (t-t0)/tau)) for every location t
    fn exponential_decay<NData>(
        tvec: &OwnedVector<f64, NData>,
        t0: f64,
        tau: f64,
    ) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
    {
        assert!(tau > 0f64, "Parameter tau must be greater than zero");
        tvec.map(|t| ((t - t0) / tau).exp())
    }

    /// partial derivative of the exponential decay with respect to t0
    fn exponential_decay_dt0<NData>(
        tvec: &OwnedVector<f64, NData>,
        t0: f64,
        tau: f64,
    ) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
    {
        assert!(tau > 0f64, "Parameter tau must be greater than zero");
        exponential_decay(tvec, t0, tau).map(|val| val / tau)
    }

    /// partial derivative of the exponential decay with respect to tau
    fn exponential_decay_dtau<NData>(
        tvec: &OwnedVector<f64, NData>,
        t0: f64,
        tau: f64,
    ) -> OwnedVector<f64, NData>
    where
        NData: Dim,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, NData>, //see https://github.com/dimforge/nalgebra/issues/580 {
    {
        assert!(tau > 0f64, "Parameter tau must be greater than zero");
        tvec.map(|t| ((t - t0) / tau).exp() * (t0 - t) / tau.powi(2))
    }

    #[test]
    // check that the modelfunction builder assigns the function and derivatives correctly
    // and that they can be called using the model parameters and produce the correct results
    fn modelfunction_builder_creates_correct_modelfunction_with_valid_parameters() {
        let model_parameters = vec![
            "foo".to_string(),
            "t0".to_string(),
            "bar".to_string(),
            "tau".to_string(),
        ];
        let mf = ModelFunctionBuilder::<f64,Dynamic>::new(
            model_parameters,
            ["t0".to_string(), "tau".to_string()].as_ref(),
            |t, params| exponential_decay(t, params[0], params[1]),
        )
        .partial_deriv("t0", |t, params| {
            exponential_decay_dt0(t, params[0], params[1])
        })
        .partial_deriv("tau", |t, params| {
            exponential_decay_dtau(t, params[0], params[1])
        })
        .build()
        .expect("Modelfunction builder with valid parameters should not return an error");

        let t0 = 2.;
        let tau = 1.5;
        let model_params = OwnedVector::<f64,Dynamic>::from(vec!{-2.,t0,-1.,tau});
        let t = OwnedVector::<f64,Dynamic>::from(vec!{0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.});
        assert_eq!( (mf.function)(&t,&model_params),exponential_decay(&t,t0,tau),"Function must produce correct results");
        assert_eq!( (mf.derivatives.get(&1).expect("Derivative for t0 must be in set"))(&t,&model_params),exponential_decay_dt0(&t,t0,tau),"Derivative for t0 must produce correct results");
        assert_eq!( (mf.derivatives.get(&3).expect("Derivative for tau must be in set"))(&t,&model_params),exponential_decay_dtau(&t,t0,tau),"Derivative for tau must produce correct results");
    }
}
