//! Native bridges for Python neural-network functional operators.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::types::{PyAny, PyModule};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    python::PyTensor,
    python_argument_schema::{ArgumentSchema, parse_float_like_argument},
    python_tensor_errors::tensor_error,
};

const STANDARD_DROPOUT_METADATA: DropoutMetadata = DropoutMetadata {
    public_function: "dropout",
    operation: "dropout",
    inplace_operation: "dropout_",
    supports_tensor_probability: true,
    supports_probability_one: true,
    required_rank: None,
};

const DROPOUT_METADATA: [DropoutMetadata; 6] = [
    STANDARD_DROPOUT_METADATA,
    DropoutMetadata {
        public_function: "alpha_dropout",
        operation: "alpha_dropout",
        inplace_operation: "alpha_dropout_",
        supports_tensor_probability: false,
        supports_probability_one: false,
        required_rank: None,
    },
    DropoutMetadata {
        public_function: "feature_alpha_dropout",
        operation: "feature_alpha_dropout",
        inplace_operation: "feature_alpha_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        required_rank: None,
    },
    DropoutMetadata {
        public_function: "dropout1d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        required_rank: Some(3),
    },
    DropoutMetadata {
        public_function: "dropout2d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        required_rank: Some(4),
    },
    DropoutMetadata {
        public_function: "dropout3d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        required_rank: Some(5),
    },
];

#[derive(Clone, Copy)]
struct DropoutMetadata {
    public_function: &'static str,
    operation: &'static str,
    inplace_operation: &'static str,
    supports_tensor_probability: bool,
    supports_probability_one: bool,
    required_rank: Option<usize>,
}

struct DropoutSchema {
    input: ArgumentSchema,
    probability: ArgumentSchema,
    training: ArgumentSchema,
}

#[derive(Clone, Copy)]
enum DropoutCallSite {
    Functional,
    TopLevel,
}

impl DropoutSchema {
    const fn new(metadata: DropoutMetadata, inplace: bool) -> Self {
        let operation = if inplace {
            metadata.inplace_operation
        } else {
            metadata.operation
        };
        Self {
            input: ArgumentSchema::new(operation, "input", 1, "Tensor"),
            probability: ArgumentSchema::new(operation, "p", 2, "float"),
            training: ArgumentSchema::new(operation, "train", 3, "bool"),
        }
    }
}

fn dropout_metadata(kind: &str) -> PyResult<DropoutMetadata> {
    DROPOUT_METADATA
        .iter()
        .copied()
        .find(|metadata| metadata.public_function == kind)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown dropout kind: {kind}")))
}

fn format_probability(py: Python<'_>, probability: f64) -> PyResult<String> {
    if probability.is_nan() {
        return Ok(if probability.is_sign_negative() {
            "-nan".to_owned()
        } else {
            "nan".to_owned()
        });
    }
    PyModule::import(py, "builtins")?
        .getattr("format")?
        .call1((probability, ".6g"))?
        .extract()
}

fn apply_dropout(
    py: Python<'_>,
    metadata: DropoutMetadata,
    tensor: &Bound<'_, PyTensor>,
    probability: f64,
    training: bool,
    inplace: bool,
    call_site: DropoutCallSite,
) -> PyResult<Py<PyAny>> {
    if !(0.0..=1.0).contains(&probability) {
        let probability = format_probability(py, probability)?;
        return Err(PyRuntimeError::new_err(format!(
            "dropout probability has to be between 0 and 1, but got {probability}"
        )));
    }

    if let Some(required_rank) = metadata.required_rank
        && tensor.try_borrow()?.inner().shape().len() != required_rank
    {
        return Err(PyNotImplementedError::new_err(format!(
            "torch_rs.nn.functional.{} only supports rank-{required_rank} inputs",
            metadata.public_function
        )));
    }

    let input_is_empty = tensor.try_borrow()?.inner().numel() == 0;
    if !training || probability == 0.0 || input_is_empty {
        return Ok(tensor.clone().unbind().into_any());
    }

    if metadata.supports_probability_one && !inplace && probability.to_bits() == 1.0_f64.to_bits() {
        let output = tensor
            .try_borrow()?
            .inner()
            .mul_scalar(0.0)
            .map_err(|error| tensor_error(&error))?;
        return PyTensor::new(output).into_py_any(py);
    }

    let public_path = match call_site {
        DropoutCallSite::Functional => {
            format!("torch_rs.nn.functional.{}", metadata.public_function)
        }
        DropoutCallSite::TopLevel => "torch_rs.dropout".to_owned(),
    };
    Err(PyNotImplementedError::new_err(format!(
        "{public_path} does not support sampling"
    )))
}

pub(crate) fn parse_top_level_dropout_probability(
    probability: &Bound<'_, PyAny>,
    position: Option<usize>,
) -> PyResult<f64> {
    parse_float_like_argument(
        ArgumentSchema::with_optional_position("dropout", "p", position, "float"),
        probability,
    )
}

pub(crate) fn parse_top_level_dropout_training(
    training: &Bound<'_, PyAny>,
    position: Option<usize>,
) -> PyResult<bool> {
    ArgumentSchema::with_optional_position("dropout", "train", position, "bool")
        .parse_exact_bool(training)
}

pub(crate) fn apply_top_level_dropout(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    probability: f64,
    training: bool,
) -> PyResult<Py<PyAny>> {
    apply_dropout(
        py,
        STANDARD_DROPOUT_METADATA,
        tensor,
        probability,
        training,
        false,
        DropoutCallSite::TopLevel,
    )
}

#[pyfunction]
fn _nn_functional_dropout(
    py: Python<'_>,
    kind: &str,
    input: &Bound<'_, PyAny>,
    probability: &Bound<'_, PyAny>,
    training: &Bound<'_, PyAny>,
    inplace: bool,
) -> PyResult<Py<PyAny>> {
    // This private bridge mirrors the native operator's schema checks for the
    // supported deterministic cases. It deliberately owns no random state or
    // mutation.
    let metadata = dropout_metadata(kind)?;
    let schema = DropoutSchema::new(metadata, inplace);
    let Ok(tensor) = input.cast::<PyTensor>() else {
        return Err(schema.input.type_error(input)?);
    };
    if !metadata.supports_tensor_probability && probability.cast::<PyTensor>().is_ok() {
        // Keep alpha dropout's supported probability surface to real scalars;
        // standard dropout already owns the scalar-Tensor compatibility path.
        return Err(schema.probability.type_error(probability)?);
    }
    let probability = parse_float_like_argument(schema.probability, probability)?;
    let training = schema.training.parse_exact_bool(training)?;
    apply_dropout(
        py,
        metadata,
        tensor,
        probability,
        training,
        inplace,
        DropoutCallSite::Functional,
    )
}

#[pyfunction]
fn _nn_functional_dropout_tensor_autograd_suffix(input: &PyTensor) -> String {
    if !input.inner().requires_grad() {
        return String::new();
    }
    input.inner().grad_fn_name().map_or_else(
        || ", requires_grad=True".to_owned(),
        |name| format!(", grad_fn=<{name}>"),
    )
}

pub(crate) fn add_nn_functional_bridges(module: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(_nn_functional_dropout, module)?,
        wrap_pyfunction!(_nn_functional_dropout_tensor_autograd_suffix, module)?,
    ] {
        let name = function.getattr("__name__")?;
        module.add_function(function.clone())?;
        module.getattr("__all__")?.call_method1("remove", (name,))?;
    }
    Ok(())
}
