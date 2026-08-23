//! Native bridges for Python neural-network functional operators.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::types::{PyAny, PyModule};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    python::PyTensor,
    python_argument_schema::{ArgumentSchema, parse_float_like_argument},
    python_tensor_errors::tensor_error,
};

const DROPOUT_METADATA: [DropoutMetadata; 6] = [
    DropoutMetadata {
        public_function: "dropout",
        operation: "dropout",
        inplace_operation: "dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "alpha_dropout",
        operation: "alpha_dropout",
        inplace_operation: "alpha_dropout_",
        supports_tensor_probability: false,
        supports_probability_one: false,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "feature_alpha_dropout",
        operation: "feature_alpha_dropout",
        inplace_operation: "feature_alpha_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "dropout1d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        supported_ranks: Some(&[2, 3]),
    },
    DropoutMetadata {
        public_function: "dropout2d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        supported_ranks: Some(&[2, 3, 4]),
    },
    DropoutMetadata {
        public_function: "dropout3d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: Some(&[5]),
    },
];

#[derive(Clone, Copy)]
struct DropoutMetadata {
    public_function: &'static str,
    operation: &'static str,
    inplace_operation: &'static str,
    supports_tensor_probability: bool,
    supports_probability_one: bool,
    supported_ranks: Option<&'static [usize]>,
}

struct DropoutSchema {
    input: ArgumentSchema,
    probability: ArgumentSchema,
    training: ArgumentSchema,
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

    if !(0.0..=1.0).contains(&probability) {
        let probability = format_probability(py, probability)?;
        return Err(PyRuntimeError::new_err(format!(
            "dropout probability has to be between 0 and 1, but got {probability}"
        )));
    }

    let input_rank = tensor.try_borrow()?.inner().shape().len();
    if let Some(supported_ranks) = metadata.supported_ranks
        && !supported_ranks.contains(&input_rank)
    {
        let supported_ranks = supported_ranks
            .iter()
            .map(|rank| format!("rank-{rank}"))
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(PyNotImplementedError::new_err(format!(
            "torch_rs.nn.functional.{} only supports {supported_ranks} inputs",
            metadata.public_function,
        )));
    }

    let input_is_empty = tensor.try_borrow()?.inner().numel() == 0;
    let is_unbatched_dropout1d = metadata.public_function == "dropout1d" && input_rank == 2;
    if is_unbatched_dropout1d && inplace {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.dropout1d does not support inplace=True for rank-2 inputs",
        ));
    }
    if !training || probability == 0.0 || input_is_empty {
        if is_unbatched_dropout1d {
            // PyTorch routes an unbatched input through unsqueeze(0) and
            // squeeze(0), so even an identity feature-dropout call returns a
            // distinct view with a SqueezeBackward1 edge.
            let output = tensor
                .try_borrow()?
                .inner()
                .unsqueeze_front()
                .and_then(|input| input.squeeze_dim(0))
                .map_err(|error| tensor_error(&error))?;
            return PyTensor::new(output).into_py_any(py);
        }
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

    Err(PyNotImplementedError::new_err(format!(
        "torch_rs.nn.functional.{} does not support sampling",
        metadata.public_function
    )))
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
