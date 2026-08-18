//! Native bridges for Python neural-network functional operators.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::{
    python::PyTensor,
    python_argument_schema::{ArgumentSchema, parse_float_like_argument},
};

const DROPOUT_INPUT: ArgumentSchema = ArgumentSchema::new("dropout", "input", 1, "Tensor");
const DROPOUT_PROBABILITY: ArgumentSchema = ArgumentSchema::new("dropout", "p", 2, "float");
const DROPOUT_TRAINING: ArgumentSchema = ArgumentSchema::new("dropout", "train", 3, "bool");
const DROPOUT_INPLACE_INPUT: ArgumentSchema = ArgumentSchema::new("dropout_", "input", 1, "Tensor");
const DROPOUT_INPLACE_PROBABILITY: ArgumentSchema =
    ArgumentSchema::new("dropout_", "p", 2, "float");
const DROPOUT_INPLACE_TRAINING: ArgumentSchema =
    ArgumentSchema::new("dropout_", "train", 3, "bool");
const ALPHA_DROPOUT_INPUT: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout", "input", 1, "Tensor");
const ALPHA_DROPOUT_PROBABILITY: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout", "p", 2, "float");
const ALPHA_DROPOUT_TRAINING: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout", "train", 3, "bool");
const ALPHA_DROPOUT_INPLACE_INPUT: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout_", "input", 1, "Tensor");
const ALPHA_DROPOUT_INPLACE_PROBABILITY: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout_", "p", 2, "float");
const ALPHA_DROPOUT_INPLACE_TRAINING: ArgumentSchema =
    ArgumentSchema::new("alpha_dropout_", "train", 3, "bool");

struct DropoutSchema {
    input: ArgumentSchema,
    probability: ArgumentSchema,
    training: ArgumentSchema,
}

impl DropoutSchema {
    const fn dropout(inplace: bool) -> Self {
        if inplace {
            Self {
                input: DROPOUT_INPLACE_INPUT,
                probability: DROPOUT_INPLACE_PROBABILITY,
                training: DROPOUT_INPLACE_TRAINING,
            }
        } else {
            Self {
                input: DROPOUT_INPUT,
                probability: DROPOUT_PROBABILITY,
                training: DROPOUT_TRAINING,
            }
        }
    }

    const fn alpha_dropout(inplace: bool) -> Self {
        if inplace {
            Self {
                input: ALPHA_DROPOUT_INPLACE_INPUT,
                probability: ALPHA_DROPOUT_INPLACE_PROBABILITY,
                training: ALPHA_DROPOUT_INPLACE_TRAINING,
            }
        } else {
            Self {
                input: ALPHA_DROPOUT_INPUT,
                probability: ALPHA_DROPOUT_PROBABILITY,
                training: ALPHA_DROPOUT_TRAINING,
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DropoutKind {
    Standard,
    Alpha,
}

impl DropoutKind {
    const fn schema(self, inplace: bool) -> DropoutSchema {
        match self {
            Self::Standard => DropoutSchema::dropout(inplace),
            Self::Alpha => DropoutSchema::alpha_dropout(inplace),
        }
    }

    const fn supports_tensor_probability(self) -> bool {
        match self {
            Self::Standard => true,
            Self::Alpha => false,
        }
    }

    const fn unsupported_sampling_error(self) -> &'static str {
        match self {
            Self::Standard => "torch_rs.nn.functional.dropout does not support sampling",
            Self::Alpha => "torch_rs.nn.functional.alpha_dropout does not support sampling",
        }
    }
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

fn nn_functional_dropout_identity(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    probability: &Bound<'_, PyAny>,
    training: &Bound<'_, PyAny>,
    inplace: bool,
    kind: DropoutKind,
) -> PyResult<Py<PyAny>> {
    // This private bridge mirrors the native operator's schema checks for the
    // identity cases only. It deliberately owns no random state or mutation.
    let schema = kind.schema(inplace);
    let Ok(tensor) = input.cast::<PyTensor>() else {
        return Err(schema.input.type_error(input)?);
    };
    if !kind.supports_tensor_probability() && probability.cast::<PyTensor>().is_ok() {
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

    let input_is_empty = tensor.try_borrow()?.inner().numel() == 0;
    if !training || probability == 0.0 || input_is_empty {
        return Ok(tensor.clone().unbind().into_any());
    }

    Err(PyNotImplementedError::new_err(
        kind.unsupported_sampling_error(),
    ))
}

#[pyfunction]
fn _nn_functional_dropout(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    probability: &Bound<'_, PyAny>,
    training: &Bound<'_, PyAny>,
    inplace: bool,
) -> PyResult<Py<PyAny>> {
    nn_functional_dropout_identity(
        py,
        input,
        probability,
        training,
        inplace,
        DropoutKind::Standard,
    )
}

#[pyfunction]
fn _nn_functional_alpha_dropout(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    probability: &Bound<'_, PyAny>,
    training: &Bound<'_, PyAny>,
    inplace: bool,
) -> PyResult<Py<PyAny>> {
    nn_functional_dropout_identity(
        py,
        input,
        probability,
        training,
        inplace,
        DropoutKind::Alpha,
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
        wrap_pyfunction!(_nn_functional_alpha_dropout, module)?,
        wrap_pyfunction!(_nn_functional_dropout_tensor_autograd_suffix, module)?,
    ] {
        let name = function.getattr("__name__")?;
        module.add_function(function.clone())?;
        module.getattr("__all__")?.call_method1("remove", (name,))?;
    }
    Ok(())
}
