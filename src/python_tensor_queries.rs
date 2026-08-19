//! Top-level Python bindings for tensor metadata and scalar truth queries.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::python::{PyTensor, bind_legacy_single_tensor_argument, python_type_name};

/// Returns the total number of elements in the input tensor.
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn numel(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<usize> {
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "numel() takes 1 positional argument but {} were given",
            args.len()
        )));
    }

    let keyword_input = match kwargs {
        Some(values) => values.get_item("input")?,
        None => None,
    };
    if args.is_empty() && keyword_input.is_none() {
        return Err(PyTypeError::new_err(
            "numel() missing 1 required positional arguments: \"input\"",
        ));
    }

    let (input, position) = if args.is_empty() {
        (
            keyword_input
                .as_ref()
                .expect("the required keyword input was checked above"),
            None,
        )
    } else {
        (&args.get_item(0)?, Some(1))
    };
    let Ok(tensor) = input.cast::<PyTensor>() else {
        let position =
            position.map_or_else(String::new, |position| format!(" (position {position})"));
        let input_type = python_type_name(input)?;
        return Err(PyTypeError::new_err(format!(
            "numel(): argument 'input'{position} must be Tensor, not {input_type}"
        )));
    };

    if !args.is_empty() && keyword_input.is_some() {
        return Err(PyTypeError::new_err(
            "numel() got multiple values for argument 'input'",
        ));
    }
    if let Some(kwargs) = kwargs {
        for key in kwargs.keys() {
            let key = key.extract::<String>()?;
            if key != "input" {
                return Err(PyTypeError::new_err(format!(
                    "numel() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }

    Ok(tensor.try_borrow()?.inner().numel())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_nonzero(input) -> (bool)\n\nReturns True if the :attr:`input` is a single element tensor which is not equal to zero\nafter type conversions.\ni.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor([0])`` or\n``torch.tensor([False])``.\nThrows a ``RuntimeError`` if ``torch.numel() != 1`` (even in case\nof sparse tensors).\n\nArgs:\n    input (Tensor): the input tensor.\n\nExamples::\n\n    >>> torch.is_nonzero(torch.tensor([0.]))\n    False\n    >>> torch.is_nonzero(torch.tensor([1.5]))\n    True\n    >>> torch.is_nonzero(torch.tensor([False]))\n    False\n    >>> torch.is_nonzero(torch.tensor([3]))\n    True\n    >>> torch.is_nonzero(torch.tensor([1, 3, 5]))\n    Traceback (most recent call last):\n    ...\n    RuntimeError: Boolean value of Tensor with more than one value is ambiguous\n    >>> torch.is_nonzero(torch.tensor([]))\n    Traceback (most recent call last):\n    ...\n    RuntimeError: Boolean value of Tensor with no values is ambiguous\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_nonzero(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_nonzero", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_nonzero input type was checked while binding");
    tensor.try_borrow()?.truth_value()
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_complex(input: Tensor) -> bool\n\nReturns True if the data type of :attr:`input` is a complex data type i.e.,\none of ``torch.complex64``, and ``torch.complex128``.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex64))\n    True\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex128))\n    True\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.int32))\n    False\n    >>> torch.is_complex(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n    False\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_complex(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_complex", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_complex input type was checked while binding");
    Ok(tensor.try_borrow()?.inner().is_complex())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_floating_point(input: Tensor) -> bool\n\nReturns True if the data type of :attr:`input` is a floating point data type i.e.,\none of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0]))\n    True\n    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.int32))\n    False\n    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n    True\n    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.complex64))\n    False\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_floating_point(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_floating_point", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_floating_point input type was checked while binding");
    Ok(tensor.try_borrow()?.inner().is_floating_point())
}

#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_signed(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_signed", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_signed input type was checked while binding");
    Ok(tensor.try_borrow()?.inner().is_signed())
}

pub(crate) fn add_tensor_queries(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(numel, module)?)?;
    module.add_function(wrap_pyfunction!(is_nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(is_complex, module)?)?;
    module.add_function(wrap_pyfunction!(is_floating_point, module)?)?;
    module.add_function(wrap_pyfunction!(is_signed, module)?)?;
    Ok(())
}
