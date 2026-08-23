//! Python dtype descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString, PyTuple};

use crate::{
    DType,
    python::{PyTensor, PyTensorBase, dispatch_tensorbase_method_mode, python_type_name},
    python_dtype::{PyDType, dtype_object},
};

struct TypeCallArgument<'py> {
    value: Bound<'py, PyAny>,
    position: Option<usize>,
}

fn bind_type_arguments<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if args.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "type() takes from 0 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut dtype = if args.is_empty() {
        None
    } else {
        Some(args.get_item(0)?)
    };
    let mut non_blocking = if args.len() < 2 {
        None
    } else {
        Some(TypeCallArgument {
            value: args.get_item(1)?,
            position: Some(2),
        })
    };
    let mut keyword_error = None;

    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "dtype" => {
                    if dtype.is_some() {
                        keyword_error.get_or_insert_with(|| {
                            PyTypeError::new_err("type() got multiple values for argument 'dtype'")
                        });
                    } else {
                        dtype = Some(value);
                    }
                }
                "non_blocking" => {
                    if non_blocking.is_some() {
                        keyword_error.get_or_insert_with(|| {
                            PyTypeError::new_err(
                                "type() got multiple values for argument 'non_blocking'",
                            )
                        });
                    } else {
                        non_blocking = Some(TypeCallArgument {
                            value,
                            position: None,
                        });
                    }
                }
                _ => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "type() got an unexpected keyword argument '{key}'"
                        ))
                    });
                }
            }
        }
    }

    // PyTorch validates this generated-parser argument before invoking an
    // active TorchFunctionMode. Requiring the exact bool type avoids silently
    // accepting integers or arbitrary truthy objects.
    if let Some(non_blocking) = non_blocking
        && !non_blocking.value.is_exact_instance_of::<PyBool>()
    {
        let actual = python_type_name(&non_blocking.value)?;
        let position = non_blocking
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        return Err(PyTypeError::new_err(format!(
            "type(): argument 'non_blocking'{position} must be bool, not {actual}"
        )));
    }

    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    Ok(dtype)
}

fn validate_identity_type_target(dtype: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(dtype) = dtype.cast::<PyDType>()
        && dtype.try_borrow()?.inner() == DType::Float32
    {
        return Ok(());
    }
    if let Ok(name) = dtype.cast::<PyString>()
        && name.to_str()? == "torch.FloatTensor"
    {
        return Ok(());
    }

    Err(PyTypeError::new_err(
        "type(): only torch.float32 and 'torch.FloatTensor' are supported",
    ))
}

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ntype(dtype=None, non_blocking=False, **kwargs) -> str or Tensor\nReturns the type if `dtype` is not provided, else casts this object to\nthe specified type.\n\nIf this is already of the correct type, no copy is performed and the\noriginal object is returned.\n\nArgs:\n    dtype (dtype or string): The desired type\n    non_blocking (bool): If ``True``, and the source is in pinned memory\n        and destination is on the GPU or vice versa, the copy is performed\n        asynchronously with respect to the host. Otherwise, the argument\n        has no effect.\n    **kwargs: For compatibility, may contain the key ``async`` in place of\n        the ``non_blocking`` argument. The ``async`` arg is deprecated.\n"]
    // Keep PyTorch's variadic descriptor shape. A fixed PyO3 signature would
    // expose generated inspection metadata unlike PyTorch's native method.
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn r#type(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let dtype = bind_type_arguments(args, kwargs)?;

        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "type",
            "torch.Tensor.type",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        if let Some(dtype) = dtype.as_ref()
            && !dtype.is_none()
        {
            validate_identity_type_target(dtype)?;
            // Float32 on CPU is the only supported dtype/device pair, so every
            // accepted conversion is the exact receiver. This avoids borrowing
            // or changing storage, layout metadata, or autograd state.
            return Ok(tensor.clone().unbind().into_any());
        }

        // An omitted or explicit-None dtype remains the legacy type-name query.
        "torch.FloatTensor".into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nAlias for :meth:`~Tensor.element_size()`\n"]
    #[getter]
    fn itemsize(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().dtype().element_size())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nelement_size() -> int\n\nReturns the size in bytes of an individual element.\n\nExample::\n\n    >>> torch.tensor([]).element_size()\n    4\n    >>> torch.tensor([], dtype=torch.uint8).element_size()\n    1\n\n"]
    #[pyo3(text_signature = None)]
    fn element_size(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().element_size())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_complex() -> bool\n\nReturns True if the data type of :attr:`self` is a complex data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_complex(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().is_complex())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_signed() -> bool\n\nReturns True if the data type of :attr:`self` is a signed data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_signed(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().is_signed())
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        Ok(dtype_object(py, self.inner().dtype())?.clone_ref(py))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_floating_point() -> bool\n\nReturns True if the data type of :attr:`self` is a floating point data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_floating_point(&self) -> bool {
        self.inner().is_floating_point()
    }
}
