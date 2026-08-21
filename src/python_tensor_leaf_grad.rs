//! Python leaf-gradient descriptors for native tensors.

use pyo3::{
    exceptions::{PyAttributeError, PyNotImplementedError, PyRuntimeError},
    prelude::*,
};

use crate::{
    DType,
    python::{
        PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode, dispatch_tensorbase_set_mode,
    },
    python_dtype::{PyDType, dtype_object},
    python_tensor_errors::tensor_error,
};

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nThe allowed dtype of :attr:``grad`` for this tensor.\n\n:attr:``grad_dtype`` can be set to a specific dtype or ``None``. By default,\n``t.grad_dtype == t.dtype``. When not None, the autograd engine casts\nincoming gradients to this dtype. This attribute is only accessible and\nsettable for leaf tensors.\n\n.. warning::\n    Use with caution. Diverging the dtypes of a tensor and its gradient may\n    break downstream systems that assume they match.\n\nExample::\n\n    >>> x = torch.tensor([1.0, 2.0], requires_grad=True)\n    >>> x.grad_dtype\n    torch.float32\n\n    >>> x.grad_dtype = torch.float16\n    >>> x.grad_dtype\n    torch.float16\n\n    >>> # Allow any gradient dtype\n    >>> x.grad_dtype = None\n    >>> x.grad_dtype\n"]
    #[getter]
    fn grad_dtype(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "grad_dtype")? {
            return Ok(result);
        }

        let dtype = {
            let tensor = tensor.try_borrow()?;
            if !tensor.inner().is_leaf() {
                return Err(PyRuntimeError::new_err(
                    "grad_dtype can only be accessed on leaf tensors.",
                ));
            }
            tensor.inner().dtype()
        };
        Ok(dtype_object(slf.py(), dtype)?
            .clone_ref(slf.py())
            .into_any())
    }

    #[setter(grad_dtype)]
    fn set_grad_dtype(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if dispatch_tensorbase_set_mode(slf.py(), tensor, "grad_dtype", value)?.is_some() {
            return Ok(());
        }

        let requested_dtype = if value.is_none() {
            None
        } else if let Ok(dtype) = value.cast::<PyDType>() {
            Some(dtype.try_borrow()?.inner())
        } else {
            let type_name = value.get_type().name()?;
            return Err(PyRuntimeError::new_err(format!(
                "grad_dtype must be a torch.dtype or None, but got {type_name}"
            )));
        };

        if !tensor.try_borrow()?.inner().is_leaf() {
            return Err(PyRuntimeError::new_err(
                "grad_dtype can only be set on leaf tensors.",
            ));
        }

        if requested_dtype != Some(DType::Float32) {
            return Err(PyNotImplementedError::new_err(
                "torch_rs only supports setting grad_dtype to torch.float32",
            ));
        }

        // Float32 is already the native accumulator dtype. Accepting this
        // assignment is therefore a metadata no-op and does not require a
        // second dtype field that later configurable-dtype work must unwind.
        Ok(())
    }

    #[deleter(grad_dtype)]
    fn delete_grad_dtype(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(PyAttributeError::new_err(
            "attribute 'grad_dtype' of 'torch._C.TensorBase' objects is not writable",
        ))
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner().requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner().is_leaf()
    }

    #[getter]
    fn grad(&self, py: Python<'_>) -> PyResult<Option<Py<Self>>> {
        if let Some(gradient) = self.grad_cache().get(py) {
            return Ok(Some(gradient.clone_ref(py)));
        }
        let Some(inner) = self
            .inner()
            .live_grad()
            .map_err(|error| tensor_error(&error))?
        else {
            return Ok(None);
        };
        let gradient = self
            .grad_cache()
            .get_or_try_init(py, || Py::new(py, Self::new(inner)))?;
        Ok(Some(gradient.clone_ref(py)))
    }
}
