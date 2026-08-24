//! Python leaf-gradient descriptors for native tensors.

use pyo3::{
    exceptions::{PyAttributeError, PyNotImplementedError, PyRuntimeError, PyTypeError},
    prelude::*,
};

use crate::{
    python::{PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode, python_type_name},
    python_dtype::dtype_object,
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

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nThis attribute is ``None`` by default and becomes a Tensor the first time a call to\n:func:`backward` computes gradients for ``self``.\nThe attribute will then contain the gradients computed and future calls to\n:func:`backward` will accumulate (add) gradients into it.\n"]
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

    #[setter]
    fn set_grad(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if !value.is_none() {
            if value.cast::<Self>().is_ok() {
                return Err(PyNotImplementedError::new_err(
                    "torch_rs only supports assigning None to Tensor.grad",
                ));
            }
            let type_name = python_type_name(value)?;
            return Err(PyTypeError::new_err(format!(
                "assigned grad expected to be a Tensor or None but got grad of type {type_name}"
            )));
        }
        if !self.inner().clear_leaf_grad() {
            return Err(PyRuntimeError::new_err(
                "grad can only be cleared on leaf tensors",
            ));
        }
        self.clear_grad_cache();
        Ok(())
    }

    #[deleter]
    fn delete_grad(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(PyAttributeError::new_err(
            "Tensor.grad cannot be deleted; assign None to clear it",
        ))
    }
}
