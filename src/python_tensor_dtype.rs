//! Python dtype descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{
    python::{PyTensor, PyTensorBase, dispatch_tensorbase_method_mode},
    python_dtype::{PyDType, dtype_object},
};

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ntype(dtype=None, non_blocking=False, **kwargs) -> str or Tensor\nReturns the type if `dtype` is not provided, else casts this object to\nthe specified type.\n\nIf this is already of the correct type, no copy is performed and the\noriginal object is returned.\n\nArgs:\n    dtype (dtype or string): The desired type\n    non_blocking (bool): If ``True``, and the source is in pinned memory\n        and destination is on the GPU or vice versa, the copy is performed\n        asynchronously with respect to the host. Otherwise, the argument\n        has no effect.\n    **kwargs: For compatibility, may contain the key ``async`` in place of\n        the ``non_blocking`` argument. The ``async`` arg is deprecated.\n"]
    // Keep PyTorch's variadic descriptor shape even though the supported
    // surface is deliberately limited to the no-argument query. A METH_NOARGS
    // descriptor would gain a synthesized signature on CPython 3.13+.
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn r#type(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "type() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }
        if let Some(kwargs) = kwargs
            && let Some((key, _)) = kwargs.iter().next()
        {
            let key = key.extract::<String>()?;
            return Err(PyTypeError::new_err(format!(
                "type() got an unexpected keyword argument '{key}'"
            )));
        }

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

        // Float32 on CPU is the only supported dtype/device pair. Report its
        // canonical legacy tensor type without borrowing or changing storage,
        // metadata, or autograd state. Conversion overloads remain unsupported.
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
