//! Python shape descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{
    python::{
        PyTensor, PyTensorBase, bind_size_dimension, dispatch_tensorbase_getset_mode,
        dispatch_tensorbase_method_mode, dispatch_tensorbase_no_argument_mode,
        extract_dimension_swap_dimension, normalize_dimension,
    },
    python_size::construct_size,
};

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nshape() -> torch.Size\n\nReturns the size of the :attr:`self` tensor. Alias for :attr:`size`.\n\nSee also :meth:`Tensor.size`.\n\nExample::\n\n    >>> t = torch.empty(3, 4, 5)\n    >>> t.size()\n    torch.Size([3, 4, 5])\n    >>> t.shape\n    torch.Size([3, 4, 5])\n\n"]
    #[getter]
    fn shape(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "shape")? {
            return Ok(result);
        }

        let dimensions = {
            let tensor = tensor.try_borrow()?;
            PyTuple::new(slf.py(), tensor.inner().shape().iter().copied())?
        };
        construct_size(slf.py(), dimensions.as_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndim() -> int\n\nReturns the number of dimensions of :attr:`self` tensor.\n"]
    #[pyo3(text_signature = None)]
    fn dim(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "dim")? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .shape()
            .len()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsize(dim=None) -> torch.Size or int\n\nReturns the size of the :attr:`self` tensor. If ``dim`` is not specified,\nthe returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.\nIf ``dim`` is specified, returns an int holding the size of that dimension.\n\nArgs:\n  dim (int, optional): The dimension for which to retrieve the size.\n\nExample::\n\n    >>> t = torch.empty(3, 4, 5)\n    >>> t.size()\n    torch.Size([3, 4, 5])\n    >>> t.size(dim=1)\n    4\n\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn size(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let dimension = bind_size_dimension(args, kwargs)?;
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "size",
            "torch.Tensor.size",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        let Some(dimension) = dimension else {
            let dimensions = {
                let tensor = tensor.try_borrow()?;
                PyTuple::new(slf.py(), tensor.inner().shape().iter().copied())?
            };
            return construct_size(slf.py(), dimensions.as_any());
        };

        let dimension = extract_dimension_swap_dimension(&dimension.value)?;
        let tensor = tensor.try_borrow()?;
        let axis = normalize_dimension(dimension, tensor.inner().shape().len())?;
        tensor.inner().shape()[axis].into_py_any(slf.py())
    }
}

#[pymethods]
impl PyTensor {
    /// Alias for [`Tensor.dim()`](https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html).
    #[getter]
    fn ndim(&self) -> usize {
        self.inner().shape().len()
    }

    /// Alias for [`Tensor.dim()`](https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html).
    #[pyo3(text_signature = None)]
    fn ndimension(&self) -> usize {
        self.inner().shape().len()
    }

    /// Alias for [`Tensor.numel()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html).
    #[pyo3(text_signature = None)]
    fn nelement(&self) -> usize {
        self.inner().numel()
    }

    /// Returns the total number of elements in the tensor.
    #[pyo3(text_signature = None)]
    fn numel(&self) -> usize {
        self.inner().numel()
    }
}
