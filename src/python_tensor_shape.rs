//! Python shape descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};

use crate::{
    DType, Device, TensorError,
    python::{
        PyTensor, PyTensorBase, bind_size_dimension, dispatch_tensorbase_getset_mode,
        dispatch_tensorbase_method_mode, dispatch_tensorbase_no_argument_mode,
        extract_dimension_swap_dimension, normalize_dimension,
    },
    python_size::construct_size,
    python_tensor_errors::tensor_error,
};

pub(crate) fn validate_unflatten_storage(tensor: &PyTensor) -> PyResult<()> {
    if tensor.inner().dtype() != DType::Float32 || tensor.inner().device() != Device::Cpu {
        return Err(PyNotImplementedError::new_err(
            "unflatten(): only exact native CPU float32 Tensor inputs are supported",
        ));
    }
    Ok(())
}

// Both Python bindings call this after their own schema and scope checks.
// Already-parsed integers reach the retained view/autograd implementation
// without converting them again or looking up a Python-owned method.
pub(crate) fn unflatten_native(tensor: &PyTensor, dim: i64, sizes: &[i64]) -> PyResult<PyTensor> {
    tensor
        .inner()
        .unflatten(dim, sizes)
        .map(PyTensor::new)
        .map_err(|error| match error {
            TensorError::UnflattenScalar { .. }
            | TensorError::ReshapeMultipleInferredDimensions
            | TensorError::ReshapeInvalidDimension { .. }
            | TensorError::ReshapeAmbiguousZeroElements { .. } => {
                PyRuntimeError::new_err(format!("unflatten got an unexpected error:\n{error}"))
            }
            _ => tensor_error(&error),
        })
}

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
    // The Python-owned public method supplies CPython argument binding and the
    // empty-sizes check before entering this native implementation.
    fn _unflatten(
        slf: &Bound<'_, Self>,
        dim: &Bound<'_, PyAny>,
        sizes: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let tensor = slf.try_borrow()?;
        if !slf.as_any().is_exact_instance_of::<Self>() {
            return Err(PyNotImplementedError::new_err(
                "unflatten(): only exact native CPU float32 Tensor inputs are supported",
            ));
        }
        validate_unflatten_storage(&tensor)?;
        if dim.is_instance_of::<PyBool>() || !dim.hasattr("__index__")? {
            return Err(PyTypeError::new_err(format!(
                "unflatten(): argument 'dim' (position 1) must be int, not {}",
                dim.get_type().name()?
            )));
        }
        let dim = extract_dimension_swap_dimension(dim)?;
        if !sizes.is_instance_of::<PyTuple>() && !sizes.is_instance_of::<PyList>() {
            return Err(PyTypeError::new_err(format!(
                "unflatten(): argument 'sizes' (position 2) must be tuple of ints, not {}",
                sizes.get_type().name()?
            )));
        }
        let mut parsed = Vec::new();
        for (index, size) in sizes.try_iter()?.enumerate() {
            let size = size?;
            if size.is_instance_of::<PyBool>() || !size.hasattr("__index__")? {
                return Err(PyTypeError::new_err(format!(
                    "unflatten(): argument 'sizes' (position 2) must be tuple of ints, but found element of type {} at pos {index}",
                    size.get_type().name()?
                )));
            }
            let size = extract_dimension_swap_dimension(&size).map_err(|error| {
                PyTypeError::new_err(format!(
                    "unflatten(): argument 'sizes' failed to unpack the object at pos {} with error \"{}\"",
                    index + 1,
                    error.value(slf.py())
                ))
            })?;
            parsed.push(size);
        }
        unflatten_native(&tensor, dim, &parsed)
    }

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
