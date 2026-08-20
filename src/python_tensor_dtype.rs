//! Python dtype descriptors for native tensors.

use pyo3::prelude::*;

use crate::{
    python::{PyTensor, PyTensorBase},
    python_dtype::{PyDType, dtype_object},
};

#[pymethods]
impl PyTensorBase {
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
