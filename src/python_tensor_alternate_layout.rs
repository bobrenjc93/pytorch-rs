//! Python alternate-layout metadata for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::python::{
    PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode, dispatch_tensorbase_no_argument_mode,
};

#[pymethods]
impl PyTensorBase {
    #[getter]
    fn is_mkldnn(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_mkldnn")? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .is_mkldnn()
            .into_py_any(slf.py())
    }

    #[getter]
    fn is_nested(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_nested")? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .is_nested()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor uses sparse COO storage layout, ``False`` otherwise.\n"]
    #[getter]
    fn is_sparse(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().is_sparse())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor uses sparse CSR storage layout, ``False`` otherwise.\n"]
    #[getter]
    fn is_sparse_csr(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().is_sparse_csr())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndense_dim() -> int\n\nReturn the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.\n\n.. note::\n  Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.\n\nSee also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn dense_dim(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "dense_dim")? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .dense_dim()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsparse_dim() -> int\n\nReturn the number of sparse dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.\n\n.. note::\n  Returns ``0`` if :attr:`self` is not a sparse tensor.\n\nSee also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn sparse_dim(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "sparse_dim")?
        {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .sparse_dim()
            .into_py_any(slf.py())
    }
}
