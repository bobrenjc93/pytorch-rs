//! Python lazy-bit bindings for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::python::{PyTensor, PyTensorBase, dispatch_tensorbase_no_argument_mode};

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_conj() -> bool\n\nReturns True if the conjugate bit of :attr:`self` is set to true.\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn is_conj(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "is_conj")? {
            return Ok(result);
        }

        // Complex storage and conjugate views are unsupported. Every current
        // Tensor therefore has a clear conjugate bit, which can be reported
        // without borrowing storage or touching its autograd graph.
        false.into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nresolve_conj() -> Tensor\n\nSee :func:`torch.resolve_conj`\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn resolve_conj(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "resolve_conj")?
        {
            return Ok(result);
        }

        // Complex storage and conjugate views are unsupported, so is_conj()
        // is false for every reachable Tensor. Resolving that clear bit is the
        // exact receiver and requires no storage borrow, metadata rewrite, or
        // autograd operation.
        Ok(tensor.clone().unbind().into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_neg() -> bool\n\nReturns True if the negative bit of :attr:`self` is set to true.\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn is_neg(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "is_neg")? {
            return Ok(result);
        }

        // Lazy negative views are unsupported, and eager negation does not set
        // the negative bit. Every reachable Tensor can therefore report a
        // clear bit without borrowing storage or touching its autograd graph.
        false.into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nresolve_neg() -> Tensor\n\nSee :func:`torch.resolve_neg`\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn resolve_neg(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "resolve_neg")?
        {
            return Ok(result);
        }

        // Lazy negative views are unsupported, so is_neg() is false for every
        // reachable Tensor. Resolving that clear bit is the exact receiver and
        // requires no storage borrow, metadata rewrite, or autograd operation.
        Ok(tensor.clone().unbind().into_any())
    }
}
