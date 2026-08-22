//! Python device descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::python::{
    PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode, dispatch_tensorbase_no_argument_mode,
};

#[pymethods]
impl PyTensorBase {
    // Keep the method as METH_NOARGS with no docstring or embedded signature.
    // PyTorch exposes the same native descriptor metadata and CPython 3.13+
    // derives `($self, /)` from that descriptor shape.
    #[pyo3(text_signature = None)]
    fn is_distributed(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "is_distributed")?
        {
            return Ok(result);
        }

        // Distributed tensor wrappers are unsupported. Every reachable Tensor
        // is therefore local, independent of its shape, view, or autograd state.
        false.into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the CPU, ``False`` otherwise.\n"]
    #[getter]
    fn is_cpu(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_cpu")? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner().is_cpu().into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the GPU, ``False`` otherwise.\n"]
    #[getter]
    fn is_cuda(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_cuda")? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner().is_cuda().into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the XPU, ``False`` otherwise.\n"]
    #[getter]
    fn is_xpu(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_xpu")? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner().is_xpu().into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on an XLA device, ``False`` otherwise.\n"]
    #[getter]
    fn is_xla(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_xla")? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner().is_xla().into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the MPS device, ``False`` otherwise.\n"]
    #[getter]
    fn is_mps(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_mps")? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner().is_mps().into_py_any(slf.py())
    }

    // PyTorch 2.13 exposes this descriptor without a docstring.
    #[getter]
    fn is_vulkan(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "is_vulkan")? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .is_vulkan()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is a meta tensor, ``False`` otherwise.  Meta tensors\nare like normal tensors, but they carry no data.\n"]
    #[getter]
    fn is_meta(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().is_meta())
    }
}
