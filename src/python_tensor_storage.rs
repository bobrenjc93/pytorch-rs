//! Python storage-pointer descriptors for native tensors.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::python::{PyTensor, PyTensorBase, dispatch_tensorbase_no_argument_mode};

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nconst_data_ptr() -> int\n\nReturns the address of the first element of :attr:`self` tensor.\n\nUnlike :meth:`data_ptr`, this is guaranteed to be a read-only access\nthat will not trigger copy-on-write materialization. For regular\n(non-COW) tensors, the return value is identical to :meth:`data_ptr`.\n\n.. warning::\n\n    The returned pointer must not be used to mutate the tensor data.\n    Use :meth:`data_ptr` when write access is needed.\n"]
    #[pyo3(text_signature = None)]
    fn const_data_ptr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "const_data_ptr")?
        {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner()
            .const_data_ptr()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndata_ptr() -> int\n\nReturns the address of the first element of :attr:`self` tensor.\n\n.. note::\n\n    If the tensor is a copy-on-write tensor (e.g. created via\n    :meth:`_lazy_clone`), calling this method will materialize the\n    copy. Use :meth:`const_data_ptr` if you only need read-only access\n    to the data pointer.\n"]
    #[pyo3(text_signature = None)]
    fn data_ptr(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner().data_ptr())
    }
}
