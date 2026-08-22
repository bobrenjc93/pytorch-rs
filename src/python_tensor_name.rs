//! Python tensor-name descriptor for unnamed native tensors.

use pyo3::prelude::*;

use crate::python::{PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode};

#[pymethods]
impl PyTensorBase {
    // PyTorch 2.13 exposes this descriptor without a docstring.
    #[getter]
    fn name(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "name")? {
            return Ok(result);
        }

        // Named dimensions are unsupported, so every reachable tensor has no name.
        Ok(slf.py().None())
    }
}
