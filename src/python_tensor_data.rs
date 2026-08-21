//! Python detached-data descriptor for native tensors.

use pyo3::prelude::*;

use crate::{
    python::{PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode},
    python_tensor_errors::tensor_error,
};

#[pymethods]
impl PyTensorBase {
    #[getter]
    fn data(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "data")? {
            return Ok(result);
        }

        let inner = tensor
            .try_borrow()?
            .inner()
            .detach()
            .map_err(|error| tensor_error(&error))?;
        Ok(Py::new(slf.py(), PyTensor::new(inner))?.into_any())
    }
}
