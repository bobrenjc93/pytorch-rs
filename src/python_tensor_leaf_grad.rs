//! Python leaf-gradient descriptors for native tensors.

use pyo3::prelude::*;

use crate::{python::PyTensor, python_tensor_errors::tensor_error};

#[pymethods]
impl PyTensor {
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner().requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner().is_leaf()
    }

    #[getter]
    fn grad(&self, py: Python<'_>) -> PyResult<Option<Py<Self>>> {
        if let Some(gradient) = self.grad_cache().get(py) {
            return Ok(Some(gradient.clone_ref(py)));
        }
        let Some(inner) = self
            .inner()
            .live_grad()
            .map_err(|error| tensor_error(&error))?
        else {
            return Ok(None);
        };
        let gradient = self
            .grad_cache()
            .get_or_try_init(py, || Py::new(py, Self::new(inner)))?;
        Ok(Some(gradient.clone_ref(py)))
    }
}
