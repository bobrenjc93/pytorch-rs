//! Python bindings for native scalar types.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use crate::DType;

static FLOAT32: PyOnceLock<Py<PyDType>> = PyOnceLock::new();

/// Python scalar-type descriptor backed by a native [`DType`].
#[pyclass(name = "dtype", module = "torch_rs", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDType {
    inner: DType,
}

impl PyDType {
    pub(crate) const fn inner(&self) -> DType {
        self.inner
    }
}

#[pymethods]
impl PyDType {
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.element_size()
    }

    #[getter]
    fn is_floating_point(&self) -> bool {
        self.inner.is_floating_point()
    }

    #[getter]
    fn is_complex(&self) -> bool {
        self.inner.is_complex()
    }

    #[getter]
    fn is_signed(&self) -> bool {
        self.inner.is_signed()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            DType::Float32 => "torch.float32",
        }
    }

    fn __str__(&self) -> &'static str {
        self.__repr__()
    }

    fn __reduce__(&self) -> &'static str {
        match self.inner {
            DType::Float32 => "float32",
        }
    }
}

/// Returns the canonical Python descriptor for a native scalar type.
pub(crate) fn dtype_object(py: Python<'_>, dtype: DType) -> PyResult<&'static Py<PyDType>> {
    match dtype {
        DType::Float32 => FLOAT32.get_or_try_init(py, || Py::new(py, PyDType { inner: dtype })),
    }
}
