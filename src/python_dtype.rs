//! Python bindings for native scalar types.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyModule};

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
    fn abbr(&self) -> &'static str {
        self.inner.abbr()
    }

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

    #[pyo3(text_signature = None)]
    fn to_real(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        Ok(dtype_object(py, self.inner.to_real())?.clone_ref(py))
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

#[pyfunction(signature = (d, /))]
fn _set_default_dtype(d: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(dtype) = d.cast::<PyDType>()
        && dtype.try_borrow()?.inner() == DType::Float32
    {
        return Ok(());
    }

    Err(PyTypeError::new_err(
        "invalid dtype object: only floating-point types are supported as the default type",
    ))
}

pub(crate) fn add_default_dtype_validator(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_set_default_dtype, module)?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_set_default_dtype",))?;
    Ok(())
}
