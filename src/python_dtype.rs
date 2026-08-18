//! Python bindings for native scalar types.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use crate::{
    DType,
    python::{CallKeywordOrder, call_type_summary, python_argument_type_name},
};

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

#[pyclass(name = "finfo", module = "torch_rs", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PyFInfo {
    inner: DType,
}

fn parse_finfo_dtype(value: &Bound<'_, PyAny>, position: Option<usize>) -> PyResult<DType> {
    if let Ok(dtype) = value.cast::<PyDType>() {
        let dtype = dtype.try_borrow()?.inner();
        if dtype.is_floating_point() {
            return Ok(dtype);
        }
        return Err(PyTypeError::new_err(
            "torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'",
        ));
    }

    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_argument_type_name(value)?;
    Err(PyTypeError::new_err(format!(
        "finfo(): argument 'type'{position} must be torch.dtype, not {actual}"
    )))
}

fn invalid_finfo_arguments(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let actual = call_type_summary(args, kwargs, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "finfo() received an invalid combination of arguments - got ({actual}), but expected one of:\n * (torch.dtype type)\n * ()\n"
    )))
}

fn bind_finfo_dtype(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<DType> {
    let keyword_count = kwargs.map_or(0, pyo3::types::PyDictMethods::len);
    match (args.len(), keyword_count) {
        (0, 0) => Ok(DType::default()),
        (0, 1) => {
            let kwargs = kwargs.expect("one keyword requires a keyword dictionary");
            if let Some(dtype) = kwargs.get_item("type")? {
                parse_finfo_dtype(&dtype, None)
            } else {
                Err(PyTypeError::new_err(
                    "finfo() missing 1 required positional arguments: \"type\"",
                ))
            }
        }
        (1, 0) => parse_finfo_dtype(&args.get_item(0)?, Some(1)),
        _ => Err(invalid_finfo_arguments(args, kwargs)?),
    }
}

fn format_finfo_value(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust scientific formatting always contains an exponent");
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    let exponent = exponent
        .parse::<i32>()
        .expect("Rust scientific formatting always uses an integer exponent");
    format!("{mantissa}e{exponent:+03}")
}

#[pymethods]
impl PyFInfo {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: bind_finfo_dtype(args, kwargs)?,
        })
    }

    #[getter]
    fn bits(&self) -> usize {
        self.inner.floating_point_info().bits()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.floating_point_info().eps()
    }

    #[getter]
    fn max(&self) -> f64 {
        self.inner.floating_point_info().max()
    }

    #[getter]
    fn min(&self) -> f64 {
        self.inner.floating_point_info().min()
    }

    #[getter]
    fn resolution(&self) -> f64 {
        self.inner.floating_point_info().resolution()
    }

    #[getter]
    fn smallest_normal(&self) -> f64 {
        self.inner.floating_point_info().smallest_normal()
    }

    #[getter]
    fn tiny(&self) -> f64 {
        self.smallest_normal()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        let info = self.inner.floating_point_info();
        format!(
            "finfo(resolution={}, min={}, max={}, eps={}, smallest_normal={}, tiny={}, dtype={})",
            format_finfo_value(info.resolution()),
            format_finfo_value(info.min()),
            format_finfo_value(info.max()),
            format_finfo_value(info.eps()),
            format_finfo_value(info.smallest_normal()),
            format_finfo_value(info.smallest_normal()),
            self.inner.name(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getnewargs__(&self) -> PyResult<()> {
        let qualified_type = match self.inner {
            DType::Float32 => "torch_rs.finfo",
        };
        Err(PyTypeError::new_err(format!(
            "cannot pickle '{qualified_type}' object"
        )))
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
