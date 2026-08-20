//! Python bindings for process-global float32 matrix-multiplication precision.

use std::ffi::CString;
use std::sync::atomic::{AtomicU8, Ordering};

use pyo3::exceptions::{PyRuntimeError, PyUserWarning};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyString};

use crate::python::python_type_name;

const HIGHEST: u8 = 0;
const HIGH: u8 = 1;
const MEDIUM: u8 = 2;
const INVALID_PRECISION_TYPE_PREFIX: &str = "set_float32_matmul_precision expects a str, but got ";
const INVALID_PRECISION_WARNING_SUFFIX: &[u8] = b" is not one of 'highest', 'high', or 'medium'; the currentsetFloat32MatmulPrecision call has no effect.";
#[cfg(target_os = "macos")]
const INVALID_PRECISION_WARNING_LOCATION: &[u8] =
    b" (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/Context.cpp:458.)";
#[cfg(target_os = "linux")]
const INVALID_PRECISION_WARNING_LOCATION: &[u8] =
    b" (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/Context.cpp:458.)";
#[cfg(target_os = "windows")]
const INVALID_PRECISION_WARNING_LOCATION: &[u8] = b" (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\Context.cpp:458.)";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const INVALID_PRECISION_WARNING_LOCATION: &[u8] =
    b" (Triggered internally at aten/src/ATen/Context.cpp:458.)";

static FLOAT32_MATMUL_PRECISION: AtomicU8 = AtomicU8::new(HIGHEST);

fn allocation_error() -> PyErr {
    PyRuntimeError::new_err("std::bad_alloc")
}

#[pyfunction]
fn _get_float32_matmul_precision() -> &'static str {
    match FLOAT32_MATMUL_PRECISION.load(Ordering::SeqCst) {
        HIGHEST => "highest",
        HIGH => "high",
        MEDIUM => "medium",
        _ => unreachable!("float32 matmul precision stores only validated values"),
    }
}

fn precision_bytes<'a>(precision: &'a Bound<'_, PyAny>) -> PyResult<&'a [u8]> {
    if let Ok(precision) = precision.cast::<PyString>() {
        return precision
            .to_str()
            .map(str::as_bytes)
            .map_err(|_| PyRuntimeError::new_err("error unpacking string as utf-8"));
    }
    if let Ok(precision) = precision.cast::<PyBytes>() {
        return Ok(precision.as_bytes());
    }

    let type_name = python_type_name(precision)?;
    let capacity = INVALID_PRECISION_TYPE_PREFIX
        .len()
        .checked_add(type_name.len())
        .ok_or_else(allocation_error)?;
    let mut message = String::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    message.push_str(INVALID_PRECISION_TYPE_PREFIX);
    message.push_str(&type_name);
    Err(PyRuntimeError::new_err(message))
}

fn warn_invalid_precision(py: Python<'_>, precision: &[u8]) -> PyResult<()> {
    let capacity = precision
        .len()
        .checked_add(INVALID_PRECISION_WARNING_SUFFIX.len())
        .and_then(|length| length.checked_add(INVALID_PRECISION_WARNING_LOCATION.len()))
        .and_then(|length| length.checked_add(1))
        .ok_or_else(allocation_error)?;
    let mut message = Vec::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    message.extend_from_slice(precision);
    message.extend_from_slice(INVALID_PRECISION_WARNING_SUFFIX);
    message.extend_from_slice(INVALID_PRECISION_WARNING_LOCATION);
    if let Some(nul) = message.iter().position(|byte| *byte == 0) {
        message.truncate(nul);
    }
    let message = CString::new(message)
        .map_err(|_| PyRuntimeError::new_err("unable to construct precision warning"))?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)
}

#[pyfunction]
fn _set_float32_matmul_precision(py: Python<'_>, precision: &Bound<'_, PyAny>) -> PyResult<()> {
    let precision = precision_bytes(precision)?;
    let value = match precision {
        b"highest" => HIGHEST,
        b"high" => HIGH,
        b"medium" => MEDIUM,
        _ => return warn_invalid_precision(py, precision),
    };
    FLOAT32_MATMUL_PRECISION.store(value, Ordering::SeqCst);
    Ok(())
}

pub(crate) fn add_float32_matmul_precision(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_get_float32_matmul_precision, module)?)?;
    module.add_function(wrap_pyfunction!(_set_float32_matmul_precision, module)?)?;
    let exports = module.getattr("__all__")?;
    for name in [
        "_get_float32_matmul_precision",
        "_set_float32_matmul_precision",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    Ok(())
}
