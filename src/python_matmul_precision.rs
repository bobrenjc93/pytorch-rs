//! Process-global float32 matrix-multiplication precision state.

use std::ffi::CStr;
use std::sync::atomic::{AtomicU8, Ordering};

use pyo3::exceptions::{PyRuntimeError, PyUserWarning};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyString};

use crate::python::python_type_name;

const HIGHEST: u8 = 0;
const HIGH: u8 = 1;
const MEDIUM: u8 = 2;

static FLOAT32_MATMUL_PRECISION: AtomicU8 = AtomicU8::new(HIGHEST);

const INVALID_PRECISION_WARNING: &[u8] =
    b" is not one of 'highest', 'high', or 'medium'; the currentsetFloat32MatmulPrecision call has no effect.";

#[cfg(target_os = "linux")]
const WARNING_SOURCE: &[u8] =
    b" (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/Context.cpp:458.)";
#[cfg(target_os = "macos")]
const WARNING_SOURCE: &[u8] =
    b" (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/Context.cpp:458.)";
#[cfg(target_os = "windows")]
const WARNING_SOURCE: &[u8] = b" (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\Context.cpp:458.)";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const WARNING_SOURCE: &[u8] = b"";

fn allocation_error() -> PyErr {
    PyRuntimeError::new_err("std::bad_alloc")
}

#[pyfunction]
fn _get_float32_matmul_precision() -> &'static str {
    match FLOAT32_MATMUL_PRECISION.load(Ordering::SeqCst) {
        HIGHEST => "highest",
        HIGH => "high",
        MEDIUM => "medium",
        _ => unreachable!("float32 matmul precision contains an invalid state"),
    }
}

#[pyfunction(signature = (precision, /))]
fn _set_float32_matmul_precision(py: Python<'_>, precision: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(value) = precision.cast::<PyBytes>() {
        return set_float32_matmul_precision_bytes(py, value.as_bytes());
    }
    if let Ok(value) = precision.cast::<PyString>() {
        let value = value
            .to_str()
            .map_err(|_| PyRuntimeError::new_err("error unpacking string as utf-8"))?;
        return set_float32_matmul_precision_bytes(py, value.as_bytes());
    }

    let type_name = python_type_name(precision)?;
    Err(PyRuntimeError::new_err(format!(
        "set_float32_matmul_precision expects a str, but got {type_name}"
    )))
}

fn set_float32_matmul_precision_bytes(py: Python<'_>, precision: &[u8]) -> PyResult<()> {
    let state = match precision {
        b"highest" => Some(HIGHEST),
        b"high" => Some(HIGH),
        b"medium" => Some(MEDIUM),
        _ => None,
    };
    if let Some(state) = state {
        FLOAT32_MATMUL_PRECISION.store(state, Ordering::SeqCst);
        return Ok(());
    }

    warn_invalid_precision(py, precision)
}

fn warn_invalid_precision(py: Python<'_>, precision: &[u8]) -> PyResult<()> {
    let visible_precision = precision
        .iter()
        .position(|byte| *byte == 0)
        .map_or(precision, |index| &precision[..index]);
    let include_explanation = visible_precision.len() == precision.len();
    let explanation_length = if include_explanation {
        INVALID_PRECISION_WARNING
            .len()
            .checked_add(WARNING_SOURCE.len())
            .ok_or_else(allocation_error)?
    } else {
        0
    };
    let capacity = visible_precision
        .len()
        .checked_add(explanation_length)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(allocation_error)?;
    let mut message = Vec::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    message.extend_from_slice(visible_precision);
    if include_explanation {
        message.extend_from_slice(INVALID_PRECISION_WARNING);
        message.extend_from_slice(WARNING_SOURCE);
    }
    message.push(0);
    let message = CStr::from_bytes_with_nul(&message)
        .map_err(|_| PyRuntimeError::new_err("invalid float32 matmul precision warning"))?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), message, 1)
}

pub(crate) fn add_float32_matmul_precision(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_get_float32_matmul_precision, module)?)?;
    module.add_function(wrap_pyfunction!(_set_float32_matmul_precision, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_get_float32_matmul_precision",))?;
    exports.call_method1("remove", ("_set_float32_matmul_precision",))?;
    Ok(())
}
