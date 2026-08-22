//! Stable-ABI binding for the default-state autocast query.

use std::ffi::CStr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyCFunction, PyDict, PyModule, PyString, PyTuple};

use crate::{
    python::{legacy_dict_get_item_string, python_type_name, pytorch_ordered_keyword_entries},
    python_device::{RecognizedDeviceType, parse_device_specification},
};

const INVALID_COMBINATION_SUFFIX: &str = "), but expected one of:\n * (str device_type)\n * ()\n";
const IS_AUTOCAST_ENABLED_DOC: &CStr = c"";

fn invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let keyword_arguments = match keywords {
        Some(keywords) => pytorch_ordered_keyword_entries(keywords)?,
        None => Vec::new(),
    };
    let mut arguments = Vec::with_capacity(positional.len() + keyword_arguments.len());
    for value in positional {
        arguments.push(python_type_name(&value)?);
    }
    for (name, value) in keyword_arguments {
        arguments.push(format!("{name}={}", python_type_name(&value)?));
    }
    let mut summary = arguments.join(", ");
    if positional.is_empty() && !summary.is_empty() {
        summary.push_str(", ");
    }
    Ok(PyTypeError::new_err(format!(
        "is_autocast_enabled() received an invalid combination of arguments - got ({summary}{INVALID_COMBINATION_SUFFIX}"
    )))
}

fn invalid_byte_device_string_error(py: Python<'_>, specification: &[u8]) -> PyErr {
    let mut message = Vec::with_capacity("Invalid device string: ''".len() + specification.len());
    message.extend_from_slice(b"Invalid device string: '");
    message.extend_from_slice(specification);
    message.push(b'\'');
    if let Some(nul) = message.iter().position(|byte| *byte == 0) {
        message.truncate(nul);
    }
    let encoded = PyBytes::new(py, &message);
    match PyString::from_encoded_object(encoded.as_any(), None, None) {
        Ok(message) => PyRuntimeError::new_err(message.to_string()),
        Err(error) => error,
    }
}

fn device_type_error(value: &Bound<'_, PyAny>, positional: bool) -> PyResult<PyErr> {
    let position = if positional { " (position 1)" } else { "" };
    let actual = python_type_name(value)?;
    Ok(PyTypeError::new_err(format!(
        "is_autocast_enabled(): argument 'device_type'{position} must be str, not {actual}"
    )))
}

fn parse_autocast_device_type(
    value: &Bound<'_, PyAny>,
    positional: bool,
) -> PyResult<RecognizedDeviceType> {
    let device_type = if let Ok(value) = value.cast::<PyString>() {
        let specification = value
            .to_str()
            .map_err(|_| PyRuntimeError::new_err("error unpacking string as utf-8"))?;
        parse_device_specification(specification)?.0
    } else if let Ok(value) = value.cast::<PyBytes>() {
        let bytes = value.as_bytes();
        let specification = std::str::from_utf8(bytes)
            .map_err(|_| invalid_byte_device_string_error(value.py(), bytes))?;
        parse_device_specification(specification)?.0
    } else {
        return Err(device_type_error(value, positional)?);
    };

    if !device_type.supports_autocast() {
        return Err(PyRuntimeError::new_err(
            "unknown device type for autocast in get_autocast_dispatch_key_from_device_type",
        ));
    }
    Ok(device_type)
}

fn is_autocast_enabled(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    let keyword_count = keywords.map_or(0, pyo3::types::PyDictMethods::len);
    let device_type = match (positional.len(), keyword_count) {
        (0, 0) => None,
        (0, 1) => {
            let keywords = keywords.expect("one keyword argument has a dictionary");
            let Some(value) = legacy_dict_get_item_string(keywords, c"device_type") else {
                return Err(PyTypeError::new_err(
                    "is_autocast_enabled() missing 1 required positional arguments: \"device_type\"",
                ));
            };
            Some((value, false))
        }
        (1, 0) => Some((positional.get_item(0)?, true)),
        _ => return Err(invalid_combination(positional, keywords)?),
    };
    if let Some((value, positional)) = device_type {
        parse_autocast_device_type(&value, positional)?;
    }
    // The native engine deliberately exposes only the default disabled state;
    // autocast contexts, setters, dtype controls, and transitions are absent.
    Ok(false)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_autocast_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards a live positional tuple and either a
    // null keyword pointer or a live dictionary for the duration of the call.
    let positional =
        unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    let keywords = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    is_autocast_enabled(&positional, keywords.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

pub(crate) fn add_autocast_enabled(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_autocast_enabled_callback
        ),
        c"is_autocast_enabled",
        IS_AUTOCAST_ENABLED_DOC,
        Some(module),
    )?)?;
    Ok(())
}
