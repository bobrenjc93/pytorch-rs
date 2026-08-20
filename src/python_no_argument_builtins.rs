//! Stable-ABI bindings for PyTorch-style no-argument built-in functions.

use std::ffi::CStr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCFunction, PyDict, PyModule, PyTuple};

use crate::{
    DType, is_grad_enabled as core_is_grad_enabled,
    python_dtype::{PyDType, dtype_object},
};

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n"]
fn is_grad_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_grad_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_grad_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(core_is_grad_enabled())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nis_inference_mode_enabled() -> (bool)\n\nReturns True if inference mode is currently enabled.\n"]
fn is_inference_mode_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_inference_mode_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_inference_mode_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(false)
}

fn is_anomaly_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_anomaly_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_anomaly_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(false)
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n"
)]
#[cfg_attr(doc, doc = "Get the current default floating-point dtype.")]
fn get_default_dtype(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDType>> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.get_default_dtype() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.get_default_dtype() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(dtype_object(py, DType::Float32)?.clone_ref(py))
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nget_num_threads() -> int\n\nReturns the number of threads used for parallelizing CPU operations\n"]
fn get_num_threads(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<usize> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.get_num_threads() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.get_num_threads() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(1)
}

const IS_GRAD_ENABLED_DOC: &CStr =
    c"\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n";
const IS_GRAD_ENABLED_SIGNATURE_DOC: &CStr = c"is_grad_enabled($self, /)\n--\n\n\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n";
const IS_INFERENCE_MODE_ENABLED_DOC: &CStr = c"\nis_inference_mode_enabled() -> (bool)\n\nReturns True if inference mode is currently enabled.\n";
const IS_INFERENCE_MODE_ENABLED_SIGNATURE_DOC: &CStr = c"is_inference_mode_enabled($self, /)\n--\n\n\nis_inference_mode_enabled() -> (bool)\n\nReturns True if inference mode is currently enabled.\n";
// PyTorch leaves this built-in's documentation null. On CPython 3.13+ its
// METH_NOARGS definition nevertheless exposes the synthesized `($self, /)`
// signature; a signature-only internal doc reproduces both observations while
// retaining the custom PyTorch-qualified argument diagnostics below.
const IS_ANOMALY_ENABLED_SIGNATURE_DOC: &CStr = c"is_anomaly_enabled($self, /)\n--\n\n";
const GET_DEFAULT_DTYPE_DOC: &CStr = c"\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n";
const GET_DEFAULT_DTYPE_SIGNATURE_DOC: &CStr = c"get_default_dtype($self, /)\n--\n\n\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n";
const GET_NUM_THREADS_DOC: &CStr = c"\nget_num_threads() -> int\n\nReturns the number of threads used for parallelizing CPU operations\n";
const GET_NUM_THREADS_SIGNATURE_DOC: &CStr = c"get_num_threads($self, /)\n--\n\n\nget_num_threads() -> int\n\nReturns the number of threads used for parallelizing CPU operations\n";

#[allow(
    unsafe_code,
    reason = "CPython passes borrowed tuple and dictionary pointers to C function callbacks"
)]
unsafe fn no_argument_builtin_arguments(
    py: Python<'_>,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<(Bound<'_, PyTuple>, Option<Bound<'_, PyDict>>)> {
    // SAFETY: the C callback contract supplies a live positional tuple.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is either null or a live dictionary for the
    // duration of the callback.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    Ok((args, kwargs))
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_grad_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_grad_enabled(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_inference_mode_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_inference_mode_enabled(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_anomaly_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_anomaly_enabled(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn get_default_dtype_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    get_default_dtype(py, &args, kwargs.as_ref()).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn get_num_threads_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    get_num_threads(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

pub(crate) fn add_no_argument_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let (
        is_grad_enabled_doc,
        is_inference_mode_enabled_doc,
        is_anomaly_enabled_doc,
        get_default_dtype_doc,
        get_num_threads_doc,
    ) = if py.version_info() >= (3, 13) {
        (
            IS_GRAD_ENABLED_SIGNATURE_DOC,
            IS_INFERENCE_MODE_ENABLED_SIGNATURE_DOC,
            IS_ANOMALY_ENABLED_SIGNATURE_DOC,
            GET_DEFAULT_DTYPE_SIGNATURE_DOC,
            GET_NUM_THREADS_SIGNATURE_DOC,
        )
    } else {
        (
            IS_GRAD_ENABLED_DOC,
            IS_INFERENCE_MODE_ENABLED_DOC,
            c"",
            GET_DEFAULT_DTYPE_DOC,
            GET_NUM_THREADS_DOC,
        )
    };
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_grad_enabled_callback
        ),
        c"is_grad_enabled",
        is_grad_enabled_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_inference_mode_enabled_callback
        ),
        c"is_inference_mode_enabled",
        is_inference_mode_enabled_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_anomaly_enabled_callback
        ),
        c"is_anomaly_enabled",
        is_anomaly_enabled_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            get_default_dtype_callback
        ),
        c"get_default_dtype",
        get_default_dtype_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            get_num_threads_callback
        ),
        c"get_num_threads",
        get_num_threads_doc,
        Some(module),
    )?)?;
    Ok(())
}
