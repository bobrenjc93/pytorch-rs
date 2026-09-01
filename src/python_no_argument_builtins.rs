//! Stable-ABI bindings for PyTorch-style state built-in functions.

use std::{
    cell::Cell,
    ffi::{CStr, c_char},
    os::raw::c_long,
};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyCFunction, PyDict, PyInt, PyModule, PyTuple, PyType};

use crate::{
    DType, is_grad_enabled as core_is_grad_enabled,
    python::python_type_name,
    python_dtype::{PyDType, dtype_object},
};

thread_local! {
    static AUTOCAST_CACHE_ENABLED: Cell<bool> = const { Cell::new(true) };
}

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

fn is_multithreading_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch._C._is_multithreading_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch._C._is_multithreading_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    // Expose PyTorch's supported default without adding its setter or a
    // parallel backward scheduler to the native engine.
    Ok(true)
}

fn is_autocast_cache_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_autocast_cache_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_autocast_cache_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(AUTOCAST_CACHE_ENABLED.get())
}

fn clear_autocast_cache(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.clear_autocast_cache() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.clear_autocast_cache() takes no arguments ({} given)",
            args.len()
        )));
    }

    // Autocast execution is not implemented, so the associated weight cache
    // is always empty. Clearing it is therefore an exact no-op and must not
    // disturb the thread-local cache-enabled flag or grad mode.
    Ok(())
}

fn set_autocast_cache_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.set_autocast_cache_enabled() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(PyTypeError::new_err(format!(
            "torch.set_autocast_cache_enabled() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let enabled = args.get_item(0)?;
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(&enabled)?;
        return Err(PyTypeError::new_err(format!(
            "enabled must be a bool (got {type_name})"
        )));
    }
    AUTOCAST_CACHE_ENABLED.set(enabled.is_truthy()?);
    Ok(())
}

fn is_view_replay_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch._C._is_view_replay_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch._C._is_view_replay_enabled() takes no arguments ({} given)",
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

fn is_anomaly_check_nan_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_anomaly_check_nan_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_anomaly_check_nan_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(true)
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

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nget_num_interop_threads() -> int\n\nReturns the number of threads used for inter-op parallelism on CPU\n(e.g. in JIT interpreter)\n"]
fn get_num_interop_threads(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<usize> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.get_num_interop_threads() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.get_num_interop_threads() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(1)
}

#[repr(C)]
struct PyTypeObjectNamePrefix {
    _ob_base: ffi::PyVarObject,
    tp_name: *const c_char,
}

#[allow(
    unsafe_code,
    reason = "CPython exposes tp_name only as a type-object field before Python 3.13"
)]
fn cpython_type_object_name<'a>(value_type: &'a Bound<'_, PyType>) -> PyResult<&'a CStr> {
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: every classic CPython type object starts with PyVarObject and
    // tp_name. The attached interpreter keeps the live type object stable
    // while the non-overridable C name is inspected.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: CPython requires tp_name to remain NUL-terminated for the
    // lifetime of the live type object.
    Ok(unsafe { CStr::from_ptr(name) })
}

#[allow(
    unsafe_code,
    reason = "PyType_GetFlags reads immutable flags from a live type through the stable ABI"
)]
fn is_native_immutable_python_type(value_type: &Bound<'_, PyType>) -> bool {
    // SAFETY: value_type is a live Python type object for the duration of the call.
    let flags = unsafe { ffi::PyType_GetFlags(value_type.as_type_ptr()) };
    flags & ffi::Py_TPFLAGS_IMMUTABLETYPE != 0 && flags & ffi::Py_TPFLAGS_HEAPTYPE == 0
}

fn has_numpy_integer_ancestry(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    // Calling type's descriptor directly bypasses metaclass overrides, while
    // __mro__ itself is immutable for the duration of this check.
    let mro = py
        .get_type::<PyType>()
        .getattr("__getattribute__")?
        .call1((value.get_type(), "__mro__"))?
        .cast_into::<PyTuple>()?;
    for base in mro.iter() {
        let base = base.cast_into::<PyType>()?;
        if is_native_immutable_python_type(&base)
            && cpython_type_object_name(&base)? == c"numpy.integer"
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    reason = "PyLong_AsLongAndOverflow reads an int subclass without dispatching overrides"
)]
fn parse_thread_count(function: &str, value: &Bound<'_, PyAny>) -> PyResult<i32> {
    let py = value.py();
    let is_integer = !value.is_instance_of::<PyBool>()
        && (value.is_instance_of::<PyInt>() || has_numpy_integer_ancestry(value)?);
    if !is_integer {
        let type_name = python_type_name(value)?;
        return Err(PyRuntimeError::new_err(format!(
            "{function} expects an int, but got {type_name}"
        )));
    }

    let mut overflow = 0;
    // SAFETY: validation above accepts only Python int instances and NumPy
    // integer scalars. The object remains live and overflow is writable.
    let value = unsafe { ffi::PyLong_AsLongAndOverflow(value.as_ptr(), &raw mut overflow) };
    if PyErr::occurred(py) {
        return Err(PyErr::fetch(py));
    }
    if overflow != 0 {
        return Err(PyValueError::new_err("Overflow when unpacking long long"));
    }
    if value > c_long::from(i32::MAX) || value < c_long::from(i32::MIN) {
        return Err(PyValueError::new_err("Overflow when unpacking long"));
    }
    let value = value as i32;
    if value <= 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{function} expects a positive integer"
        )));
    }
    Ok(value)
}

fn set_singleton_thread_count(
    function: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(format!(
            "torch.{function}() takes no keyword arguments"
        )));
    }
    if args.len() != 1 {
        return Err(PyTypeError::new_err(format!(
            "torch.{function}() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let threads = parse_thread_count(function, &args.get_item(0)?)?;
    if threads != 1 {
        return Err(PyNotImplementedError::new_err(format!(
            "torch.{function}() only supports the singleton thread count 1"
        )));
    }
    Ok(())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nset_num_threads(int)\n\nSets the number of threads used for intraop parallelism on CPU.\n\n.. warning::\n    To ensure that the correct number of threads is used, set_num_threads\n    must be called before running eager, JIT or autograd code.\n"]
fn set_num_threads(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    set_singleton_thread_count("set_num_threads", args, kwargs)
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nset_num_interop_threads(int)\n\nSets the number of threads used for interop parallelism\n(e.g. in JIT interpreter) on CPU.\n\n.. warning::\n    Can only be called once and before any inter-op parallel work\n    is started (e.g. JIT execution).\n"]
fn set_num_interop_threads(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    set_singleton_thread_count("set_num_interop_threads", args, kwargs)
}

const IS_GRAD_ENABLED_DOC: &CStr =
    c"\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n";
const IS_GRAD_ENABLED_SIGNATURE_DOC: &CStr = c"is_grad_enabled($self, /)\n--\n\n\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n";
const IS_INFERENCE_MODE_ENABLED_DOC: &CStr = c"\nis_inference_mode_enabled() -> (bool)\n\nReturns True if inference mode is currently enabled.\n";
const IS_INFERENCE_MODE_ENABLED_SIGNATURE_DOC: &CStr = c"is_inference_mode_enabled($self, /)\n--\n\n\nis_inference_mode_enabled() -> (bool)\n\nReturns True if inference mode is currently enabled.\n";
const IS_MULTITHREADING_ENABLED_DOC: &CStr =
    c"Returns True if multithreading is currently enabled.";
const IS_MULTITHREADING_ENABLED_SIGNATURE_DOC: &CStr =
    c"_is_multithreading_enabled($self, /)\n--\n\nReturns True if multithreading is currently enabled.";
const IS_VIEW_REPLAY_ENABLED_DOC: &CStr = c"Returns True if view-replay is currently enabled.";
const IS_VIEW_REPLAY_ENABLED_SIGNATURE_DOC: &CStr =
    c"_is_view_replay_enabled($self, /)\n--\n\nReturns True if view-replay is currently enabled.";
// PyTorch leaves these built-ins' documentation null. On CPython 3.13+ their
// METH_NOARGS and METH_O definitions nevertheless expose synthesized
// signatures; signature-only internal docs reproduce both observations while
// retaining the custom PyTorch-qualified argument diagnostics below.
const IS_AUTOCAST_CACHE_ENABLED_SIGNATURE_DOC: &CStr =
    c"is_autocast_cache_enabled($self, /)\n--\n\n";
const CLEAR_AUTOCAST_CACHE_SIGNATURE_DOC: &CStr = c"clear_autocast_cache($self, /)\n--\n\n";
const SET_AUTOCAST_CACHE_ENABLED_SIGNATURE_DOC: &CStr =
    c"set_autocast_cache_enabled($self, object, /)\n--\n\n";
const IS_ANOMALY_ENABLED_SIGNATURE_DOC: &CStr = c"is_anomaly_enabled($self, /)\n--\n\n";
const IS_ANOMALY_CHECK_NAN_ENABLED_SIGNATURE_DOC: &CStr =
    c"is_anomaly_check_nan_enabled($self, /)\n--\n\n";
const GET_DEFAULT_DTYPE_DOC: &CStr = c"\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n";
const GET_DEFAULT_DTYPE_SIGNATURE_DOC: &CStr = c"get_default_dtype($self, /)\n--\n\n\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n";
const GET_NUM_THREADS_DOC: &CStr = c"\nget_num_threads() -> int\n\nReturns the number of threads used for parallelizing CPU operations\n";
const GET_NUM_THREADS_SIGNATURE_DOC: &CStr = c"get_num_threads($self, /)\n--\n\n\nget_num_threads() -> int\n\nReturns the number of threads used for parallelizing CPU operations\n";
const GET_NUM_INTEROP_THREADS_DOC: &CStr = c"\nget_num_interop_threads() -> int\n\nReturns the number of threads used for inter-op parallelism on CPU\n(e.g. in JIT interpreter)\n";
const GET_NUM_INTEROP_THREADS_SIGNATURE_DOC: &CStr = c"get_num_interop_threads($self, /)\n--\n\n\nget_num_interop_threads() -> int\n\nReturns the number of threads used for inter-op parallelism on CPU\n(e.g. in JIT interpreter)\n";
const SET_NUM_THREADS_DOC: &CStr = c"\nset_num_threads(int)\n\nSets the number of threads used for intraop parallelism on CPU.\n\n.. warning::\n    To ensure that the correct number of threads is used, set_num_threads\n    must be called before running eager, JIT or autograd code.\n";
const SET_NUM_INTEROP_THREADS_DOC: &CStr = c"\nset_num_interop_threads(int)\n\nSets the number of threads used for interop parallelism\n(e.g. in JIT interpreter) on CPU.\n\n.. warning::\n    Can only be called once and before any inter-op parallel work\n    is started (e.g. JIT execution).\n";

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
unsafe fn is_multithreading_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_multithreading_enabled(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_autocast_cache_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_autocast_cache_enabled(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn clear_autocast_cache_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    clear_autocast_cache(&args, kwargs.as_ref())?;
    Ok(py.None().into_ptr())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_autocast_cache_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    set_autocast_cache_enabled(&args, kwargs.as_ref())?;
    Ok(py.None().into_ptr())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn is_view_replay_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_view_replay_enabled(&args, kwargs.as_ref())?
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
unsafe fn is_anomaly_check_nan_enabled_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    is_anomaly_check_nan_enabled(&args, kwargs.as_ref())?
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

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn get_num_interop_threads_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    get_num_interop_threads(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_num_threads_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    set_num_threads(&args, kwargs.as_ref())?;
    Ok(py.None().into_ptr())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_num_interop_threads_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let (args, kwargs) = unsafe { no_argument_builtin_arguments(py, args, kwargs) }?;
    set_num_interop_threads(&args, kwargs.as_ref())?;
    Ok(py.None().into_ptr())
}

fn add_autocast_cache_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let (is_autocast_cache_enabled_doc, clear_autocast_cache_doc, set_autocast_cache_enabled_doc) =
        if py.version_info() >= (3, 13) {
            (
                IS_AUTOCAST_CACHE_ENABLED_SIGNATURE_DOC,
                CLEAR_AUTOCAST_CACHE_SIGNATURE_DOC,
                SET_AUTOCAST_CACHE_ENABLED_SIGNATURE_DOC,
            )
        } else {
            (c"", c"", c"")
        };
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_autocast_cache_enabled_callback
        ),
        c"is_autocast_cache_enabled",
        is_autocast_cache_enabled_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            clear_autocast_cache_callback
        ),
        c"clear_autocast_cache",
        clear_autocast_cache_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            set_autocast_cache_enabled_callback
        ),
        c"set_autocast_cache_enabled",
        set_autocast_cache_enabled_doc,
        Some(module),
    )?)?;
    Ok(())
}

fn add_multithreading_builtin(
    module: &Bound<'_, PyModule>,
    is_multithreading_enabled_doc: &'static CStr,
) -> PyResult<()> {
    let py = module.py();
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_multithreading_enabled_callback
        ),
        c"_is_multithreading_enabled",
        is_multithreading_enabled_doc,
        Some(module),
    )?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_is_multithreading_enabled",))?;
    Ok(())
}

fn add_view_replay_builtin(
    module: &Bound<'_, PyModule>,
    is_view_replay_enabled_doc: &'static CStr,
) -> PyResult<()> {
    let py = module.py();
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            is_view_replay_enabled_callback
        ),
        c"_is_view_replay_enabled",
        is_view_replay_enabled_doc,
        Some(module),
    )?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_is_view_replay_enabled",))?;
    Ok(())
}

fn add_anomaly_builtins(
    module: &Bound<'_, PyModule>,
    is_anomaly_enabled_doc: &'static CStr,
    is_anomaly_check_nan_enabled_doc: &'static CStr,
) -> PyResult<()> {
    let py = module.py();
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
            is_anomaly_check_nan_enabled_callback
        ),
        c"is_anomaly_check_nan_enabled",
        is_anomaly_check_nan_enabled_doc,
        Some(module),
    )?)?;
    Ok(())
}

fn add_thread_count_builtins(
    module: &Bound<'_, PyModule>,
    get_num_threads_doc: &'static CStr,
    get_num_interop_threads_doc: &'static CStr,
) -> PyResult<()> {
    let py = module.py();
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
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            set_num_threads_callback
        ),
        c"set_num_threads",
        SET_NUM_THREADS_DOC,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            get_num_interop_threads_callback
        ),
        c"get_num_interop_threads",
        get_num_interop_threads_doc,
        Some(module),
    )?)?;
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            set_num_interop_threads_callback
        ),
        c"set_num_interop_threads",
        SET_NUM_INTEROP_THREADS_DOC,
        Some(module),
    )?)?;
    Ok(())
}

pub(crate) fn add_no_argument_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let (
        is_grad_enabled_doc,
        is_inference_mode_enabled_doc,
        is_multithreading_enabled_doc,
        is_view_replay_enabled_doc,
        is_anomaly_enabled_doc,
        is_anomaly_check_nan_enabled_doc,
        get_default_dtype_doc,
        get_num_threads_doc,
        get_num_interop_threads_doc,
    ) = if py.version_info() >= (3, 13) {
        (
            IS_GRAD_ENABLED_SIGNATURE_DOC,
            IS_INFERENCE_MODE_ENABLED_SIGNATURE_DOC,
            IS_MULTITHREADING_ENABLED_SIGNATURE_DOC,
            IS_VIEW_REPLAY_ENABLED_SIGNATURE_DOC,
            IS_ANOMALY_ENABLED_SIGNATURE_DOC,
            IS_ANOMALY_CHECK_NAN_ENABLED_SIGNATURE_DOC,
            GET_DEFAULT_DTYPE_SIGNATURE_DOC,
            GET_NUM_THREADS_SIGNATURE_DOC,
            GET_NUM_INTEROP_THREADS_SIGNATURE_DOC,
        )
    } else {
        (
            IS_GRAD_ENABLED_DOC,
            IS_INFERENCE_MODE_ENABLED_DOC,
            IS_MULTITHREADING_ENABLED_DOC,
            IS_VIEW_REPLAY_ENABLED_DOC,
            c"",
            c"",
            GET_DEFAULT_DTYPE_DOC,
            GET_NUM_THREADS_DOC,
            GET_NUM_INTEROP_THREADS_DOC,
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
    add_autocast_cache_builtins(module)?;
    add_multithreading_builtin(module, is_multithreading_enabled_doc)?;
    add_view_replay_builtin(module, is_view_replay_enabled_doc)?;
    add_anomaly_builtins(
        module,
        is_anomaly_enabled_doc,
        is_anomaly_check_nan_enabled_doc,
    )?;
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
    add_thread_count_builtins(module, get_num_threads_doc, get_num_interop_threads_doc)?;
    Ok(())
}
