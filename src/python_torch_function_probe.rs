//! Stable-ABI bindings for the fast `__torch_function__` probes.

use std::ffi::CStr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PySystemError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCFunction, PyDict, PyModule, PyTuple};

use crate::{python::PyTensor, python_torch_function_mode};

const HAS_TORCH_FUNCTION_UNARY_DOC: &CStr = c"Special case of `has_torch_function` for single inputs.\n    Instead of:\n      `has_torch_function((t,))`\n    call:\n      `has_torch_function_unary(t)`\n    which skips unnecessary packing and unpacking work.\n    ";
const HAS_TORCH_FUNCTION_UNARY_SIGNATURE_DOC: &CStr = c"_has_torch_function_unary($self, object, /)\n--\n\nSpecial case of `has_torch_function` for single inputs.\n    Instead of:\n      `has_torch_function((t,))`\n    call:\n      `has_torch_function_unary(t)`\n    which skips unnecessary packing and unpacking work.\n    ";
const HAS_TORCH_FUNCTION_VARIADIC_DOC: &CStr = c"Special case of `has_torch_function` that skips tuple creation.\n\n    This uses the METH_FASTCALL protocol introduced in Python 3.7\n\n    Instead of:\n      `has_torch_function((a, b))`\n    call:\n      `has_torch_function_variadic(a, b)`\n    which skips unnecessary packing and unpacking work.\n    ";
const HAS_TORCH_FUNCTION_VARIADIC_SIGNATURE_DOC: &CStr = c"_has_torch_function_variadic($self, /, *args)\n--\n\nSpecial case of `has_torch_function` that skips tuple creation.\n\n    This uses the METH_FASTCALL protocol introduced in Python 3.7\n\n    Instead of:\n      `has_torch_function((a, b))`\n    call:\n      `has_torch_function_variadic(a, b)`\n    which skips unnecessary packing and unpacking work.\n    ";

fn is_disabled_torch_function_handler(handler: &Bound<'_, PyAny>) -> bool {
    if handler.cast::<PyCFunction>().is_err() {
        return false;
    }

    let module_matches = handler
        .getattr("__module__")
        .and_then(|module| module.eq("torch._C"))
        .unwrap_or(false);
    let name_matches = handler
        .getattr("__name__")
        .and_then(|name| name.eq("_disabled_torch_function_impl"))
        .unwrap_or(false);
    module_matches && name_matches
}

#[allow(
    unsafe_code,
    reason = "PyTorch suppresses errors while probing the __torch_function__ descriptor"
)]
fn has_torch_function_unary(value: &Bound<'_, PyAny>) -> bool {
    if !python_torch_function_mode::is_empty() {
        return true;
    }
    if value.is_exact_instance_of::<PyTensor>() {
        return false;
    }
    if value.as_ptr() == value.py().get_type::<PyTensor>().as_ptr() {
        return true;
    }

    // PyTorch's fast attribute probe clears failures from user-defined
    // descriptors instead of surfacing them from this predicate.
    // SAFETY: `value` is live for the call and the attribute name is a static,
    // NUL-terminated string. A non-null result is a new owned reference.
    let handler =
        unsafe { ffi::PyObject_GetAttrString(value.as_ptr(), c"__torch_function__".as_ptr()) };
    if handler.is_null() {
        // SAFETY: the GIL is held and clearing the descriptor lookup failure is
        // the observable behavior of PyTorch's exception-suppressing probe.
        unsafe { ffi::PyErr_Clear() };
        return false;
    }
    // SAFETY: PyObject_GetAttrString returned a new owned reference.
    let handler = unsafe { Bound::<PyAny>::from_owned_ptr(value.py(), handler) };
    !is_disabled_torch_function_handler(&handler)
}

fn bind_has_torch_function_unary(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch._C._has_torch_function_unary() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(PyTypeError::new_err(format!(
            "torch._C._has_torch_function_unary() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(has_torch_function_unary(&args.get_item(0)?))
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn has_torch_function_unary_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards a live positional tuple and either a
    // null keyword pointer or a live dictionary for the duration of the call.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    bind_has_torch_function_unary(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe fastcall trampoline"
)]
unsafe fn has_torch_function_variadic_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // PyTorch registers this probe without METH_KEYWORDS. Registering the
    // keyword-aware fastcall shape lets this package preserve PyTorch's error
    // spelling even though its importable extension name is `torch_rs.torch_rs`.
    let keyword_names = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwnames) }
        .map(Bound::cast_into::<PyTuple>)
        .transpose()?;
    if keyword_names.is_some_and(|names| !names.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch._C._has_torch_function_variadic() takes no keyword arguments",
        ));
    }

    let argument_count = usize::try_from(nargs).map_err(|_| {
        PySystemError::new_err("negative argument count passed to variadic torch function probe")
    })?;
    let arguments: &[*mut ffi::PyObject] = if argument_count == 0 {
        &[]
    } else {
        // SAFETY: CPython's FASTCALL contract provides `nargs` live positional
        // argument pointers. The borrowed objects remain live for this call.
        unsafe { std::slice::from_raw_parts(args, argument_count) }
    };
    for &argument in arguments {
        // SAFETY: each pointer belongs to the live FASTCALL argument array and
        // is borrowed only for the duration of this callback.
        let argument = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, argument) };
        if has_torch_function_unary(&argument) {
            return true.into_py_any(py).map(Py::into_ptr);
        }
    }
    false.into_py_any(py).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the leaked method definition is registered through PyO3's checked constructor"
)]
fn add_has_torch_function_variadic(
    module: &Bound<'_, PyModule>,
    doc: &'static CStr,
) -> PyResult<()> {
    let method_def = pyo3::impl_::pymethods::PyMethodDef::fastcall_cfunction_with_keywords(
        c"_has_torch_function_variadic",
        pyo3::impl_::trampoline::get_trampoline_function!(
            fastcall_cfunction_with_keywords,
            has_torch_function_variadic_callback
        ),
        doc,
    )
    .into_raw();
    // PyCFunction keeps a pointer to its method definition for its full
    // lifetime. PyO3's public constructors use the same one-time leak.
    let method_def = Box::leak(Box::new(method_def));
    // SAFETY: the leaked method definition is valid for the interpreter's
    // lifetime and its callback uses the matching FASTCALL ABI.
    let function = unsafe {
        pyo3::impl_::pyfunction::create_py_c_function(module.py(), method_def, Some(module))
    }?;
    module.add_function(function)
}

pub(crate) fn add_torch_function_probe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let (unary_doc, variadic_doc) = if py.version_info() >= (3, 13) {
        (
            HAS_TORCH_FUNCTION_UNARY_SIGNATURE_DOC,
            HAS_TORCH_FUNCTION_VARIADIC_SIGNATURE_DOC,
        )
    } else {
        (
            HAS_TORCH_FUNCTION_UNARY_DOC,
            HAS_TORCH_FUNCTION_VARIADIC_DOC,
        )
    };
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            has_torch_function_unary_callback
        ),
        c"_has_torch_function_unary",
        unary_doc,
        Some(module),
    )?)?;
    add_has_torch_function_variadic(module, variadic_doc)?;
    let native_exports = module.getattr("__all__")?;
    for name in ["_has_torch_function_unary", "_has_torch_function_variadic"] {
        native_exports.call_method1("remove", (name,))?;
    }
    Ok(())
}
