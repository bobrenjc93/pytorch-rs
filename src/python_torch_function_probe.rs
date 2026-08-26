//! Stable-ABI binding for the unary `__torch_function__` probe.

use std::ffi::CStr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCFunction, PyDict, PyModule, PyTuple};

use crate::{python::PyTensor, python_torch_function_mode};

const HAS_TORCH_FUNCTION_UNARY_DOC: &CStr = c"Special case of `has_torch_function` for single inputs.\n    Instead of:\n      `has_torch_function((t,))`\n    call:\n      `has_torch_function_unary(t)`\n    which skips unnecessary packing and unpacking work.\n    ";
const HAS_TORCH_FUNCTION_UNARY_SIGNATURE_DOC: &CStr = c"_has_torch_function_unary($self, object, /)\n--\n\nSpecial case of `has_torch_function` for single inputs.\n    Instead of:\n      `has_torch_function((t,))`\n    call:\n      `has_torch_function_unary(t)`\n    which skips unnecessary packing and unpacking work.\n    ";

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TorchFunctionHandlerProbe {
    Missing,
    Disabled,
    Enabled,
}

#[allow(
    unsafe_code,
    reason = "PyTorch suppresses errors while probing the __torch_function__ descriptor"
)]
pub(crate) fn probe_torch_function_handler(value: &Bound<'_, PyAny>) -> TorchFunctionHandlerProbe {
    // SAFETY: `value` is live for this call and the attribute name is a static,
    // NUL-terminated string. A non-null result is a new owned reference.
    let handler =
        unsafe { ffi::PyObject_GetAttrString(value.as_ptr(), c"__torch_function__".as_ptr()) };
    if handler.is_null() {
        // SAFETY: the GIL is held and clearing the descriptor lookup failure is
        // the observable behavior of PyTorch's exception-suppressing probe.
        unsafe { ffi::PyErr_Clear() };
        return TorchFunctionHandlerProbe::Missing;
    }
    // SAFETY: PyObject_GetAttrString returned a new owned reference.
    let handler = unsafe { Bound::<PyAny>::from_owned_ptr(value.py(), handler) };
    if is_disabled_torch_function_handler(&handler) {
        TorchFunctionHandlerProbe::Disabled
    } else {
        TorchFunctionHandlerProbe::Enabled
    }
}

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

    probe_torch_function_handler(value) == TorchFunctionHandlerProbe::Enabled
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

pub(crate) fn add_torch_function_probe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let doc = if py.version_info() >= (3, 13) {
        HAS_TORCH_FUNCTION_UNARY_SIGNATURE_DOC
    } else {
        HAS_TORCH_FUNCTION_UNARY_DOC
    };
    module.add_function(PyCFunction::new_with_keywords(
        py,
        pyo3::impl_::trampoline::get_trampoline_function!(
            cfunction_with_keywords,
            has_torch_function_unary_callback
        ),
        c"_has_torch_function_unary",
        doc,
        Some(module),
    )?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_has_torch_function_unary",))?;
    Ok(())
}
