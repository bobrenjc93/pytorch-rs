//! Stable-ABI construction for PyTorch-style top-level built-in functions.
//!
//! `PyO3`'s `frozen` class option protects Rust instance state, not the Python
//! type dictionary. Its `immutable_type` option needs Python 3.14 when built
//! for the stable ABI. Constructing this zero-state owner with
//! `PyType_FromSpec` keeps the package's abi3-py310 contract while preventing
//! callers from replacing methods that built-in-function pickles depend on.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{exceptions::PyRuntimeError, ffi};

use crate::python::{
    get_device_variable_function, permute_variable_function, positive_variable_function,
    scalar_tensor_variable_function,
};

const POSITIVE_DOC: &std::ffi::CStr = c"\npositive(input) -> Tensor\n\nReturns :attr:`input`.\nThrows a runtime error if :attr:`input` is a bool tensor.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> t = torch.randn(5)\n    >>> t\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n    >>> torch.positive(t)\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n";

const PERMUTE_DOC: &std::ffi::CStr = c"\npermute(input, dims) -> Tensor\n\nReturns a view of the original tensor :attr:`input` with its dimensions permuted.\n\nArgs:\n    input (Tensor): the input tensor.\n    dims (torch.Size, tuple of int or list of int): the desired ordering of dimensions.\n\nExample:\n    >>> x = torch.randn(2, 3, 5)\n    >>> x.size()\n    torch.Size([2, 3, 5])\n    >>> torch.permute(x, (2, 0, 1)).size()\n    torch.Size([5, 2, 3])\n";

#[allow(
    unsafe_code,
    reason = "CPython passes borrowed tuple and dictionary pointers to C method callbacks"
)]
unsafe fn call_arguments(
    py: Python<'_>,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<(Bound<'_, PyTuple>, Option<Bound<'_, PyDict>>)> {
    // SAFETY: the C method callback contract supplies a live positional tuple.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is either null or a live dictionary for the
    // duration of the callback.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    Ok((args, kwargs))
}

macro_rules! variable_function_callback {
    ($name:ident, $implementation:ident) => {
        #[allow(
            unsafe_code,
            reason = "the callback is entered through PyO3's panic-safe C trampoline"
        )]
        unsafe fn $name(
            py: Python<'_>,
            _owner: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
            kwargs: *mut ffi::PyObject,
        ) -> PyResult<*mut ffi::PyObject> {
            // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
            let (args, kwargs) = unsafe { call_arguments(py, args, kwargs) }?;
            $implementation(py, &args, kwargs.as_ref()).map(Py::into_ptr)
        }
    };
}

variable_function_callback!(get_device_callback, get_device_variable_function);
variable_function_callback!(scalar_tensor_callback, scalar_tensor_variable_function);
variable_function_callback!(positive_callback, positive_variable_function);
variable_function_callback!(permute_callback, permute_variable_function);

/// Creates the immutable owner for exported `_VariableFunctionsClass` methods.
///
/// # Errors
///
/// Returns a Python exception if the stable-ABI type constructor fails.
#[allow(
    unsafe_code,
    reason = "PyType_FromSpec requires an audited raw type specification"
)]
pub(crate) fn create_variable_functions_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // CPython descriptors retain pointers to their method definitions. Leak
    // this tiny table deliberately so it remains valid for the type lifetime.
    let methods = Box::leak(Box::new([
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"get_device",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                get_device_callback
            ),
            c"",
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"scalar_tensor",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                scalar_tensor_callback
            ),
            c"",
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"positive",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                positive_callback
            ),
            POSITIVE_DOC,
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"permute",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                permute_callback
            ),
            PERMUTE_DOC,
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        ffi::PyMethodDef::zeroed(),
    ]));
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_methods,
            pfunc: methods.as_mut_ptr().cast::<c_void>(),
        },
        ffi::PyType_Slot::default(),
    ];
    let flags = ffi::Py_TPFLAGS_DEFAULT
        | ffi::Py_TPFLAGS_DISALLOW_INSTANTIATION
        | ffi::Py_TPFLAGS_IMMUTABLETYPE;
    let flags = flags
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("variable-function type flags exceed unsigned int"))?;
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs._C._VariableFunctionsClass".as_ptr(),
        basicsize: 0,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the specification and terminated slot and method tables remain
    // live for the call, and CPython returns a new reference or sets an error.
    let owner = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))?
    };
    Ok(owner.unbind())
}
