//! Native autograd capability bindings without profiler execution support.

use std::{ffi::c_void, mem};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyModule, PyString, PyTuple, PyType};

const NATIVE_KINETO_AVAILABLE: bool = false;
const NATIVE_AUTOGRAD_MODULE_NAME: &str = "torch_rs._C._autograd";
const NATIVE_AUTOGRAD_RECONSTRUCTION: &str =
    "__import__('importlib').import_module('torch_rs._C._autograd')";
const KINETO_AVAILABLE_DOC: &std::ffi::CStr = c"kineto_available() -> bool\n";

// PyTorch 2.13 exposes pybind11's internal function-record owner in the
// metadata and repr of torch._C._autograd.kineto_available. Mirror the
// platform ABI spelling without linking pybind11 or importing PyTorch.
#[cfg(target_os = "linux")]
const FUNCTION_RECORD_TYPE_NAME: &std::ffi::CStr = c"pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1";
#[cfg(target_os = "macos")]
const FUNCTION_RECORD_TYPE_NAME: &std::ffi::CStr =
    c"pybind11_builtins.pybind11_detail_function_record_v1_system_libcpp_abi1";
#[cfg(target_os = "windows")]
const FUNCTION_RECORD_TYPE_NAME: &std::ffi::CStr =
    c"pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const FUNCTION_RECORD_TYPE_NAME: &std::ffi::CStr =
    c"pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1";

fn invocation_repr(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let mut positional = Vec::with_capacity(args.len());
    for argument in args.iter() {
        positional.push(argument.repr()?.to_string_lossy().into_owned());
    }

    let mut keywords = Vec::with_capacity(kwargs.map_or(0, PyDictMethods::len));
    if let Some(kwargs) = kwargs {
        for (name, value) in kwargs.iter() {
            keywords.push(format!(
                "{}={}",
                name.extract::<String>()?,
                value.repr()?.to_string_lossy()
            ));
        }
    }

    match (positional.is_empty(), keywords.is_empty()) {
        (false, false) => Ok(format!(
            "{}; kwargs: {}",
            positional.join(", "),
            keywords.join(", ")
        )),
        (false, true) => Ok(positional.join(", ")),
        (true, false) => Ok(format!("kwargs: {}", keywords.join(", "))),
        (true, true) => Ok(String::new()),
    }
}

fn kineto_available(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if args.is_empty() && kwargs.is_none_or(PyDictMethods::is_empty) {
        return Ok(NATIVE_KINETO_AVAILABLE);
    }

    let invocation = invocation_repr(args, kwargs)?;
    Err(PyTypeError::new_err(format!(
        "kineto_available(): incompatible function arguments. The following argument types are supported:\n    1. () -> bool\n\nInvoked with: {invocation}"
    )))
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn kineto_available_callback(
    py: Python<'_>,
    _function_record: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is either null or a live dictionary for the
    // duration of the callback.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    kineto_available(&args, kwargs.as_ref())?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn function_record_reduce_ex_callback(
    py: Python<'_>,
    _function_record: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
    _kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    let evaluate = PyModule::import(py, "builtins")?.getattr("eval")?;
    (evaluate, (NATIVE_AUTOGRAD_RECONSTRUCTION,))
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "PyType_FromSpec and PyCFunction_NewEx require audited raw specifications"
)]
fn create_kineto_available(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let reduce_methods = Box::leak(Box::new([
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"__reduce_ex__",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                function_record_reduce_ex_callback
            ),
            c"",
        )
        .into_raw(),
        ffi::PyMethodDef::zeroed(),
    ]));
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_methods,
            pfunc: reduce_methods.as_mut_ptr().cast::<c_void>(),
        },
        ffi::PyType_Slot::default(),
    ];
    let basicsize = mem::size_of::<ffi::PyObject>()
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("function-record size exceeds C int"))?;
    let flags = (ffi::Py_TPFLAGS_DEFAULT
        | ffi::Py_TPFLAGS_DISALLOW_INSTANTIATION
        | ffi::Py_TPFLAGS_IMMUTABLETYPE)
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("function-record flags exceed unsigned int"))?;
    let mut specification = ffi::PyType_Spec {
        name: FUNCTION_RECORD_TYPE_NAME.as_ptr(),
        basicsize,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the specification and terminated method table remain live for
    // the call, and CPython returns a new reference or sets an exception.
    let function_record_type = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))?
    }
    .cast_into::<PyType>()?;
    // SAFETY: GenericAlloc creates a zero-initialized instance of the live
    // heap type and returns a new reference or sets an exception.
    let function_record = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyType_GenericAlloc(function_record_type.as_type_ptr(), 0),
        )?
    };

    let function_definition = Box::leak(Box::new(
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"kineto_available",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                kineto_available_callback
            ),
            KINETO_AVAILABLE_DOC,
        )
        .into_raw(),
    ));
    let module_name = PyString::new(py, NATIVE_AUTOGRAD_MODULE_NAME);
    // SAFETY: the leaked method definition, live function-record owner, and
    // module-name string satisfy PyCFunction_NewEx's lifetime contract.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(
                function_definition,
                function_record.as_ptr(),
                module_name.as_ptr(),
            ),
        )
    }
}

/// Adds the deliberately minimal native autograd capability namespace.
///
/// # Errors
///
/// Returns a Python exception if native module or callable construction fails.
pub(crate) fn add_autograd_capabilities(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let native_autograd = PyModule::new(py, NATIVE_AUTOGRAD_MODULE_NAME)?;
    native_autograd.setattr("__doc__", "autograd bindings")?;
    if let Ok(filename) = module.getattr("__file__") {
        native_autograd.setattr("__file__", filename)?;
    }
    native_autograd.setattr("kineto_available", create_kineto_available(py)?)?;

    let modules = PyModule::import(py, "sys")?.getattr("modules")?;
    modules.set_item(NATIVE_AUTOGRAD_MODULE_NAME, &native_autograd)?;
    modules.set_item("torch_rs.torch_rs._autograd", &native_autograd)?;

    module.add_submodule(&native_autograd)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_autograd",))?;
    Ok(())
}
