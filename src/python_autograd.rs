//! Native bindings owned by `PyTorch`'s private autograd module.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyString, PyTuple};

const NATIVE_KINETO_AVAILABLE: bool = false;
const KINETO_AVAILABLE_DOC: &std::ffi::CStr = c"kineto_available() -> bool\n";
#[cfg(target_os = "macos")]
const FUNCTION_RECORD_NAME: &str = "pybind11_detail_function_record_v1_system_libcpp_abi1";
#[cfg(target_os = "windows")]
const FUNCTION_RECORD_NAME: &str = "pybind11_detail_function_record_v1_msvc_md_mscver19";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const FUNCTION_RECORD_NAME: &str =
    "pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1";

// pybind11 binds module functions to an internal function-record object. Keep
// the same observable callable shape and make that owner reduce to the native
// module, which is what makes copied and pickled functions resolve canonically.
#[cfg_attr(
    target_os = "macos",
    pyclass(
        frozen,
        module = "pybind11_builtins",
        name = "pybind11_detail_function_record_v1_system_libcpp_abi1"
    )
)]
#[cfg_attr(
    target_os = "windows",
    pyclass(
        frozen,
        module = "pybind11_builtins",
        name = "pybind11_detail_function_record_v1_msvc_md_mscver19"
    )
)]
#[cfg_attr(
    not(any(target_os = "macos", target_os = "windows")),
    pyclass(
        frozen,
        module = "pybind11_builtins",
        name = "pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1"
    )
)]
struct KinetoAvailableFunctionRecord;

#[pymethods]
impl KinetoAvailableFunctionRecord {
    #[allow(clippy::unused_self)]
    fn __reduce__(&self) -> PyResult<Py<PyAny>> {
        Err(PyTypeError::new_err(format!(
            "cannot pickle '{FUNCTION_RECORD_NAME}' object"
        )))
    }

    #[allow(clippy::unused_self)]
    fn __reduce_ex__(&self, py: Python<'_>, _protocol: i32) -> PyResult<Py<PyAny>> {
        let eval = PyModule::import(py, "builtins")?.getattr("eval")?;
        (
            eval,
            ("__import__('importlib').import_module('torch_rs._C._autograd')",),
        )
            .into_py_any(py)
    }
}

fn safe_repr(value: &Bound<'_, PyAny>) -> String {
    value.repr().map_or_else(
        |_| "<repr raised Error>".to_owned(),
        |representation| representation.to_string(),
    )
}

fn format_invocation(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let positional = args
        .iter()
        .map(|value| safe_repr(&value))
        .collect::<Vec<_>>()
        .join(", ");
    let keywords = if let Some(kwargs) = kwargs {
        kwargs
            .iter()
            .map(|(name, value)| {
                Ok(format!(
                    "{}={}",
                    name.extract::<String>()?,
                    safe_repr(&value)
                ))
            })
            .collect::<PyResult<Vec<_>>>()?
            .join(", ")
    } else {
        String::new()
    };

    Ok(match (positional.is_empty(), keywords.is_empty()) {
        (false, false) => format!("{positional}; kwargs: {keywords}"),
        (false, true) => positional,
        (true, false) => format!("kwargs: {keywords}"),
        (true, true) => String::new(),
    })
}

fn kineto_available(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if !args.is_empty() || kwargs.is_some_and(|values| !values.is_empty()) {
        let invocation = format_invocation(args, kwargs)?;
        return Err(PyTypeError::new_err(format!(
            "kineto_available(): incompatible function arguments. The following argument types are supported:\n    1. () -> bool\n\nInvoked with: {invocation}"
        )));
    }
    Ok(NATIVE_KINETO_AVAILABLE)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn kineto_available_callback(
    py: Python<'_>,
    _owner: *mut ffi::PyObject,
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
    reason = "PyCFunction_NewEx requires an audited raw method definition and live Python owners"
)]
fn create_kineto_available<'py>(
    py: Python<'py>,
    owner: &Bound<'py, KinetoAvailableFunctionRecord>,
    module_name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    // CPython retains the method-definition pointer for the callable's full
    // lifetime, so this one-entry definition is intentionally leaked.
    let definition = Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: c"kineto_available".as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionWithKeywords: pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                kineto_available_callback
            ),
        },
        ml_flags: ffi::METH_VARARGS | ffi::METH_KEYWORDS,
        ml_doc: KINETO_AVAILABLE_DOC.as_ptr(),
    }));

    // SAFETY: the method definition is leaked above, the owner and module name
    // are live Python objects, and CPython returns a new reference or sets an
    // exception.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, owner.as_ptr(), module_name.as_ptr()),
        )
    }
}

/// Adds the private native autograd module and its immutable Kineto capability.
///
/// # Errors
///
/// Returns a Python exception if module or callable initialization fails.
pub(crate) fn add_autograd_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let module_name = PyString::new(py, "torch_rs._C._autograd");
    let autograd = PyModule::new(py, module_name.to_str()?)?;
    autograd.setattr("__doc__", "autograd bindings")?;
    if let Ok(file) = module.getattr("__file__") {
        autograd.setattr("__file__", file)?;
    }

    let owner = Py::new(py, KinetoAvailableFunctionRecord)?;
    let function = create_kineto_available(py, owner.bind(py), &module_name)?;
    autograd.setattr("kineto_available", function)?;
    module.add("_autograd", &autograd)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_autograd",))?;

    let modules = PyModule::import(py, "sys")?.getattr("modules")?;
    modules.set_item("torch_rs._C._autograd", &autograd)?;
    Ok(())
}
