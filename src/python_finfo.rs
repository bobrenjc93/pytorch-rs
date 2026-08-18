//! Stable-ABI construction for immutable native floating-point metadata.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem::size_of;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyDict, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, ffi};

use crate::{
    DType,
    dtype::FloatingPointInfo,
    python::{
        legacy_dict_get_item_string, native_pytorch_type_name, pytorch_ordered_keyword_entries,
    },
    python_dtype::PyDType,
};

const INVALID_COMBINATION_PREFIX: &str =
    "finfo() received an invalid combination of arguments - got (";
const INVALID_COMBINATION_SUFFIX: &str = "), but expected one of:\n * (torch.dtype type)\n * ()\n";
const FLOAT32_CODE: c_int = 0;

static FINFO_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[repr(C)]
struct FInfoObject {
    object: ffi::PyObject,
    dtype: c_int,
}

#[repr(C)]
struct PyTypeObjectNamePrefix {
    _ob_base: ffi::PyVarObject,
    tp_name: *const c_char,
}

fn allocation_error() -> PyErr {
    PyRuntimeError::new_err("std::bad_alloc")
}

fn add_message_capacity(capacity: &mut usize, additional: usize) -> PyResult<()> {
    *capacity = capacity
        .checked_add(additional)
        .ok_or_else(allocation_error)?;
    Ok(())
}

fn try_push_message(message: &mut String, value: &str) -> PyResult<()> {
    message
        .try_reserve(value.len())
        .map_err(|_| allocation_error())?;
    message.push_str(value);
    Ok(())
}

#[allow(
    unsafe_code,
    reason = "CPython exposes the non-overridable tp_name in every live type-object prefix"
)]
fn cpython_type_name<'a>(value: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    let value_type = value.get_type();
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: every classic CPython type object starts with PyVarObject and
    // tp_name. The value keeps its type alive while the borrowed name is used.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: CPython keeps tp_name NUL-terminated and alive with the type.
    unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| PyRuntimeError::new_err("Python tp_name is not valid UTF-8"))
}

fn constructor_type_name<'a>(value: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    if let Some(name) = native_pytorch_type_name(value) {
        return Ok(name);
    }

    let name = cpython_type_name(value)?;
    Ok(match name {
        "torch_rs.dtype" => "torch.dtype",
        "torch_rs.device" => "torch.device",
        "torch_rs.layout" => "torch.layout",
        "torch_rs.memory_format" => "torch.memory_format",
        "torch_rs.Size" => "torch.Size",
        "torch_rs.finfo" => "torch.finfo",
        _ => name,
    })
}

fn invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let keyword_arguments =
        keywords.map_or_else(|| Ok(Vec::new()), pytorch_ordered_keyword_entries)?;

    let mut capacity = INVALID_COMBINATION_PREFIX.len();
    let mut has_argument = false;
    for value in positional {
        if has_argument {
            add_message_capacity(&mut capacity, ", ".len())?;
        }
        add_message_capacity(&mut capacity, constructor_type_name(&value)?.len())?;
        has_argument = true;
    }
    for (key, value) in &keyword_arguments {
        if has_argument {
            add_message_capacity(&mut capacity, ", ".len())?;
        }
        add_message_capacity(&mut capacity, key.len())?;
        add_message_capacity(&mut capacity, "=".len())?;
        add_message_capacity(&mut capacity, constructor_type_name(value)?.len())?;
        has_argument = true;
    }
    if positional.is_empty() && has_argument {
        add_message_capacity(&mut capacity, ", ".len())?;
    }
    add_message_capacity(&mut capacity, INVALID_COMBINATION_SUFFIX.len())?;

    let mut message = String::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    try_push_message(&mut message, INVALID_COMBINATION_PREFIX)?;
    has_argument = false;
    for value in positional {
        if has_argument {
            try_push_message(&mut message, ", ")?;
        }
        try_push_message(&mut message, constructor_type_name(&value)?)?;
        has_argument = true;
    }
    for (key, value) in &keyword_arguments {
        if has_argument {
            try_push_message(&mut message, ", ")?;
        }
        try_push_message(&mut message, key)?;
        try_push_message(&mut message, "=")?;
        try_push_message(&mut message, constructor_type_name(value)?)?;
        has_argument = true;
    }
    if positional.is_empty() && has_argument {
        try_push_message(&mut message, ", ")?;
    }
    try_push_message(&mut message, INVALID_COMBINATION_SUFFIX)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    Ok(PyTypeError::new_err(message))
}

fn bind_constructor<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<(Bound<'py, PyAny>, bool)>> {
    let keyword_count = keywords.map_or(0, pyo3::types::PyDictMethods::len);
    match (positional.len(), keyword_count) {
        (0, 0) => Ok(None),
        (0, 1) => {
            let keywords = keywords.expect("one keyword argument has a dictionary");
            let Some(value) = legacy_dict_get_item_string(keywords, c"type") else {
                return Err(PyTypeError::new_err(
                    "finfo() missing 1 required positional arguments: \"type\"",
                ));
            };
            Ok(Some((value, false)))
        }
        (1, 0) => Ok(Some((positional.get_item(0)?, true))),
        _ => Err(invalid_combination(positional, keywords)?),
    }
}

fn parse_dtype(value: &Bound<'_, PyAny>, positional: bool) -> PyResult<DType> {
    const PREFIX: &str = "finfo(): argument 'type'";
    const POSITION: &str = " (position 1)";
    const INFIX: &str = " must be torch.dtype, not ";

    if let Ok(dtype) = value.cast::<PyDType>() {
        return Ok(dtype.try_borrow()?.inner());
    }

    let name = constructor_type_name(value)?;
    let mut capacity = PREFIX.len();
    if positional {
        add_message_capacity(&mut capacity, POSITION.len())?;
    }
    add_message_capacity(&mut capacity, INFIX.len())?;
    add_message_capacity(&mut capacity, name.len())?;

    let mut message = String::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    try_push_message(&mut message, PREFIX)?;
    if positional {
        try_push_message(&mut message, POSITION)?;
    }
    try_push_message(&mut message, INFIX)?;
    try_push_message(&mut message, name)?;
    Err(PyTypeError::new_err(message))
}

const fn dtype_code(dtype: DType) -> c_int {
    match dtype {
        DType::Float32 => FLOAT32_CODE,
    }
}

const fn dtype_from_code(_code: c_int) -> DType {
    DType::Float32
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a live finfo instance with the audited C layout"
)]
unsafe fn instance_dtype(object: *mut ffi::PyObject) -> DType {
    // SAFETY: every finfo instance is allocated with FInfoObject's basicsize.
    dtype_from_code(unsafe { (*object.cast::<FInfoObject>()).dtype })
}

#[allow(
    unsafe_code,
    reason = "CPython supplies borrowed constructor arguments to the panic-safe PyO3 trampoline"
)]
unsafe fn finfo_new_callback(
    py: Python<'_>,
    subtype: *mut ffi::PyTypeObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the newfunc contract supplies a live positional tuple.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    let kwargs = if kwargs.is_null() {
        None
    } else {
        // SAFETY: a non-null newfunc keyword pointer is a live dictionary.
        Some(unsafe { Bound::<PyAny>::from_borrowed_ptr(py, kwargs) }.cast_into::<PyDict>()?)
    };
    let dtype = bind_constructor(&args, kwargs.as_ref())?
        .map_or(Ok(DType::Float32), |(value, positional)| {
            parse_dtype(&value, positional)
        })?;

    // SAFETY: subtype is the live, non-subclassable finfo type supplied by CPython.
    let object = unsafe { ffi::PyType_GenericAlloc(subtype, 0) };
    if object.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: GenericAlloc reserved the FInfoObject basicsize declared below.
    unsafe { (*object.cast::<FInfoObject>()).dtype = dtype_code(dtype) };
    Ok(object)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a live finfo instance to the panic-safe PyO3 trampoline"
)]
unsafe fn finfo_repr_callback(
    py: Python<'_>,
    object: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: reprfunc is installed only on the finfo type.
    let info = unsafe { instance_dtype(object) }.finfo();
    PyString::from_bytes(py, info.representation().as_bytes())
        .map(|value| value.unbind().into_ptr())
}

#[allow(
    unsafe_code,
    reason = "CPython supplies live comparison operands to the panic-safe PyO3 trampoline"
)]
unsafe fn finfo_richcompare_callback(
    py: Python<'_>,
    object: *mut ffi::PyObject,
    other: *mut ffi::PyObject,
    operation: c_int,
) -> PyResult<*mut ffi::PyObject> {
    if operation != ffi::Py_EQ && operation != ffi::Py_NE {
        return Ok(py.NotImplemented().into_ptr());
    }

    // SAFETY: richcmpfunc is installed only on the finfo type.
    let dtype = unsafe { instance_dtype(object) };
    // SAFETY: both pointers are live comparison operands supplied by CPython.
    let equal = if unsafe { ffi::Py_TYPE(object) == ffi::Py_TYPE(other) } {
        // SAFETY: matching exact types make other another live finfo instance.
        dtype == unsafe { instance_dtype(other) }
    } else {
        // SAFETY: other is borrowed and live for this callback.
        let other = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, other) };
        if let Ok(other) = other.cast::<PyDType>() {
            dtype == other.try_borrow()?.inner()
        } else {
            false
        }
    };
    (if operation == ffi::Py_EQ {
        equal
    } else {
        !equal
    })
    .into_py_any(py)
    .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "each getter reads a live finfo instance and calls a matching stable-ABI constructor"
)]
unsafe fn instance_info(object: *mut ffi::PyObject) -> FloatingPointInfo {
    // SAFETY: getset descriptors invoke this only for live finfo instances.
    unsafe { instance_dtype(object) }.finfo()
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_dtype(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.dtype's getter.
    match unsafe { instance_info(object) }.dtype() {
        // SAFETY: the static C string is NUL-terminated.
        DType::Float32 => unsafe { ffi::PyUnicode_FromString(c"float32".as_ptr()) },
    }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_bits(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.bits's getter.
    unsafe { ffi::PyLong_FromSize_t(instance_info(object).bits()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_eps(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.eps's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).eps()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_max(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.max's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).max()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_min(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.min's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).min()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_resolution(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.resolution's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).resolution()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_smallest_normal(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.smallest_normal's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).smallest_normal()) }
}

#[allow(unsafe_code, reason = "stable-ABI read-only getset callback")]
unsafe extern "C" fn get_tiny(
    object: *mut ffi::PyObject,
    _closure: *mut c_void,
) -> *mut ffi::PyObject {
    // SAFETY: this callback is installed only as finfo.tiny's getter.
    unsafe { ffi::PyFloat_FromDouble(instance_info(object).smallest_normal()) }
}

fn getset(name: &'static std::ffi::CStr, getter: ffi::getter) -> ffi::PyGetSetDef {
    ffi::PyGetSetDef {
        name: name.as_ptr(),
        get: Some(getter),
        set: None,
        doc: std::ptr::null(),
        closure: std::ptr::null_mut(),
    }
}

#[allow(
    unsafe_code,
    reason = "PyType_FromSpec requires an audited stable-ABI type specification"
)]
fn create_finfo_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let getsets = Box::leak(Box::new([
        getset(c"dtype", get_dtype),
        getset(c"bits", get_bits),
        getset(c"eps", get_eps),
        getset(c"max", get_max),
        getset(c"min", get_min),
        getset(c"resolution", get_resolution),
        getset(c"smallest_normal", get_smallest_normal),
        getset(c"tiny", get_tiny),
        ffi::PyGetSetDef::default(),
    ]));
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_new,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(newfunc, finfo_new_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_repr,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(reprfunc, finfo_repr_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_str,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(reprfunc, finfo_repr_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_hash,
            pfunc: ffi::PyObject_HashNotImplemented as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_richcompare,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                richcmpfunc,
                finfo_richcompare_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_getset,
            pfunc: getsets.as_mut_ptr().cast::<c_void>(),
        },
        ffi::PyType_Slot::default(),
    ];
    let flags = ffi::Py_TPFLAGS_DEFAULT | ffi::Py_TPFLAGS_IMMUTABLETYPE;
    let flags = flags
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("finfo type flags exceed unsigned int"))?;
    let basicsize = c_int::try_from(size_of::<FInfoObject>())
        .map_err(|_| PyRuntimeError::new_err("finfo instance size exceeds C int"))?;
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs.finfo".as_ptr(),
        basicsize,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the terminated slot tables remain live for the call, the method
    // and getset tables are leaked for the type lifetime, and every callback
    // uses the signature required by its slot.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))
            .map(Bound::unbind)
    }
}

/// Returns the process-stable immutable Python `finfo` type.
pub(crate) fn finfo_type_object(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    FINFO_TYPE.get_or_try_init(py, || create_finfo_type(py))
}
