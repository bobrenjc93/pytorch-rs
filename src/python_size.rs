//! Stable-ABI construction for the immutable `torch.Size` tuple subtype.

use std::ffi::{CStr, c_int, c_void};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyBool, PyDict, PyInt, PyModule, PyString, PyTuple, PyType};

static SIZE_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static OPERATOR_INDEX: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const NUMEL_DOC: &CStr = c"\nnumel() -> int\n\nReturns the number of elements a :class:`torch.Tensor` with the given size would contain.\n";

fn operator_index(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    OPERATOR_INDEX.get_or_try_init(py, || {
        Ok(PyModule::import(py, "operator")?.getattr("index")?.unbind())
    })
}

fn python_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let value_type = value.get_type();
    let name = value_type.name()?.to_string();
    let module = value_type.getattr("__module__")?.extract::<String>()?;
    Ok(if module == "numpy" {
        format!("numpy.{name}")
    } else {
        name
    })
}

fn has_numpy_integer_ancestry(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let value_type = value.get_type();
    for base in value_type.mro().iter() {
        let base = base.cast::<PyType>()?;
        if base.name()? == "integer" && base.getattr("__module__")?.extract::<String>()? == "numpy"
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn preserves_integral_identity(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        return Ok(true);
    }
    has_numpy_integer_ancestry(value)
}

fn normalized_dimension(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    position: usize,
) -> PyResult<Py<PyAny>> {
    if preserves_integral_identity(value)? {
        return Ok(value.clone().unbind());
    }

    match operator_index(py)?.bind(py).call1((value,)) {
        Ok(integer) => Ok(integer.unbind()),
        Err(_) => Err(PyTypeError::new_err(format!(
            "torch.Size() takes an iterable of 'int' (item {position} is '{}')",
            python_type_name(value)?
        ))),
    }
}

fn unpack_long_long(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<i64> {
    operator_index(py)?
        .bind(py)
        .call1((value,))?
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"))
}

fn tuple_from_value<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    Ok(py
        .get_type::<PyTuple>()
        .call1((value,))?
        .cast_into::<PyTuple>()?)
}

fn construct_size(py: Python<'_>, dimensions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    size_type_object(py)?
        .bind(py)
        .call1((dimensions,))
        .map(Bound::unbind)
}

fn size_new(
    py: Python<'_>,
    subtype: &Bound<'_, PyType>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let values = py
        .get_type::<PyTuple>()
        .call(args, kwargs)?
        .cast_into::<PyTuple>()?;
    let mut normalized = Vec::with_capacity(values.len());
    for position in 0..values.len() {
        normalized.push(normalized_dimension(
            py,
            &values.get_item(position)?,
            position,
        )?);
    }
    let normalized = PyTuple::new(py, normalized)?;
    tuple_new_for_subtype(py, subtype, &normalized)
}

#[allow(
    unsafe_code,
    reason = "the stable ABI exposes tuple's native constructor through PyType_GetSlot"
)]
fn tuple_new_for_subtype(
    py: Python<'_>,
    subtype: &Bound<'_, PyType>,
    values: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    // SAFETY: Py_tp_new on the live built-in tuple type has the newfunc
    // signature. The one-element argument tuple and subtype stay live for the
    // call, which returns a new reference or sets a Python exception.
    unsafe {
        let slot = ffi::PyType_GetSlot(py.get_type::<PyTuple>().as_type_ptr(), ffi::Py_tp_new);
        if slot.is_null() {
            return Err(PyRuntimeError::new_err(
                "built-in tuple type does not expose its constructor slot",
            ));
        }
        let tuple_new: ffi::newfunc = std::mem::transmute(slot);
        let arguments = PyTuple::new(py, [values])?;
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            tuple_new(
                subtype.as_type_ptr(),
                arguments.as_ptr(),
                std::ptr::null_mut(),
            ),
        )
        .map(Bound::unbind)
    }
}

fn size_repr(py: Python<'_>, value: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let mut representation = String::from("torch.Size([");
    for position in 0..value.len() {
        if position != 0 {
            representation.push_str(", ");
        }
        representation.push_str(&unpack_long_long(py, &value.get_item(position)?)?.to_string());
    }
    representation.push_str("])");
    Ok(PyString::new(py, &representation).into_any().unbind())
}

fn size_concat(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if right.cast::<PyTuple>().is_err() {
        return Err(PyTypeError::new_err(format!(
            "can only concatenate tuple (not {}) to torch.Size",
            python_type_name(right)?
        )));
    }
    let concatenated = py
        .get_type::<PyTuple>()
        .getattr("__add__")?
        .call1((left, right))?;
    construct_size(py, &concatenated)
}

fn size_add(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if left.cast::<PyTuple>().is_err() || right.cast::<PyTuple>().is_err() {
        return Ok(py.NotImplemented());
    }
    size_concat(py, left, right)
}

fn size_repeat(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    count: ffi::Py_ssize_t,
) -> PyResult<Py<PyAny>> {
    let repeated = py
        .get_type::<PyTuple>()
        .getattr("__mul__")?
        .call1((value, count))?;
    construct_size(py, &repeated)
}

fn size_subscript(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let result = py
        .get_type::<PyTuple>()
        .getattr("__getitem__")?
        .call1((value, key))?;
    if result.cast::<PyTuple>().is_ok() {
        construct_size(py, &result)
    } else {
        Ok(result.unbind())
    }
}

fn size_numel(py: Python<'_>, value: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let mut result = 1_i64;
    for position in 0..value.len() {
        result = result.wrapping_mul(unpack_long_long(py, &value.get_item(position)?)?);
    }
    result.into_py_any(py)
}

fn size_reduce(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dimensions = tuple_from_value(py, value)?;
    let arguments = PyTuple::new(py, [dimensions.into_any()])?;
    PyTuple::new(py, [value.get_type().into_any(), arguments.into_any()])
        .map(|reduction| reduction.into_any().unbind())
}

#[allow(
    unsafe_code,
    reason = "CPython supplies borrowed constructor arguments to the panic-safe PyO3 trampoline"
)]
unsafe fn size_new_callback(
    py: Python<'_>,
    subtype: *mut ffi::PyTypeObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the constructor callback supplies a live subtype and positional tuple.
    let subtype =
        unsafe { Bound::<PyAny>::from_borrowed_ptr(py, subtype.cast()) }.cast_into::<PyType>()?;
    // SAFETY: CPython owns the positional tuple for the duration of the callback.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is null or a live dictionary.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    size_new(py, &subtype, &args, kwargs.as_ref()).map(Py::into_ptr)
}

macro_rules! borrowed_binary_callback {
    ($callback:ident, $implementation:ident) => {
        #[allow(
            unsafe_code,
            reason = "CPython supplies borrowed operands to the panic-safe PyO3 trampoline"
        )]
        unsafe fn $callback(
            py: Python<'_>,
            left: *mut ffi::PyObject,
            right: *mut ffi::PyObject,
        ) -> PyResult<*mut ffi::PyObject> {
            // SAFETY: both operands are live for the duration of the slot call.
            let left = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, left) };
            // SAFETY: both operands are live for the duration of the slot call.
            let right = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, right) };
            $implementation(py, &left, &right).map(Py::into_ptr)
        }
    };
}

borrowed_binary_callback!(size_add_callback, size_add);
borrowed_binary_callback!(size_concat_callback, size_concat);
borrowed_binary_callback!(size_subscript_callback, size_subscript);

#[allow(
    unsafe_code,
    reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
)]
unsafe fn size_repr_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the repr slot supplies a live Size, which is a tuple subtype.
    let value = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, value) }.cast_into::<PyTuple>()?;
    size_repr(py, &value).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
)]
unsafe fn size_repeat_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
    count: ffi::Py_ssize_t,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the repeat slot supplies a live Size instance.
    let value = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, value) };
    size_repeat(py, &value, count).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
)]
unsafe fn size_numel_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the method owner is a live Size, which is a tuple subtype.
    let value = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, value) }.cast_into::<PyTuple>()?;
    size_numel(py, &value).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
)]
unsafe fn size_reduce_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the method owner is live for the duration of the callback.
    let value = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, value) };
    size_reduce(py, &value).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies borrowed instance and attribute pointers to the trampoline"
)]
unsafe fn size_setattr_callback(
    py: Python<'_>,
    _value: *mut ffi::PyObject,
    name: *mut ffi::PyObject,
    _assigned: *mut ffi::PyObject,
) -> PyResult<c_int> {
    // SAFETY: the attribute-name pointer is live for the duration of the callback.
    let name = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, name) }.extract::<String>()?;
    Err(PyAttributeError::new_err(format!(
        "'torch.Size' object has no attribute '{name}'"
    )))
}

#[allow(
    unsafe_code,
    reason = "CPython supplies borrowed item-assignment pointers to the trampoline"
)]
unsafe fn size_setitem_callback(
    _py: Python<'_>,
    _value: *mut ffi::PyObject,
    _key: *mut ffi::PyObject,
    _assigned: *mut ffi::PyObject,
) -> PyResult<c_int> {
    Err(PyTypeError::new_err(
        "'torch.Size' object does not support item assignment",
    ))
}

#[allow(
    unsafe_code,
    reason = "PyType_FromSpecWithBases requires an audited stable-ABI raw-pointer call"
)]
fn create_size_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let methods = Box::leak(Box::new([
        pyo3::impl_::pymethods::PyMethodDef::noargs(
            c"numel",
            pyo3::impl_::trampoline::get_trampoline_function!(noargs, size_numel_callback),
            NUMEL_DOC,
        )
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::noargs(
            c"__reduce__",
            pyo3::impl_::trampoline::get_trampoline_function!(noargs, size_reduce_callback),
            c"",
        )
        .into_raw(),
        ffi::PyMethodDef::zeroed(),
    ]));
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_new,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(newfunc, size_new_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_repr,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(reprfunc, size_repr_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_nb_add,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(binaryfunc, size_add_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_sq_concat,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                binaryfunc,
                size_concat_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_sq_repeat,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                ssizeargfunc,
                size_repeat_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_mp_subscript,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                binaryfunc,
                size_subscript_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_setattro,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                setattrofunc,
                size_setattr_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_mp_ass_subscript,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                setattrofunc,
                size_setitem_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_methods,
            pfunc: methods.as_mut_ptr().cast::<c_void>(),
        },
        ffi::PyType_Slot::default(),
    ];
    let flags = ffi::Py_TPFLAGS_DEFAULT | ffi::Py_TPFLAGS_IMMUTABLETYPE;
    let flags = flags
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("Size type flags exceed unsigned int"))?;
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs.Size".as_ptr(),
        // Zero inherits tuple's variable-size native layout.
        basicsize: 0,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };
    let bases = PyTuple::new(py, [py.get_type::<PyTuple>()])?;

    // SAFETY: the terminated slot table remains live for the call, the method
    // table is leaked for the type lifetime, every callback uses the matching
    // CPython signature, and the sole base is the live built-in tuple type.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyType_FromSpecWithBases(&raw mut specification, bases.as_ptr()),
        )
        .map(Bound::unbind)
    }
}

/// Returns the process-stable immutable Python `Size` type.
pub(crate) fn size_type_object(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    SIZE_TYPE.get_or_try_init(py, || create_size_type(py))
}
