//! Stable-ABI construction for the immutable `torch.Size` tuple subtype.

use std::ffi::{CStr, c_char, c_void};
use std::fmt::Write as _;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyBool, PyDict, PyInt, PyString, PyTuple, PyType};

use crate::python::native_pytorch_type_name;

static SIZE_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const NUMEL_DOC: &CStr = c"\nnumel() -> int\n\nReturns the number of elements a :class:`torch.Tensor` with the given size would contain.\n";

#[repr(C)]
struct PyTypeObjectNamePrefix {
    _ob_base: ffi::PyVarObject,
    tp_name: *const c_char,
}

#[allow(
    unsafe_code,
    reason = "CPython exposes tp_name only as a type-object field before Python 3.13"
)]
fn type_object_name<'a>(value_type: &'a Bound<'_, PyType>) -> PyResult<&'a CStr> {
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: every classic CPython type object starts with PyVarObject and
    // tp_name. The attached interpreter keeps the live type object stable
    // while the non-overridable C name is inspected.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: CPython requires tp_name to remain a NUL-terminated string for
    // the lifetime of the type object.
    Ok(unsafe { CStr::from_ptr(name) })
}

fn python_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let value_type = value.get_type();
    let native_name = native_pytorch_type_name(value);
    let cpython_name = type_object_name(&value_type)?;
    let name = native_name
        .or_else(|| {
            if !is_immutable_type(&value_type) {
                return None;
            }
            match cpython_name.to_bytes() {
                b"torch_rs.layout" => Some("torch.layout"),
                b"torch_rs.Size" => Some("torch.Size"),
                _ => None,
            }
        })
        .map_or_else(
            || {
                cpython_name
                    .to_str()
                    .map_err(|_| PyRuntimeError::new_err("Python tp_name is not valid UTF-8"))
            },
            Ok,
        )?;
    let mut output = String::new();
    output
        .try_reserve_exact(name.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate Python type name"))?;
    output.push_str(name);
    Ok(output)
}

fn dimension_type_error(value: &Bound<'_, PyAny>, position: usize) -> PyResult<PyErr> {
    const PREFIX: &str = "torch.Size() takes an iterable of 'int' (item ";
    const INFIX: &str = " is '";
    const SUFFIX: &str = "')";

    let name = python_type_name(value)?;
    let capacity = PREFIX
        .len()
        .checked_add(usize::BITS as usize)
        .and_then(|length| length.checked_add(INFIX.len()))
        .and_then(|length| length.checked_add(name.len()))
        .and_then(|length| length.checked_add(SUFFIX.len()))
        .ok_or_else(|| PyMemoryError::new_err("torch.Size type error is too large"))?;
    let mut message = String::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| PyMemoryError::new_err("unable to allocate torch.Size type error"))?;
    message.push_str(PREFIX);
    write!(message, "{position}")
        .map_err(|_| PyRuntimeError::new_err("unable to format torch.Size item position"))?;
    message.push_str(INFIX);
    message.push_str(&name);
    message.push_str(SUFFIX);
    Ok(PyTypeError::new_err(message))
}

fn concatenation_type_error(value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    const PREFIX: &str = "can only concatenate tuple (not ";
    const SUFFIX: &str = ") to torch.Size";

    let name = python_type_name(value)?;
    let capacity = PREFIX
        .len()
        .checked_add(name.len())
        .and_then(|length| length.checked_add(SUFFIX.len()))
        .ok_or_else(|| PyMemoryError::new_err("torch.Size concatenation error is too large"))?;
    let mut message = String::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|_| PyMemoryError::new_err("unable to allocate torch.Size concatenation error"))?;
    message.push_str(PREFIX);
    message.push_str(&name);
    message.push_str(SUFFIX);
    Ok(PyTypeError::new_err(message))
}

#[allow(
    unsafe_code,
    reason = "PyType_GetFlags reads immutable flags from a live type through the stable ABI"
)]
fn is_native_immutable_type(value_type: &Bound<'_, PyType>) -> bool {
    // SAFETY: value_type is a live Python type object for the duration of the call.
    let flags = unsafe { ffi::PyType_GetFlags(value_type.as_type_ptr()) };
    flags & ffi::Py_TPFLAGS_IMMUTABLETYPE != 0 && flags & ffi::Py_TPFLAGS_HEAPTYPE == 0
}

#[allow(
    unsafe_code,
    reason = "PyType_GetFlags reads immutable flags from a live type through the stable ABI"
)]
fn is_immutable_type(value_type: &Bound<'_, PyType>) -> bool {
    // SAFETY: value_type is a live Python type object for the duration of the call.
    (unsafe { ffi::PyType_GetFlags(value_type.as_type_ptr()) }) & ffi::Py_TPFLAGS_IMMUTABLETYPE != 0
}

pub(crate) fn has_numpy_integer_ancestry(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // Calling type's base descriptor bypasses any metaclass override of
    // __getattribute__, and __mro__ itself is immutable.
    let mro = py
        .get_type::<PyType>()
        .getattr("__getattribute__")?
        .call1((value.get_type(), "__mro__"))?
        .cast_into::<PyTuple>()?;
    for base in mro.iter() {
        let base = base.cast_into::<PyType>()?;
        if is_native_immutable_type(&base) && type_object_name(&base)? == c"numpy.integer" {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(
    unsafe_code,
    reason = "PyNumber_Index returns a new reference through the stable CPython ABI"
)]
fn number_index<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
    // SAFETY: value is live for the call. PyNumber_Index returns a new Python
    // int reference or sets an exception and returns null.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyNumber_Index(value.as_ptr()))?
            .cast_into::<PyInt>()
            .map_err(Into::into)
    }
}

#[allow(
    unsafe_code,
    reason = "PyTuple_New and PyTuple_SetItem provide fallible tuple construction through the stable ABI"
)]
fn fallible_tuple_from_iter(
    py: Python<'_>,
    values: impl ExactSizeIterator<Item = Py<PyAny>>,
) -> PyResult<Bound<'_, PyTuple>> {
    let length = ffi::Py_ssize_t::try_from(values.len())
        .map_err(|_| PyMemoryError::new_err("torch.Size tuple is too large"))?;
    // SAFETY: PyTuple_New returns a new reference or sets a Python exception.
    let tuple = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyTuple_New(length))?
            .cast_into::<PyTuple>()?
    };
    let mut position = 0;
    for value in values {
        // SAFETY: tuple is a new, uniquely owned tuple and position remains in
        // the exact-size iterator's declared range. PyTuple_SetItem steals the
        // value reference even when it reports an error.
        let status = unsafe { ffi::PyTuple_SetItem(tuple.as_ptr(), position, value.into_ptr()) };
        if status == -1 {
            return Err(PyErr::fetch(py));
        }
        position += 1;
    }
    if position != length {
        return Err(PyRuntimeError::new_err(
            "torch.Size tuple iterator changed length during construction",
        ));
    }
    Ok(tuple)
}

#[allow(
    unsafe_code,
    reason = "PyTuple_SetItem replaces a uniquely owned Size slot through the stable ABI"
)]
fn replace_size_item(
    py: Python<'_>,
    value: &Bound<'_, PyTuple>,
    position: ffi::Py_ssize_t,
    replacement: Py<PyAny>,
) -> PyResult<()> {
    // SAFETY: value is a newly allocated, uniquely owned tuple subtype and
    // position is within its initialized item range. PyTuple_SetItem steals
    // the replacement reference even if it reports an error.
    let status = unsafe { ffi::PyTuple_SetItem(value.as_ptr(), position, replacement.into_ptr()) };
    if status == -1 {
        return Err(PyErr::fetch(py));
    }
    Ok(())
}

fn normalized_dimension(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    position: usize,
) -> PyResult<Py<PyAny>> {
    if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        return Ok(value.clone().unbind());
    }
    if has_numpy_integer_ancestry(py, value)? {
        return Ok(value.clone().unbind());
    }
    let Ok(integer) = number_index(py, value) else {
        return Err(dimension_type_error(value, position)?);
    };
    Ok(integer.into_any().unbind())
}

fn unpack_long_long(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<i64> {
    number_index(py, value)?
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

pub(crate) fn construct_size(py: Python<'_>, dimensions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
    let kwargs = if py.version_info() < (3, 11) {
        None
    } else {
        kwargs
    };
    let values = py
        .get_type::<PyTuple>()
        .call(args, kwargs)?
        .cast_into::<PyTuple>()?;
    let size = tuple_new_for_subtype(py, subtype, &values)?;
    drop(values);

    let size_tuple = size.bind(py).cast::<PyTuple>()?;
    for position in 0..size_tuple.len() {
        let normalized = {
            let original = size_tuple.get_item(position)?;
            normalized_dimension(py, &original, position)?
        };
        let position = ffi::Py_ssize_t::try_from(position)
            .map_err(|_| PyMemoryError::new_err("torch.Size position exceeds platform limits"))?;
        replace_size_item(py, size_tuple, position, normalized)?;
    }
    Ok(size)
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
        let arguments =
            fallible_tuple_from_iter(py, [values.clone().into_any().unbind()].into_iter())?;
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
    let mut representation = String::new();
    representation
        .try_reserve_exact("torch.Size([".len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate torch.Size representation"))?;
    representation.push_str("torch.Size([");
    for position in 0..value.len() {
        let dimension = unpack_long_long(py, &value.get_item(position)?)?;
        let mut magnitude = dimension.unsigned_abs();
        let mut dimension_length = 1 + usize::from(dimension.is_negative());
        while magnitude >= 10 {
            magnitude /= 10;
            dimension_length += 1;
        }
        let separator_length = if position == 0 { 0 } else { ", ".len() };
        representation
            .try_reserve(separator_length + dimension_length)
            .map_err(|_| PyMemoryError::new_err("unable to allocate torch.Size representation"))?;
        if position != 0 {
            representation.push_str(", ");
        }
        write!(representation, "{dimension}")
            .map_err(|_| PyRuntimeError::new_err("unable to format torch.Size dimension"))?;
    }
    representation
        .try_reserve("])".len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate torch.Size representation"))?;
    representation.push_str("])");
    Ok(PyString::from_bytes(py, representation.as_bytes())?
        .into_any()
        .unbind())
}

fn size_concat(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if right.cast::<PyTuple>().is_err() {
        return Err(concatenation_type_error(right)?);
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

#[cfg(target_pointer_width = "64")]
const TUPLE_HASH_PRIME_1: ffi::Py_uhash_t = 11_400_714_785_074_694_791;
#[cfg(target_pointer_width = "64")]
const TUPLE_HASH_PRIME_2: ffi::Py_uhash_t = 14_029_467_366_897_019_727;
#[cfg(target_pointer_width = "64")]
const TUPLE_HASH_PRIME_5: ffi::Py_uhash_t = 2_870_177_450_012_600_261;
#[cfg(target_pointer_width = "64")]
const TUPLE_HASH_ROTATION: u32 = 31;

#[cfg(target_pointer_width = "32")]
const TUPLE_HASH_PRIME_1: ffi::Py_uhash_t = 2_654_435_761;
#[cfg(target_pointer_width = "32")]
const TUPLE_HASH_PRIME_2: ffi::Py_uhash_t = 2_246_822_519;
#[cfg(target_pointer_width = "32")]
const TUPLE_HASH_PRIME_5: ffi::Py_uhash_t = 3_747_614_393;
#[cfg(target_pointer_width = "32")]
const TUPLE_HASH_ROTATION: u32 = 13;
const TUPLE_HASH_LENGTH_MIX: ffi::Py_uhash_t = 3_527_539;

#[allow(
    unsafe_code,
    reason = "PyObject_Hash invokes each live tuple item's hash through the stable ABI"
)]
fn size_hash(py: Python<'_>, value: &Bound<'_, PyTuple>) -> PyResult<ffi::Py_hash_t> {
    let mut accumulator = TUPLE_HASH_PRIME_5;
    for position in 0..value.len() {
        let item = value.get_item(position)?;
        // SAFETY: item is live for the duration of the hash call. A -1 return
        // indicates that the item hash set a Python exception.
        let lane = unsafe { ffi::PyObject_Hash(item.as_ptr()) };
        if lane == -1 {
            return Err(PyErr::fetch(py));
        }
        let lane = ffi::Py_uhash_t::from_ne_bytes(lane.to_ne_bytes());
        accumulator = accumulator.wrapping_add(lane.wrapping_mul(TUPLE_HASH_PRIME_2));
        accumulator = accumulator.rotate_left(TUPLE_HASH_ROTATION);
        accumulator = accumulator.wrapping_mul(TUPLE_HASH_PRIME_1);
    }
    accumulator =
        accumulator.wrapping_add(value.len() ^ (TUPLE_HASH_PRIME_5 ^ TUPLE_HASH_LENGTH_MIX));
    let result = ffi::Py_hash_t::from_ne_bytes(accumulator.to_ne_bytes());
    if result == -1 {
        Ok(1_546_275_796)
    } else {
        Ok(result)
    }
}

fn size_reduce(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dimensions = tuple_from_value(py, value)?;
    let arguments = fallible_tuple_from_iter(py, [dimensions.into_any().unbind()].into_iter())?;
    fallible_tuple_from_iter(
        py,
        [
            value.get_type().into_any().unbind(),
            arguments.into_any().unbind(),
        ]
        .into_iter(),
    )
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
unsafe fn size_hash_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
) -> PyResult<ffi::Py_hash_t> {
    // SAFETY: the hash slot supplies a live Size, which is a tuple subtype.
    let value = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, value) }.cast_into::<PyTuple>()?;
    size_hash(py, &value)
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
    reason = "PyType_GetSlot and PyType_FromSpecWithBases require audited stable-ABI raw-pointer calls"
)]
fn create_size_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // SAFETY: the live built-in tuple type exposes a rich-comparison slot with
    // the signature required by Py_tp_richcompare.
    let tuple_richcompare = unsafe {
        ffi::PyType_GetSlot(
            py.get_type::<PyTuple>().as_type_ptr(),
            ffi::Py_tp_richcompare,
        )
    };
    if tuple_richcompare.is_null() {
        return Err(PyRuntimeError::new_err(
            "built-in tuple type does not expose rich comparison",
        ));
    }
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
            slot: ffi::Py_tp_hash,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(hashfunc, size_hash_callback)
                as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_richcompare,
            pfunc: tuple_richcompare,
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
    let bases = fallible_tuple_from_iter(
        py,
        [py.get_type::<PyTuple>().into_any().unbind()].into_iter(),
    )?;

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
