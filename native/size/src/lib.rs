//! Native immutable `torch.Size` compatibility type.

use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyBool, PyDict, PyInt, PyModule, PyString, PyTuple};
use std::ffi::{CStr, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

static SIZE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const TYPE_NAME: &CStr = pyo3::ffi::c_str!("torch.Size");
const NUMEL_NAME: &CStr = pyo3::ffi::c_str!("numel");

/// Returns the process-local immutable `Size` type.
///
/// # Errors
///
/// Returns an error if `CPython` cannot create the native tuple subtype.
pub fn size_type(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    SIZE.get_or_try_init(py, || create_size_type(py))
}

/// Constructs a validated native `Size` from trusted tensor dimensions.
///
/// # Errors
///
/// Returns an error if Python cannot allocate the tuple or `Size` instance.
pub fn size_from_dimensions(py: Python<'_>, dimensions: &[usize]) -> PyResult<Py<PyAny>> {
    let dimensions = PyTuple::new(py, dimensions.iter().copied())?;
    size_type(py)?
        .bind(py)
        .call1((dimensions,))
        .map(Bound::unbind)
}

fn create_size_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let methods = Box::leak(Box::new([
        ffi::PyMethodDef {
            ml_name: NUMEL_NAME.as_ptr(),
            ml_meth: ffi::PyMethodDefPointer {
                PyCFunction: size_numel,
            },
            ml_flags: ffi::METH_NOARGS,
            ml_doc: ptr::null(),
        },
        ffi::PyMethodDef::zeroed(),
    ]));
    let slots = Box::leak(Box::new([
        slot(ffi::Py_tp_new, size_new as ffi::newfunc),
        slot(ffi::Py_tp_repr, size_repr as ffi::reprfunc),
        raw_slot(ffi::Py_tp_methods, methods.as_mut_ptr().cast()),
        slot(ffi::Py_mp_subscript, size_subscript as ffi::binaryfunc),
        slot(ffi::Py_sq_concat, size_concat as ffi::binaryfunc),
        slot(ffi::Py_sq_repeat, size_repeat as ffi::ssizeargfunc),
        slot(ffi::Py_nb_add, size_add as ffi::binaryfunc),
        ffi::PyType_Slot::default(),
    ]));
    let flags = u32::try_from(ffi::Py_TPFLAGS_DEFAULT | ffi::Py_TPFLAGS_IMMUTABLETYPE)
        .map_err(|_| PyValueError::new_err("CPython type flags exceed the stable ABI field"))?;
    let mut spec = ffi::PyType_Spec {
        name: TYPE_NAME.as_ptr(),
        basicsize: 0,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };
    let bases = PyTuple::new(py, [py.get_type::<PyTuple>()])?;
    // SAFETY: `spec`, its leaked slot/method tables, and `bases` satisfy the
    // stable-ABI `PyType_FromSpecWithBases` contract. The returned new
    // reference is checked and owned immediately.
    let size = unsafe { ffi::PyType_FromSpecWithBases(&raw mut spec, bases.as_ptr()) };
    if size.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: the preceding C API call returned a new, non-null reference.
    Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, size) }.unbind())
}

fn slot<T: Copy>(identifier: i32, function: T) -> ffi::PyType_Slot {
    // SAFETY: CPython specifies every type-slot callback as a function pointer
    // stored in `void *`; the caller pairs each callback with its matching slot.
    raw_slot(identifier, unsafe { std::mem::transmute_copy(&function) })
}

const fn raw_slot(identifier: i32, function: *mut c_void) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: identifier,
        pfunc: function,
    }
}

unsafe extern "C" fn size_new(
    subtype: *mut ffi::PyTypeObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    callback(|| {
        Python::attach(|py| {
            // SAFETY: CPython calls `tp_new` with a valid positional tuple and
            // either a valid keyword dict or null.
            let args =
                unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
            if args.len() > 1 {
                return Err(PyTypeError::new_err(format!(
                    "Size expected at most 1 argument, got {}",
                    args.len()
                )));
            }
            if !kwargs.is_null() {
                // SAFETY: a non-null `kwargs` passed to `tp_new` is a dict.
                let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, kwargs) }
                    .cast_into::<PyDict>()?;
                if !kwargs.is_empty() {
                    return Err(PyTypeError::new_err("Size() takes no keyword arguments"));
                }
            }
            let iterable = if args.is_empty() {
                PyTuple::empty(py).into_any()
            } else {
                args.get_item(0)?
            };
            construct_size(py, subtype, &iterable)
        })
    })
}

fn construct_size(
    py: Python<'_>,
    subtype: *mut ffi::PyTypeObject,
    iterable: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Calling `tuple` snapshots the iterable completely before any element is
    // converted, matching CPython/PyTorch mutation and iterator-error order.
    let snapshot = py.get_type::<PyTuple>().call1((iterable,))?;
    let snapshot = snapshot.cast::<PyTuple>()?;
    let mut numpy_integer = None;
    let mut checked_numpy_integer = false;
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(snapshot.len())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyMemoryError, _>(""))?;
    for (index, dimension) in snapshot.iter().enumerate() {
        let preserve_python_integer =
            dimension.is_instance_of::<PyInt>() && !dimension.is_instance_of::<PyBool>();
        let preserve_numpy_integer = if preserve_python_integer {
            false
        } else {
            if !checked_numpy_integer {
                numpy_integer = numpy_integer_type(py);
                checked_numpy_integer = true;
            }
            numpy_integer
                .as_ref()
                .is_some_and(|kind| dimension.is_instance(kind.bind(py)).unwrap_or(false))
        };
        let dimension = if preserve_python_integer || preserve_numpy_integer {
            dimension
        } else if let Ok(indexed) = index_dimension(&dimension) {
            indexed
        } else {
            let name = dimension.get_type().name()?;
            return Err(PyTypeError::new_err(format!(
                "torch.Size() takes an iterable of 'int' (item {index} is '{name}')"
            )));
        };
        converted.push(dimension.unbind());
    }
    let converted = PyTuple::new(py, converted)?;
    allocate_tuple_subtype(py, subtype, &converted)
}

fn index_dimension<'py>(dimension: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: `dimension` is a live Python object. `PyNumber_Index` returns a
    // new reference or null with an exception set.
    let indexed = unsafe { ffi::PyNumber_Index(dimension.as_ptr()) };
    // SAFETY: the pointer follows the owned-reference contract documented
    // above, including the nullable error case.
    unsafe { Bound::<PyAny>::from_owned_ptr_or_err(dimension.py(), indexed) }
}

fn numpy_integer_type(py: Python<'_>) -> Option<Py<PyAny>> {
    match PyModule::import(py, "numpy") {
        Ok(numpy) => numpy.getattr("integer").ok().map(Bound::unbind),
        Err(error) if error.is_instance_of::<PyImportError>(py) => None,
        Err(_) => None,
    }
}

fn allocate_tuple_subtype(
    py: Python<'_>,
    subtype: *mut ffi::PyTypeObject,
    items: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    // SAFETY: `PyType_GetSlot` returns the tuple base's `tp_new` callback for
    // the stable ABI slot identifier. Its signature is `ffi::newfunc`.
    let new_slot =
        unsafe { ffi::PyType_GetSlot(py.get_type::<PyTuple>().as_type_ptr(), ffi::Py_tp_new) };
    if new_slot.is_null() {
        return Err(PyTypeError::new_err("tuple type has no constructor"));
    }
    // SAFETY: the non-null slot was requested as `Py_tp_new`.
    let tuple_new: ffi::newfunc = unsafe { std::mem::transmute(new_slot) };
    let args = PyTuple::new(py, [items])?;
    // SAFETY: `subtype` is a tuple subtype, `args` is a valid argument tuple,
    // and the base constructor returns a new reference or sets an exception.
    let output = unsafe { tuple_new(subtype, args.as_ptr(), ptr::null_mut()) };
    if output.is_null() {
        Err(PyErr::fetch(py))
    } else {
        // SAFETY: the tuple constructor returned a new non-null reference.
        Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, output) }.unbind())
    }
}

unsafe extern "C" fn size_repr(object: *mut ffi::PyObject) -> *mut ffi::PyObject {
    callback(|| {
        Python::attach(|py| {
            // SAFETY: CPython calls `tp_repr` with a valid instance.
            let object =
                unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) }.cast_into::<PyTuple>()?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(object.len())
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyMemoryError, _>(""))?;
            for dimension in object.iter() {
                values.push(unpack_dimension(&dimension)?.to_string());
            }
            Ok(
                PyString::new(py, &format!("torch.Size([{}])", values.join(", ")))
                    .into_any()
                    .unbind(),
            )
        })
    })
}

unsafe extern "C" fn size_numel(
    object: *mut ffi::PyObject,
    _noargs: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    callback(|| {
        Python::attach(|py| {
            // SAFETY: CPython calls this method with a valid Size instance.
            let object =
                unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) }.cast_into::<PyTuple>()?;
            let mut elements = 1_i64;
            for dimension in object.iter() {
                elements = elements.wrapping_mul(unpack_dimension(&dimension)?);
            }
            Ok(elements.into_pyobject(py)?.into_any().unbind())
        })
    })
}

fn unpack_dimension(dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
    let indexed = index_dimension(dimension)?;
    indexed
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"))
}

unsafe extern "C" fn size_subscript(
    object: *mut ffi::PyObject,
    index: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    callback(|| base_binary_and_wrap(object, index, ffi::Py_mp_subscript, false))
}

unsafe extern "C" fn size_concat(
    left: *mut ffi::PyObject,
    right: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    callback(|| base_binary_and_wrap(left, right, ffi::Py_sq_concat, true))
}

unsafe extern "C" fn size_add(
    left: *mut ffi::PyObject,
    right: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    callback(|| {
        Python::attach(|py| {
            // Use tuple concatenation only when both operands are tuples.
            // SAFETY: both pointers are live operands supplied by CPython.
            let left_bound = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, left) };
            // SAFETY: both pointers are live operands supplied by CPython.
            let right_bound = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, right) };
            if left_bound.cast::<PyTuple>().is_err() || right_bound.cast::<PyTuple>().is_err() {
                return Ok(py.NotImplemented());
            }
            base_binary_and_wrap(left, right, ffi::Py_sq_concat, true)
        })
    })
}

fn base_binary_and_wrap(
    left: *mut ffi::PyObject,
    right: *mut ffi::PyObject,
    identifier: i32,
    always_wrap: bool,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        // SAFETY: the requested tuple slot has the `binaryfunc` signature.
        let slot =
            unsafe { ffi::PyType_GetSlot(py.get_type::<PyTuple>().as_type_ptr(), identifier) };
        if slot.is_null() {
            return Err(PyTypeError::new_err("tuple operation is unavailable"));
        }
        // SAFETY: the non-null slot is a requested binary tuple callback.
        let operation: ffi::binaryfunc = unsafe { std::mem::transmute(slot) };
        // SAFETY: CPython supplied live operands for this callback.
        let output = unsafe { operation(left, right) };
        if output.is_null() {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: the tuple slot returned a new non-null reference.
        let output = unsafe { Bound::<PyAny>::from_owned_ptr(py, output) }.unbind();
        if !always_wrap && output.bind(py).cast::<PyTuple>().is_err() {
            return Ok(output);
        }
        let subtype = if is_size_instance(py, left) {
            left
        } else {
            right
        };
        // SAFETY: `subtype` is a live Size instance selected above.
        let subtype = unsafe { ffi::Py_TYPE(subtype) };
        construct_size(py, subtype, output.bind(py))
    })
}

unsafe extern "C" fn size_repeat(
    object: *mut ffi::PyObject,
    count: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    callback(|| {
        Python::attach(|py| {
            // SAFETY: the requested tuple slot has the `ssizeargfunc` signature.
            let slot = unsafe {
                ffi::PyType_GetSlot(py.get_type::<PyTuple>().as_type_ptr(), ffi::Py_sq_repeat)
            };
            if slot.is_null() {
                return Err(PyTypeError::new_err("tuple repetition is unavailable"));
            }
            // SAFETY: the non-null slot is the requested repetition callback.
            let operation: ffi::ssizeargfunc = unsafe { std::mem::transmute(slot) };
            // SAFETY: CPython supplied a live Size instance.
            let output = unsafe { operation(object, count) };
            if output.is_null() {
                return Err(PyErr::fetch(py));
            }
            // SAFETY: the tuple slot returned a new non-null reference.
            let output = unsafe { Bound::<PyAny>::from_owned_ptr(py, output) }.unbind();
            // SAFETY: `object` is a live Size instance.
            construct_size(py, unsafe { ffi::Py_TYPE(object) }, output.bind(py))
        })
    })
}

fn is_size_instance(py: Python<'_>, object: *mut ffi::PyObject) -> bool {
    // SAFETY: callbacks only pass live operand pointers here.
    unsafe { ffi::PyObject_TypeCheck(object, size_type(py).unwrap().as_ptr().cast()) != 0 }
}

fn callback(function: impl FnOnce() -> PyResult<Py<PyAny>>) -> *mut ffi::PyObject {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(output)) => output.into_ptr(),
        Ok(Err(error)) => {
            Python::attach(|py| error.restore(py));
            ptr::null_mut()
        }
        Err(_) => {
            Python::attach(|py| {
                pyo3::exceptions::PyRuntimeError::new_err("panic in Size callback").restore(py);
            });
            ptr::null_mut()
        }
    }
}
