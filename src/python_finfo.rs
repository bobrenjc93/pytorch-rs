//! Stable-ABI construction for the immutable float32-only `torch.finfo` type.

use std::ffi::{CStr, c_void};
use std::ptr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyDict, PyTuple};

use crate::{
    DType,
    python::{CallKeywordOrder, call_type_summary, python_argument_type_name},
    python_dtype::PyDType,
};

static FINFO_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[repr(C)]
struct PyFInfoObject {
    object: ffi::PyObject,
    dtype: u8,
}

const FLOAT32_DTYPE_TAG: u8 = 0;

const fn finfo_dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::Float32 => FLOAT32_DTYPE_TAG,
    }
}

#[derive(Clone, Copy)]
enum FInfoField {
    Bits,
    DType,
    Eps,
    Max,
    Min,
    Resolution,
    SmallestNormal,
    Tiny,
}

fn parse_finfo_dtype(value: &Bound<'_, PyAny>, position: Option<usize>) -> PyResult<DType> {
    if let Ok(dtype) = value.cast::<PyDType>() {
        let dtype = dtype.try_borrow()?.inner();
        if dtype.is_floating_point() {
            return Ok(dtype);
        }
        return Err(PyTypeError::new_err(
            "torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'",
        ));
    }

    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_argument_type_name(value)?;
    Err(PyTypeError::new_err(format!(
        "finfo(): argument 'type'{position} must be torch.dtype, not {actual}"
    )))
}

fn invalid_finfo_arguments(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let actual = call_type_summary(args, kwargs, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "finfo() received an invalid combination of arguments - got ({actual}), but expected one of:\n * (torch.dtype type)\n * ()\n"
    )))
}

fn bind_finfo_dtype(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<DType> {
    let keyword_count = kwargs.map_or(0, pyo3::types::PyDictMethods::len);
    match (args.len(), keyword_count) {
        (0, 0) => Ok(DType::default()),
        (0, 1) => {
            let kwargs = kwargs.expect("one keyword requires a keyword dictionary");
            if let Some(dtype) = kwargs.get_item("type")? {
                parse_finfo_dtype(&dtype, None)
            } else {
                Err(PyTypeError::new_err(
                    "finfo() missing 1 required positional arguments: \"type\"",
                ))
            }
        }
        (1, 0) => parse_finfo_dtype(&args.get_item(0)?, Some(1)),
        _ => Err(invalid_finfo_arguments(args, kwargs)?),
    }
}

fn format_finfo_value(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust scientific formatting always contains an exponent");
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    let exponent = exponent
        .parse::<i32>()
        .expect("Rust scientific formatting always uses an integer exponent");
    format!("{mantissa}e{exponent:+03}")
}

fn finfo_repr(dtype: DType) -> String {
    let info = dtype.floating_point_info();
    format!(
        "finfo(resolution={}, min={}, max={}, eps={}, smallest_normal={}, tiny={}, dtype={})",
        format_finfo_value(info.resolution()),
        format_finfo_value(info.min()),
        format_finfo_value(info.max()),
        format_finfo_value(info.eps()),
        format_finfo_value(info.smallest_normal()),
        format_finfo_value(info.smallest_normal()),
        dtype.name(),
    )
}

fn finfo_field(py: Python<'_>, dtype: DType, field: FInfoField) -> PyResult<Py<PyAny>> {
    let info = dtype.floating_point_info();
    match field {
        FInfoField::Bits => info.bits().into_py_any(py),
        FInfoField::DType => dtype.name().into_py_any(py),
        FInfoField::Eps => info.eps().into_py_any(py),
        FInfoField::Max => info.max().into_py_any(py),
        FInfoField::Min => info.min().into_py_any(py),
        FInfoField::Resolution => info.resolution().into_py_any(py),
        FInfoField::SmallestNormal | FInfoField::Tiny => info.smallest_normal().into_py_any(py),
    }
}

#[allow(
    unsafe_code,
    reason = "the stable-ABI finfo instance owns a native dtype tag"
)]
unsafe fn finfo_dtype(value: *mut ffi::PyObject) -> DType {
    // SAFETY: every callback receiving this value is installed only on the
    // finfo type, whose basicsize is PyFInfoObject and whose allocation is
    // initialized by finfo_new_callback after PyType_GenericAlloc.
    match unsafe { (*value.cast::<PyFInfoObject>()).dtype } {
        FLOAT32_DTYPE_TAG => DType::Float32,
        _ => unreachable!("finfo contains an unknown native dtype tag"),
    }
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
    // SAFETY: CPython owns the positional tuple for the duration of the callback.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is null or a live dictionary.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    let dtype = bind_finfo_dtype(&args, kwargs.as_ref())?;

    // SAFETY: the new slot supplies the live finfo type. GenericAlloc returns
    // a new instance or sets a Python exception and returns null.
    let instance =
        unsafe { Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_GenericAlloc(subtype, 0))? };
    // SAFETY: GenericAlloc allocated at least PyFInfoObject::basicsize bytes
    // for the exact, non-subclassable finfo type.
    unsafe {
        (*instance.as_ptr().cast::<PyFInfoObject>()).dtype = finfo_dtype_tag(dtype);
    }
    Ok(instance.unbind().into_ptr())
}

#[allow(
    unsafe_code,
    reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
)]
unsafe fn finfo_repr_callback(
    py: Python<'_>,
    value: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: the repr slot is installed only on the finfo type.
    let dtype = unsafe { finfo_dtype(value) };
    finfo_repr(dtype).into_py_any(py).map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "CPython supplies borrowed operands to the panic-safe PyO3 trampoline"
)]
unsafe fn finfo_richcompare_callback(
    py: Python<'_>,
    left: *mut ffi::PyObject,
    right: *mut ffi::PyObject,
    operation: std::ffi::c_int,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: both operands are live for the duration of the rich comparison.
    let same_type = unsafe { ffi::Py_TYPE(left) == ffi::Py_TYPE(right) };
    let equal = same_type && unsafe { finfo_dtype(left) == finfo_dtype(right) };
    match operation {
        ffi::Py_EQ => equal.into_py_any(py).map(Py::into_ptr),
        ffi::Py_NE => (!equal).into_py_any(py).map(Py::into_ptr),
        _ => Ok(py.NotImplemented().into_ptr()),
    }
}

macro_rules! finfo_getter {
    ($getter:ident, $callback:ident, $field:expr) => {
        #[allow(
            unsafe_code,
            reason = "CPython supplies a borrowed instance to the panic-safe PyO3 trampoline"
        )]
        unsafe fn $callback(
            py: Python<'_>,
            value: *mut ffi::PyObject,
        ) -> PyResult<*mut ffi::PyObject> {
            // SAFETY: the descriptor is installed only on the finfo type.
            let dtype = unsafe { finfo_dtype(value) };
            finfo_field(py, dtype, $field).map(Py::into_ptr)
        }

        #[allow(
            unsafe_code,
            reason = "the getter signature includes an unused CPython closure pointer"
        )]
        unsafe extern "C" fn $getter(
            value: *mut ffi::PyObject,
            _closure: *mut c_void,
        ) -> *mut ffi::PyObject {
            // SAFETY: the descriptor passes a live finfo instance, and the
            // PyO3 trampoline catches panics and restores Python exceptions.
            unsafe { pyo3::impl_::trampoline::get_trampoline_function!(reprfunc, $callback)(value) }
        }
    };
}

finfo_getter!(finfo_bits_getter, finfo_bits_callback, FInfoField::Bits);
finfo_getter!(finfo_dtype_getter, finfo_dtype_callback, FInfoField::DType);
finfo_getter!(finfo_eps_getter, finfo_eps_callback, FInfoField::Eps);
finfo_getter!(finfo_max_getter, finfo_max_callback, FInfoField::Max);
finfo_getter!(finfo_min_getter, finfo_min_callback, FInfoField::Min);
finfo_getter!(
    finfo_resolution_getter,
    finfo_resolution_callback,
    FInfoField::Resolution
);
finfo_getter!(
    finfo_smallest_normal_getter,
    finfo_smallest_normal_callback,
    FInfoField::SmallestNormal
);
finfo_getter!(finfo_tiny_getter, finfo_tiny_callback, FInfoField::Tiny);

fn read_only_getset(name: &'static CStr, get: ffi::getter) -> ffi::PyGetSetDef {
    ffi::PyGetSetDef {
        name: name.as_ptr(),
        get: Some(get),
        set: None,
        doc: ptr::null(),
        closure: ptr::null_mut(),
    }
}

#[allow(
    unsafe_code,
    reason = "PyType_FromSpec requires an audited stable-ABI raw type specification"
)]
fn create_finfo_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // CPython descriptors retain pointers to their get-set definitions. Leak
    // this tiny table deliberately so it remains valid for the type lifetime.
    let getsets = Box::leak(Box::new([
        read_only_getset(c"bits", finfo_bits_getter),
        read_only_getset(c"dtype", finfo_dtype_getter),
        read_only_getset(c"eps", finfo_eps_getter),
        read_only_getset(c"max", finfo_max_getter),
        read_only_getset(c"min", finfo_min_getter),
        read_only_getset(c"resolution", finfo_resolution_getter),
        read_only_getset(c"smallest_normal", finfo_smallest_normal_getter),
        read_only_getset(c"tiny", finfo_tiny_getter),
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
            slot: ffi::Py_tp_richcompare,
            pfunc: pyo3::impl_::trampoline::get_trampoline_function!(
                richcmpfunc,
                finfo_richcompare_callback
            ) as *mut c_void,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_hash,
            pfunc: ffi::PyObject_HashNotImplemented as *mut c_void,
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
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs.finfo".as_ptr(),
        // The zero-initialized trailing dtype tag is Float32. Besides keeping
        // the scalar type native, the fixed C state makes generic pickling
        // reject reconstruction exactly like PyTorch's native finfo object.
        basicsize: std::mem::size_of::<PyFInfoObject>()
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("finfo instance size exceeds C int"))?,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the specification and terminated slot and get-set tables remain
    // live for the call, every callback uses the matching CPython signature,
    // and CPython returns a new reference or sets an exception.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))
            .map(Bound::unbind)
    }
}

/// Returns the process-stable immutable Python `finfo` type.
pub(crate) fn finfo_type_object(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    FINFO_TYPE.get_or_try_init(py, || create_finfo_type(py))
}

/// Reports whether a value is an exact instance of the native `finfo` type.
pub(crate) fn is_finfo_instance(value: &Bound<'_, PyAny>) -> bool {
    FINFO_TYPE
        .get(value.py())
        .is_some_and(|finfo| value.get_type().as_ptr().cast() == finfo.as_ptr())
}
