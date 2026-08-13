//! Stable-ABI construction for the immutable `torch.layout` descriptor.
//!
//! `CPython` 3.10 added `Py_TPFLAGS_IMMUTABLETYPE`, but did not expose an API
//! for freezing an already-created heap type through the stable ABI until
//! Python 3.14. Constructing the zero-state type with the flag already set is
//! supported by `PyType_FromSpec` across the package's abi3-py310 range.

use std::ffi::c_void;

use pyo3::{Bound, Py, PyAny, PyResult, Python, exceptions::PyRuntimeError, ffi};

/// The canonical Python layout type and its sole supported instance.
pub struct LayoutObjects {
    /// Immutable `torch_rs.layout` type object.
    pub layout: Py<PyAny>,
    /// Canonical `torch.strided` descriptor.
    pub strided: Py<PyAny>,
}

unsafe extern "C" fn layout_repr(_object: *mut ffi::PyObject) -> *mut ffi::PyObject {
    // SAFETY: the literal is a static NUL-terminated string and the CPython
    // repr slot requires a new reference, which PyUnicode_FromString returns.
    unsafe { ffi::PyUnicode_FromString(c"torch.strided".as_ptr()) }
}

/// Creates an immutable, non-constructible layout type and its canonical value.
///
/// # Errors
///
/// Returns a Python exception if the stable-ABI type or singleton allocation
/// fails.
pub fn create_layout_objects(py: Python<'_>) -> PyResult<LayoutObjects> {
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_repr,
            pfunc: layout_repr as *mut c_void,
        },
        ffi::PyType_Slot::default(),
    ];
    let flags = ffi::Py_TPFLAGS_DEFAULT
        | ffi::Py_TPFLAGS_DISALLOW_INSTANTIATION
        | ffi::Py_TPFLAGS_IMMUTABLETYPE;
    let flags = flags
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("layout type flags exceed unsigned int"))?;
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs.layout".as_ptr(),
        // Zero inherits the object base size and keeps instances state-free.
        basicsize: 0,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the specification and terminated slot table remain live for
    // the call, every slot has the required signature, and CPython returns a
    // new reference or sets a Python exception and returns null.
    let layout = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))?
    };
    // SAFETY: PyType_FromSpec returned a type object; GenericAlloc accepts
    // that live type pointer and returns a new reference (or a Python error).
    let strided = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyType_GenericAlloc(layout.as_ptr().cast(), 0),
        )?
    };

    Ok(LayoutObjects {
        layout: layout.unbind(),
        strided: strided.unbind(),
    })
}
