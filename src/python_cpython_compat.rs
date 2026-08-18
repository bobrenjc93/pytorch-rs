//! CPython-version compatibility boundaries for Python tensor dispatch.

use std::cell::Cell;
use std::ffi::CStr;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyModule, PyTuple};

static TORCH_FUNCTION_DESCRIPTOR_CALLER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

thread_local! {
    static CONST_DATA_PTR_LEGACY_REDISPATCH_DEPTH: Cell<usize> = const { Cell::new(0) };
}

const TORCH_FUNCTION_DESCRIPTOR_CALLER_SOURCE: &CStr = cr"
def _call_descriptor(function, args):
    return function(*args)
";

#[allow(
    unsafe_code,
    reason = "mode dispatch must use CPython's combined descriptor lookup and call boundary, matching PyTorch"
)]
pub(crate) fn call_torch_function_mode_handler(
    py: Python<'_>,
    mode: &Bound<'_, PyAny>,
    function: &Py<PyAny>,
    types: &Bound<'_, PyTuple>,
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    // PyTorch invokes a mode through PyObject_CallMethod after separately
    // validating one descriptor resolution. Keeping the second resolution and
    // invocation in the same CPython operation is observable at the recursion
    // limit for stateful descriptors.
    let result = unsafe {
        ffi::PyObject_CallMethod(
            mode.as_ptr(),
            c"__torch_function__".as_ptr(),
            c"OOO".as_ptr(),
            function.as_ptr(),
            types.as_ptr(),
            args.as_ptr(),
        )
    };
    if result.is_null() {
        Err(PyErr::fetch(py))
    } else {
        // SAFETY: PyObject_CallMethod returned a new owned reference.
        Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, result) }.unbind())
    }
}

pub(crate) fn torch_function_descriptor_caller(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    TORCH_FUNCTION_DESCRIPTOR_CALLER.get_or_try_init(py, || {
        let helpers = PyModule::from_code(
            py,
            TORCH_FUNCTION_DESCRIPTOR_CALLER_SOURCE,
            c"torch_rs/_torch_function.py",
            c"torch_rs._torch_function",
        )?;
        Ok(helpers.getattr("_call_descriptor")?.unbind())
    })
}

pub(crate) fn initialize_torch_function_descriptor_caller(py: Python<'_>) -> PyResult<()> {
    // Build the recursive TensorBase fallback while import has normal recursion
    // headroom. Lazy construction is observably too late near the recursion limit.
    let _ = torch_function_descriptor_caller(py)?;
    Ok(())
}

pub(crate) fn uses_legacy_tensorbase_redispatch(py: Python<'_>) -> bool {
    py.version_info() < (3, 12)
}

pub(crate) struct ConstDataPtrLegacyRedispatchGuard;

pub(crate) fn enter_const_data_ptr_legacy_redispatch() -> ConstDataPtrLegacyRedispatchGuard {
    CONST_DATA_PTR_LEGACY_REDISPATCH_DEPTH.with(|depth| depth.set(depth.get() + 1));
    ConstDataPtrLegacyRedispatchGuard
}

impl Drop for ConstDataPtrLegacyRedispatchGuard {
    fn drop(&mut self) {
        CONST_DATA_PTR_LEGACY_REDISPATCH_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

fn const_data_ptr_legacy_redispatch_depth() -> usize {
    CONST_DATA_PTR_LEGACY_REDISPATCH_DEPTH.with(Cell::get)
}

#[allow(
    unsafe_code,
    reason = "CPython 3.10 and 3.11 TensorBase parity requires probing legacy recursive dispatch boundaries"
)]
pub(crate) fn probe_const_data_ptr_legacy_redispatch(py: Python<'_>) -> PyResult<()> {
    // The Python helper retains the recursive fallback frame, while the native
    // PyTorch path also crosses alternating subclass and callable recursion
    // checks before retrying the descriptor. Probe those checks only after the
    // mode declines so accepted results and handler exceptions win.
    let python_310_contexts = [
        c" in __subclasscheck__",
        c" while calling a Python object",
        c" in __subclasscheck__",
        c" while calling a Python object",
        c" in __subclasscheck__",
        c" while calling a Python object",
    ];
    let redispatch_depth = const_data_ptr_legacy_redispatch_depth();
    let python_311_contexts = [
        c" while calling a Python object",
        c" in __subclasscheck__",
        c" while calling a Python object",
        c" in __subclasscheck__",
        // The initial Tensor.__torch_function__ retry reaches its subclass
        // boundary before another callable boundary at tight headroom.
        if redispatch_depth == 1 {
            c" in __subclasscheck__"
        } else {
            c" while calling a Python object"
        },
    ];
    let contexts: &[&CStr] = if py.version_info() < (3, 11) {
        &python_310_contexts
    } else {
        &python_311_contexts
    };
    let mut entered = 0;
    for context in contexts {
        if unsafe { ffi::Py_EnterRecursiveCall(context.as_ptr()) } != 0 {
            let error = PyErr::fetch(py);
            for _ in 0..entered {
                unsafe { ffi::Py_LeaveRecursiveCall() };
            }
            return Err(error);
        }
        entered += 1;
    }
    for _ in 0..entered {
        unsafe { ffi::Py_LeaveRecursiveCall() };
    }
    Ok(())
}
