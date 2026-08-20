//! Thread-local `TorchFunctionMode` stack bindings.

use std::cell::RefCell;
use std::ffi::CStr;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCFunction, PyModule, PyString};

use crate::python::PyTensor;

const HAS_TORCH_FUNCTION_DOC: &CStr = c"Check for __torch_function__ implementations in the elements of an iterable\n    or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s\n    and ``Parameter`` s non-dispatchable.  Use this to guard a call to\n    :func:`handle_torch_function`; don't use it to test if something\n    is Tensor-like, use :func:`is_tensor_like` instead.\n    Arguments\n    ---------\n    relevant_args : iterable\n        Iterable or arguments to check for __torch_function__ methods.\n    Returns\n    -------\n    bool\n        True if any of the elements of relevant_args have __torch_function__\n        implementations, False otherwise.\n    See Also\n    ________\n    torch.is_tensor_like\n        Checks if something is a Tensor-like, including an exact ``Tensor``.\n    ";

thread_local! {
    static TORCH_FUNCTION_MODE_STACK: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct ActiveTorchFunctionMode {
    mode: Option<Py<PyAny>>,
}

impl ActiveTorchFunctionMode {
    pub(crate) fn get(&self) -> Option<&Py<PyAny>> {
        self.mode.as_ref()
    }

    pub(crate) fn restore(&mut self) {
        if let Some(mode) = self.mode.take() {
            TORCH_FUNCTION_MODE_STACK.with(|stack| stack.borrow_mut().push(mode));
        }
    }
}

impl Drop for ActiveTorchFunctionMode {
    fn drop(&mut self) {
        self.restore();
    }
}

pub(crate) fn is_empty() -> bool {
    TORCH_FUNCTION_MODE_STACK.with(|stack| stack.borrow().is_empty())
}

pub(crate) fn pop() -> ActiveTorchFunctionMode {
    let mode = TORCH_FUNCTION_MODE_STACK.with(|stack| stack.borrow_mut().pop());
    ActiveTorchFunctionMode { mode }
}

fn is_disabled_torch_function_impl(handler: &Bound<'_, PyAny>) -> bool {
    if handler.cast::<PyCFunction>().is_err() {
        return false;
    }
    let Ok(module) = handler
        .getattr("__module__")
        .and_then(|value| value.extract::<String>())
    else {
        return false;
    };
    let Ok(name) = handler
        .getattr("__name__")
        .and_then(|value| value.extract::<String>())
    else {
        return false;
    };
    module == "torch._C" && name == "_disabled_torch_function_impl"
}

#[allow(
    unsafe_code,
    reason = "PyTorch uses PySequence_Fast and exception-suppressing attribute lookup for this native probe"
)]
fn has_torch_function(py: Python<'_>, relevant_args: &Bound<'_, PyAny>) -> PyResult<bool> {
    // PySequence_Fast accepts arbitrary iterables, fully materializes them
    // before examining their elements, and provides PyTorch's exact error for
    // non-iterables.
    let sequence = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PySequence_Fast(relevant_args.as_ptr(), c"expected a sequence".as_ptr()),
        )?
    };
    if sequence.len()? != 0 && !is_empty() {
        return Ok(true);
    }

    let tensor_type = py.get_type::<PyTensor>();
    for argument in sequence.try_iter()? {
        let argument = argument?;
        if argument.is(tensor_type.as_any()) {
            return Ok(true);
        }
        if argument.get_type().is(&tensor_type) {
            continue;
        }

        // PyTorch performs one legacy attribute lookup and suppresses every
        // exception raised by user-defined descriptors during the probe.
        let handler = unsafe {
            ffi::PyObject_GetAttrString(argument.as_ptr(), c"__torch_function__".as_ptr())
        };
        if handler.is_null() {
            unsafe { ffi::PyErr_Clear() };
            continue;
        }
        let handler = unsafe { Bound::<PyAny>::from_owned_ptr(py, handler) };
        if !is_disabled_torch_function_impl(&handler) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn has_torch_function_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    relevant_args: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    let relevant_args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, relevant_args) };
    has_torch_function(py, &relevant_args)?
        .into_py_any(py)
        .map(Py::into_ptr)
}

#[allow(
    unsafe_code,
    reason = "METH_O and a package-local _C owner are required to match PyTorch's native callable contract"
)]
fn add_has_torch_function(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    // CPython retains the method definition pointer for the function lifetime.
    let method = Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: c"_has_torch_function".as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunction: pyo3::impl_::trampoline::get_trampoline_function!(
                binaryfunc,
                has_torch_function_callback
            ),
        },
        ml_flags: ffi::METH_O,
        ml_doc: HAS_TORCH_FUNCTION_DOC.as_ptr(),
    }));
    let owner_name = PyString::new(py, "torch_rs._C");
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(method, module.as_ptr(), owner_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_has_torch_function",))?;
    Ok(())
}

#[pyfunction]
fn _push_on_torch_function_stack(mode: Py<PyAny>) {
    TORCH_FUNCTION_MODE_STACK.with(|stack| stack.borrow_mut().push(mode));
}

#[pyfunction]
fn _pop_torch_function_stack() -> PyResult<Py<PyAny>> {
    TORCH_FUNCTION_MODE_STACK
        .with(|stack| stack.borrow_mut().pop())
        .ok_or_else(|| PyRuntimeError::new_err("trying to pop from empty mode stack"))
}

#[pyfunction]
fn _len_torch_function_stack() -> usize {
    TORCH_FUNCTION_MODE_STACK.with(|stack| stack.borrow().len())
}

#[pyfunction]
fn _get_function_stack_at(py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
    TORCH_FUNCTION_MODE_STACK.with(|stack| {
        stack
            .borrow()
            .get(index)
            .map(|mode| mode.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("Tried to get stack at idx that's too big"))
    })
}

pub(crate) fn add_torch_function_mode_stack(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_has_torch_function(module)?;
    for function in [
        wrap_pyfunction!(_push_on_torch_function_stack, module)?,
        wrap_pyfunction!(_pop_torch_function_stack, module)?,
        wrap_pyfunction!(_len_torch_function_stack, module)?,
        wrap_pyfunction!(_get_function_stack_at, module)?,
    ] {
        let name = function.getattr("__name__")?;
        module.add_function(function.clone())?;
        module.getattr("__all__")?.call_method1("remove", (name,))?;
    }
    Ok(())
}
