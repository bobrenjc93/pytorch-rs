//! Thread-local `TorchFunctionMode` stack bindings.

use std::cell::RefCell;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

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
