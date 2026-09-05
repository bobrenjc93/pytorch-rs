//! Python bindings for gradient-mode context managers.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

use crate::{
    enter_enable_grad, enter_no_grad, exit_grad_mode, grad_mode::GradModeToken,
    python::python_type_name, set_grad_enabled as set_grad_enabled_state,
};

const GRAD_MODE_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys


def _decorate_grad_mode(context_factory, function):
    if inspect.isgeneratorfunction(function):
        @functools.wraps(function)
        def generator_context(*args, **kwargs):
            generator = function(*args, **kwargs)
            try:
                with context_factory():
                    response = generator.send(None)

                while True:
                    try:
                        request = yield response
                    except GeneratorExit:
                        with context_factory():
                            generator.close()
                        raise
                    except BaseException:
                        with context_factory():
                            response = generator.throw(*sys.exc_info())
                    else:
                        with context_factory():
                            response = generator.send(request)
            except StopIteration as error:
                return error.value

        return generator_context

    @functools.wraps(function)
    def decorate_context(*args, **kwargs):
        with context_factory():
            return function(*args, **kwargs)

    return decorate_context


def _make_no_grad(context_base):
    class no_grad(context_base):
        def __new__(cls, original_function=None):
            if original_function is not None:
                return cls()(original_function)
            return super().__new__(cls)

        def __call__(self, function):
            return _decorate_grad_mode(type(self), function)

        def __reduce__(self):
            from torch_rs.autograd.grad_mode import _reduce_no_grad

            return _reduce_no_grad(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd.grad_mode import _reduce_no_grad

            return _reduce_no_grad(self, protocol)

    no_grad.__module__ = "torch_rs.autograd.grad_mode"
    no_grad.__qualname__ = "no_grad"
    return no_grad


def _make_enable_grad(context_base):
    class enable_grad(context_base):
        def __new__(cls, original_function=None):
            if original_function is not None:
                return cls()(original_function)
            return super().__new__(cls)

        def __call__(self, function):
            return _decorate_grad_mode(type(self), function)

        def __reduce__(self):
            from torch_rs.autograd.grad_mode import _reduce_enable_grad

            return _reduce_enable_grad(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd.grad_mode import _reduce_enable_grad

            return _reduce_enable_grad(self, protocol)

    enable_grad.__module__ = "torch_rs.autograd.grad_mode"
    enable_grad.__qualname__ = "enable_grad"
    return enable_grad
"#;

fn push_context_token(
    tokens_by_thread: &Mutex<HashMap<ThreadId, Vec<GradModeToken>>>,
    token: GradModeToken,
) {
    tokens_by_thread
        .lock()
        .expect("grad-mode context token mutex is poisoned")
        .entry(thread::current().id())
        .or_default()
        .push(token);
}

fn pop_context_token(
    tokens_by_thread: &Mutex<HashMap<ThreadId, Vec<GradModeToken>>>,
) -> Option<GradModeToken> {
    let mut tokens_by_thread = tokens_by_thread
        .lock()
        .expect("grad-mode context token mutex is poisoned");
    let thread_id = thread::current().id();
    let tokens = tokens_by_thread.get_mut(&thread_id)?;
    let token = tokens.pop();
    if tokens.is_empty() {
        tokens_by_thread.remove(&thread_id);
    }
    token
}

/// Thread-local autograd recording guard underlying the Python `torch.no_grad` class.
#[pyclass(
    name = "_NoGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyNoGrad {
    tokens_by_thread: Mutex<HashMap<ThreadId, Vec<GradModeToken>>>,
}

#[pymethods]
impl PyNoGrad {
    #[new]
    fn new() -> Self {
        Self {
            tokens_by_thread: Mutex::new(HashMap::new()),
        }
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        let token = enter_no_grad();
        push_context_token(&self.tokens_by_thread, token);
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        if let Some(token) = pop_context_token(&self.tokens_by_thread) {
            exit_grad_mode(token);
        }
    }
}

/// Thread-local autograd recording guard underlying the Python `torch.enable_grad` class.
#[pyclass(
    name = "_EnableGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyEnableGrad {
    tokens_by_thread: Mutex<HashMap<ThreadId, Vec<GradModeToken>>>,
}

#[pymethods]
impl PyEnableGrad {
    #[new]
    fn new() -> Self {
        Self {
            tokens_by_thread: Mutex::new(HashMap::new()),
        }
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        let token = enter_enable_grad();
        push_context_token(&self.tokens_by_thread, token);
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        if let Some(token) = pop_context_token(&self.tokens_by_thread) {
            exit_grad_mode(token);
        }
    }
}

#[pyfunction(name = "_set_grad_enabled", signature = (enabled))]
fn set_grad_enabled_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyTypeError::new_err(format!(
            "set_grad_enabled(): argument 'enabled' (position 1) must be bool, not {type_name}"
        )));
    }
    set_grad_enabled_state(enabled.is_truthy()?);
    Ok(())
}

pub(crate) fn add_grad_mode_contexts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyEnableGrad>()?;
    module.add_function(wrap_pyfunction!(set_grad_enabled_native, module)?)?;
    let grad_mode_helpers = PyModule::from_code(
        py,
        GRAD_MODE_WRAPPER_SOURCE,
        c"torch_rs/_grad_mode.py",
        c"torch_rs._grad_mode",
    )?;
    let no_grad_class = grad_mode_helpers
        .getattr("_make_no_grad")?
        .call1((module.getattr("_NoGradContext")?,))?;
    let enable_grad_class = grad_mode_helpers
        .getattr("_make_enable_grad")?
        .call1((module.getattr("_EnableGradContext")?,))?;
    let exports = module.getattr("__all__")?;
    for name in ["_NoGradContext", "_EnableGradContext", "_set_grad_enabled"] {
        exports.call_method1("remove", (name,))?;
    }
    module.delattr("_NoGradContext")?;
    module.delattr("_EnableGradContext")?;
    module.add("no_grad", no_grad_class)?;
    module.add("enable_grad", enable_grad_class)?;
    Ok(())
}
