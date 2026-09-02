//! Python bindings for gradient-mode context managers.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::{
    enter_enable_grad, enter_no_grad, exit_grad_mode, grad_mode::GradModeToken, is_grad_enabled,
    set_grad_enabled_state,
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


def _set_grad_enabled_bool_type_name(value):
    value_type = type(value)
    module = value_type.__module__
    name = value_type.__name__
    if module == "numpy":
        return f"numpy.{name}"
    return name


def _require_set_grad_enabled_bool(value):
    if type(value) is not bool:
        raise TypeError(
            "set_grad_enabled(): argument 'enabled' "
            f"(position 1) must be bool, not {_set_grad_enabled_bool_type_name(value)}"
        )
    return value


_set_grad_enabled_signature = inspect.Signature(
    parameters=(
        inspect.Parameter(
            "mode",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=bool,
        ),
    ),
    return_annotation=None,
)


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


def _make_set_grad_enabled(context_base):
    class set_grad_enabled(context_base):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __init__(self, mode: bool) -> None:
            mode = _require_set_grad_enabled_bool(mode)
            self.prev = self._push_initial_mode(mode)
            self.mode = mode

        def __enter__(self) -> None:
            self._enter_mode(self.mode)

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._exit_mode()

        def __call__(self, function):
            self.__exit__(None, None, None)
            mode = self.mode
            return _decorate_grad_mode(lambda: type(self)(mode), function)

        def __str__(self):
            return f"{type(self).__module__}.{type(self).__qualname__}(mode={self.mode})"

        def __repr__(self):
            return str(self)

        def clone(self):
            return type(self)(self.mode)

        def __reduce__(self):
            from torch_rs.autograd.grad_mode import _reduce_set_grad_enabled

            return _reduce_set_grad_enabled(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd.grad_mode import _reduce_set_grad_enabled

            return _reduce_set_grad_enabled(self, protocol)

    set_grad_enabled.__module__ = "torch_rs.autograd.grad_mode"
    set_grad_enabled.__qualname__ = "set_grad_enabled"
    set_grad_enabled.__init__.__module__ = "torch_rs.autograd.grad_mode"
    set_grad_enabled.__init__.__qualname__ = "set_grad_enabled.__init__"
    set_grad_enabled.__signature__ = _set_grad_enabled_signature
    return set_grad_enabled
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

#[derive(Clone, Copy)]
struct SetGradEnabledState {
    previous_enabled: bool,
    pending_enter: bool,
}

fn push_set_grad_enabled_state(
    states_by_thread: &Mutex<HashMap<ThreadId, Vec<SetGradEnabledState>>>,
    previous_enabled: bool,
    pending_enter: bool,
) {
    states_by_thread
        .lock()
        .expect("grad-mode context token mutex is poisoned")
        .entry(thread::current().id())
        .or_default()
        .push(SetGradEnabledState {
            previous_enabled,
            pending_enter,
        });
}

fn pop_set_grad_enabled_state(
    states_by_thread: &Mutex<HashMap<ThreadId, Vec<SetGradEnabledState>>>,
) -> Option<bool> {
    let mut states_by_thread = states_by_thread
        .lock()
        .expect("grad-mode context token mutex is poisoned");
    let thread_id = thread::current().id();
    let states = states_by_thread.get_mut(&thread_id)?;
    let previous = states.pop().map(|entry| entry.previous_enabled);
    if states.is_empty() {
        states_by_thread.remove(&thread_id);
    }
    previous
}

fn consume_pending_set_grad_enabled_state(
    states_by_thread: &Mutex<HashMap<ThreadId, Vec<SetGradEnabledState>>>,
    enabled: bool,
) -> bool {
    let mut states_by_thread = states_by_thread
        .lock()
        .expect("grad-mode context token mutex is poisoned");
    let Some(states) = states_by_thread.get_mut(&thread::current().id()) else {
        return false;
    };
    let Some(state) = states.last_mut() else {
        return false;
    };
    if !state.pending_enter || is_grad_enabled() != enabled {
        return false;
    }
    state.pending_enter = false;
    true
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

/// Thread-local autograd recording guard underlying the Python `torch.set_grad_enabled` class.
#[pyclass(
    name = "_SetGradEnabledContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PySetGradEnabled {
    states_by_thread: Mutex<HashMap<ThreadId, Vec<SetGradEnabledState>>>,
}

#[pymethods]
impl PySetGradEnabled {
    #[new]
    fn new() -> Self {
        Self {
            states_by_thread: Mutex::new(HashMap::new()),
        }
    }

    fn _push_initial_mode(&self, enabled: bool) -> bool {
        let previous = set_grad_enabled_state(enabled);
        push_set_grad_enabled_state(&self.states_by_thread, previous, true);
        previous
    }

    fn _enter_mode(&self, enabled: bool) {
        if consume_pending_set_grad_enabled_state(&self.states_by_thread, enabled) {
            return;
        }
        let previous = set_grad_enabled_state(enabled);
        push_set_grad_enabled_state(&self.states_by_thread, previous, false);
    }

    fn _exit_mode(&self) {
        if let Some(previous) = pop_set_grad_enabled_state(&self.states_by_thread) {
            set_grad_enabled_state(previous);
        }
    }
}

pub(crate) fn add_grad_mode_contexts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyEnableGrad>()?;
    module.add_class::<PySetGradEnabled>()?;
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
    let set_grad_enabled_class = grad_mode_helpers
        .getattr("_make_set_grad_enabled")?
        .call1((module.getattr("_SetGradEnabledContext")?,))?;
    let exports = module.getattr("__all__")?;
    for name in [
        "_NoGradContext",
        "_EnableGradContext",
        "_SetGradEnabledContext",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    module.delattr("_NoGradContext")?;
    module.delattr("_EnableGradContext")?;
    module.delattr("_SetGradEnabledContext")?;
    module.add("no_grad", no_grad_class)?;
    module.add("enable_grad", enable_grad_class)?;
    module.add("set_grad_enabled", set_grad_enabled_class)?;
    Ok(())
}
