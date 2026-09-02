//! Python bindings for gradient-mode context managers.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::{
    enter_enable_grad, enter_grad_mode, enter_no_grad, exit_grad_mode, grad_mode::GradModeToken,
    is_grad_enabled, is_grad_mode_token_current,
};

const GRAD_MODE_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys
import threading


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


def _grad_mode_bool_type_name(value):
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    name = getattr(value_type, "__name__", value_type.__class__.__name__)
    if module == "numpy":
        return f"numpy.{name}"
    return name


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


def _make_set_grad_enabled(
    enter_grad_mode,
    exit_grad_mode,
    is_grad_enabled,
    is_grad_mode_token_current,
):
    class set_grad_enabled:
        """Context-manager that sets gradient calculation on or off."""

        def __new__(cls, mode=None):
            instance = super().__new__(cls)
            instance._tokens_by_thread = {}
            return instance

        def __init__(self, mode: bool) -> None:
            if type(mode) is not bool:
                raise TypeError(
                    "set_grad_enabled(): argument 'enabled' (position 1) "
                    f"must be bool, not {_grad_mode_bool_type_name(mode)}"
                )
            self.prev = is_grad_enabled()
            self.mode = mode
            self._push_context()

        def _has_current_context_token(self):
            tokens_by_thread = getattr(self, "_tokens_by_thread", None)
            if tokens_by_thread is None:
                self._tokens_by_thread = {}
                return False
            tokens = tokens_by_thread.get(threading.get_ident())
            return bool(tokens) and is_grad_mode_token_current(tokens[-1])

        def _push_context(self):
            token = enter_grad_mode(self.mode)
            self._tokens_by_thread.setdefault(threading.get_ident(), []).append(token)

        def _pop_context(self):
            tokens_by_thread = getattr(self, "_tokens_by_thread", None)
            if not tokens_by_thread:
                return
            thread_id = threading.get_ident()
            tokens = tokens_by_thread.get(thread_id)
            if not tokens:
                return
            token = tokens.pop()
            if not tokens:
                del tokens_by_thread[thread_id]
            exit_grad_mode(token)

        def __call__(self, function):
            self._pop_context()
            return _decorate_grad_mode(self.clone, function)

        def __enter__(self):
            if not self._has_current_context_token():
                self._push_context()
            return None

        def __exit__(self, exception_type, exception_value, traceback):
            self._pop_context()

        def __reduce__(self):
            from torch_rs.autograd.grad_mode import _reduce_set_grad_enabled

            return _reduce_set_grad_enabled(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd.grad_mode import _reduce_set_grad_enabled

            return _reduce_set_grad_enabled(self, protocol)

        def __str__(self):
            context_type = type(self)
            return (
                f"{context_type.__module__}.{context_type.__qualname__}"
                f"(mode={self.mode})"
            )

        def __repr__(self):
            return str(self)

        def clone(self):
            return type(self)(self.mode)

    set_grad_enabled.__module__ = "torch_rs.autograd.grad_mode"
    set_grad_enabled.__qualname__ = "set_grad_enabled"
    set_grad_enabled.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                "mode",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
            ),
        ],
        return_annotation=None,
    )
    return set_grad_enabled
"#;

#[pyfunction(name = "_grad_mode_enter")]
fn grad_mode_enter_py(enabled: bool) -> usize {
    enter_grad_mode(enabled).raw()
}

#[pyfunction(name = "_grad_mode_exit")]
fn grad_mode_exit_py(token: usize) {
    exit_grad_mode(GradModeToken::from_raw(token));
}

#[pyfunction(name = "_is_grad_enabled")]
fn is_grad_enabled_py() -> bool {
    is_grad_enabled()
}

#[pyfunction(name = "_grad_mode_token_is_current")]
fn grad_mode_token_is_current_py(token: usize) -> bool {
    is_grad_mode_token_current(GradModeToken::from_raw(token))
}

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

pub(crate) fn add_grad_mode_contexts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyEnableGrad>()?;
    module.add_function(wrap_pyfunction!(grad_mode_enter_py, module)?)?;
    module.add_function(wrap_pyfunction!(grad_mode_exit_py, module)?)?;
    module.add_function(wrap_pyfunction!(is_grad_enabled_py, module)?)?;
    module.add_function(wrap_pyfunction!(grad_mode_token_is_current_py, module)?)?;
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
        .call1((
            module.getattr("_grad_mode_enter")?,
            module.getattr("_grad_mode_exit")?,
            module.getattr("_is_grad_enabled")?,
            module.getattr("_grad_mode_token_is_current")?,
        ))?;
    let exports = module.getattr("__all__")?;
    for name in [
        "_NoGradContext",
        "_EnableGradContext",
        "_grad_mode_enter",
        "_grad_mode_exit",
        "_is_grad_enabled",
        "_grad_mode_token_is_current",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    module.delattr("_NoGradContext")?;
    module.delattr("_EnableGradContext")?;
    module.delattr("_grad_mode_enter")?;
    module.delattr("_grad_mode_exit")?;
    module.delattr("_is_grad_enabled")?;
    module.delattr("_grad_mode_token_is_current")?;
    module.add("no_grad", no_grad_class)?;
    module.add("enable_grad", enable_grad_class)?;
    module.add("set_grad_enabled", set_grad_enabled_class)?;
    Ok(())
}
