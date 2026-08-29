//! Python bindings for gradient-mode context managers.

use std::ffi::CStr;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::{GradModeToken, enter_enable_grad, enter_no_grad, try_exit_grad_mode};

const GRAD_MODE_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys


def _decorate_context(context_factory, function):
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


def _make_grad_mode_context(context_base, name, reduce_name):
    class grad_mode_context(context_base):
        def __new__(cls, original_function=None):
            if original_function is not None:
                return cls()(original_function)
            return super().__new__(cls)

        def __call__(self, function):
            return _decorate_context(type(self), function)

        def __reduce__(self):
            from torch_rs.autograd import grad_mode

            return getattr(grad_mode, reduce_name)(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd import grad_mode

            return getattr(grad_mode, reduce_name)(self, protocol)

    grad_mode_context.__name__ = name
    grad_mode_context.__module__ = "torch_rs.autograd.grad_mode"
    grad_mode_context.__qualname__ = name
    return grad_mode_context
"#;

/// Thread-local autograd recording guard underlying the Python `torch.no_grad` class.
#[pyclass(
    name = "_NoGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyNoGrad {
    entries: Mutex<Vec<GradModeToken>>,
}

#[pymethods]
impl PyNoGrad {
    #[new]
    fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        self.entries
            .lock()
            .expect("Python no-grad token mutex was poisoned")
            .push(enter_no_grad());
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        let token = self
            .entries
            .lock()
            .expect("Python no-grad token mutex was poisoned")
            .pop();
        if let Some(token) = token {
            try_exit_grad_mode(token);
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
    entries: Mutex<Vec<GradModeToken>>,
}

#[pymethods]
impl PyEnableGrad {
    #[new]
    fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        self.entries
            .lock()
            .expect("Python enable-grad token mutex was poisoned")
            .push(enter_enable_grad());
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        let token = self
            .entries
            .lock()
            .expect("Python enable-grad token mutex was poisoned")
            .pop();
        if let Some(token) = token {
            try_exit_grad_mode(token);
        }
    }
}

pub(crate) fn add_grad_mode_contexts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyEnableGrad>()?;
    let grad_mode_helpers = PyModule::from_code(
        py,
        GRAD_MODE_WRAPPER_SOURCE,
        c"torch_rs/_grad_mode.py",
        c"torch_rs._grad_mode",
    )?;
    let make_context = grad_mode_helpers.getattr("_make_grad_mode_context")?;
    let no_grad_class = make_context.call1((
        module.getattr("_NoGradContext")?,
        "no_grad",
        "_reduce_no_grad",
    ))?;
    let enable_grad_class = make_context.call1((
        module.getattr("_EnableGradContext")?,
        "enable_grad",
        "_reduce_enable_grad",
    ))?;
    let exports = module.getattr("__all__")?;
    for name in ["_NoGradContext", "_EnableGradContext"] {
        exports.call_method1("remove", (name,))?;
    }
    module.delattr("_NoGradContext")?;
    module.delattr("_EnableGradContext")?;
    module.add("no_grad", no_grad_class)?;
    module.add("enable_grad", enable_grad_class)?;
    Ok(())
}
