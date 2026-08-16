//! Python bindings for gradient-mode context managers.

use std::cell::Cell;
use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::{enter_no_grad, exit_no_grad};

const NO_GRAD_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys


def _decorate_no_grad(context_factory, function):
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
            return _decorate_no_grad(type(self), function)

        def __reduce__(self):
            from torch_rs.autograd.grad_mode import _reduce_no_grad

            return _reduce_no_grad(self, 0)

        def __reduce_ex__(self, protocol):
            from torch_rs.autograd.grad_mode import _reduce_no_grad

            return _reduce_no_grad(self, protocol)

    no_grad.__module__ = "torch_rs.autograd.grad_mode"
    no_grad.__qualname__ = "no_grad"
    return no_grad
"#;

thread_local! {
    static NO_GRAD_CONTEXT_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Thread-local autograd recording guard underlying the Python `torch.no_grad` class.
#[pyclass(
    name = "_NoGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyNoGrad;

#[pymethods]
impl PyNoGrad {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        enter_no_grad();
        NO_GRAD_CONTEXT_DEPTH.set(
            NO_GRAD_CONTEXT_DEPTH
                .get()
                .checked_add(1)
                .expect("Python no-grad nesting depth overflowed usize"),
        );
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        if let Some(depth) = NO_GRAD_CONTEXT_DEPTH.get().checked_sub(1) {
            NO_GRAD_CONTEXT_DEPTH.set(depth);
            exit_no_grad();
        }
    }
}

pub(crate) fn add_no_grad(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    let no_grad_helpers = PyModule::from_code(
        py,
        NO_GRAD_WRAPPER_SOURCE,
        c"torch_rs/_no_grad.py",
        c"torch_rs._no_grad",
    )?;
    let no_grad_class = no_grad_helpers
        .getattr("_make_no_grad")?
        .call1((module.getattr("_NoGradContext")?,))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_NoGradContext",))?;
    module.delattr("_NoGradContext")?;
    module.add("no_grad", no_grad_class)?;
    Ok(())
}
