//! Python bindings for gradient-mode context managers.

use std::cell::{Cell, RefCell};
use std::ffi::CStr;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

use crate::{enter_no_grad, exit_no_grad, grad_mode_depth, restore_grad_mode};

const GRAD_MODE_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import TypeVar as _TypeVar
from typing import Union as _Union

try:
    from typing import Self as _Self
except ImportError:
    from typing_extensions import Self as _Self


F = _TypeVar("F", bound=_Callable)


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
        r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        tensor([2.])
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True
        >>> @torch.enable_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> with torch.no_grad():
        ...     z = tripler(x)
        >>> z.requires_grad
        True

    """

        def __new__(cls, orig_func: _Optional[F] = None) -> _Union[_Self, F]:
            if orig_func is not None:
                return cls()(orig_func)
            return super().__new__(cls)

        def __call__(self, orig_func: F) -> F:
            return _decorate_grad_mode(type(self), orig_func)

        def __enter__(self) -> None:
            return super().__enter__()

        def __exit__(
            self,
            exc_type: _Any,
            exc_value: _Any,
            traceback: _Any,
        ) -> None:
            return super().__exit__(exc_type, exc_value, traceback)

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

thread_local! {
    static NO_GRAD_CONTEXT_DEPTH: Cell<usize> = const { Cell::new(0) };
    static ENABLE_GRAD_CONTEXT_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
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

/// Thread-local autograd recording guard underlying the Python `torch.enable_grad` class.
#[pyclass(
    name = "_EnableGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyEnableGrad;

#[pymethods]
impl PyEnableGrad {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __enter__(slf: &Bound<'_, Self>) -> PyResult<()> {
        let previous_no_grad_depth = grad_mode_depth();
        slf.as_any().setattr("prev", previous_no_grad_depth == 0)?;
        ENABLE_GRAD_CONTEXT_STACK.with(|stack| {
            stack.borrow_mut().push(previous_no_grad_depth);
        });
        restore_grad_mode(0);
        Ok(())
    }

    fn __exit__(
        slf: &Bound<'_, Self>,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let previous_no_grad_depth =
            ENABLE_GRAD_CONTEXT_STACK.with(|stack| stack.borrow_mut().pop());
        if let Some(depth) = previous_no_grad_depth {
            restore_grad_mode(depth);
            return Ok(());
        }

        let previous_enabled = slf.as_any().getattr("prev")?;
        if !previous_enabled.is_instance_of::<PyBool>() {
            let type_name = previous_enabled
                .get_type()
                .getattr("__name__")?
                .extract::<String>()?;
            return Err(PyTypeError::new_err(format!(
                "set_grad_enabled(): argument 'enabled' (position 1) must be bool, not {type_name}"
            )));
        }
        restore_grad_mode(usize::from(!previous_enabled.extract::<bool>()?));
        Ok(())
    }
}

pub(crate) fn add_grad_mode_contexts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyEnableGrad>()?;
    let grad_mode_helpers = PyModule::from_code(
        py,
        GRAD_MODE_WRAPPER_SOURCE,
        c"torch_rs/_grad_mode_contexts.py",
        c"torch_rs._grad_mode_contexts",
    )?;
    let no_grad_class = grad_mode_helpers
        .getattr("_make_no_grad")?
        .call1((module.getattr("_NoGradContext")?,))?;
    let enable_grad_class = grad_mode_helpers
        .getattr("_make_enable_grad")?
        .call1((module.getattr("_EnableGradContext")?,))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_NoGradContext",))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_EnableGradContext",))?;
    module.delattr("_NoGradContext")?;
    module.delattr("_EnableGradContext")?;
    module.add("no_grad", no_grad_class)?;
    module.add("enable_grad", enable_grad_class)?;
    Ok(())
}
