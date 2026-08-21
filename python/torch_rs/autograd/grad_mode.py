"""Gradient-mode context managers."""

import copyreg as _copyreg
import functools as _functools
import inspect as _inspect
import sys as _sys
import warnings as _warnings
from typing import Any as _Any

from .. import _C as _C
from ..torch_rs import no_grad as no_grad


def _decorate_context(context_factory, function):
    if _inspect.isgeneratorfunction(function):
        @_functools.wraps(function)
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
                            response = generator.throw(*_sys.exc_info())
                    else:
                        with context_factory():
                            response = generator.send(request)
            except StopIteration as error:
                return error.value

        return generator_context

    @_functools.wraps(function)
    def decorate_context(*args, **kwargs):
        with context_factory():
            return function(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, original_function):
        if _inspect.isclass(original_function):
            _warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            function = lambda *args, **kwargs: original_function(*args, **kwargs)
        else:
            function = original_function

        return _decorate_context(self.clone, function)

    def clone(self):
        return self.__class__()


class set_multithreading_enabled(_DecoratorContextManager):
    r"""Context-manager that enables or disables multithreaded backward.

    Ordinarily, when :ref:`accelerator<accelerators>` devices are in use,
    the backward pass runs on device-specific worker threads. The engine
    creates these threads based on the number of available devices and
    reuses them across iterations.

    When ``mode=False``, the backward pass runs on the calling thread
    instead. ``mode=True`` restores the default behavior.

    This can be used as a context-manager or as a function. It is
    thread-local and will not affect computation in other threads.

    Args:
        mode (bool): Whether to enable multithreaded backward (``True``,
                    default) or disable (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`,
        which never uses multithreading.

    """

    def __init__(self, mode: bool) -> None:
        self.prev = _C._is_multithreading_enabled()
        _C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: _Any,
        exc_value: _Any,
        traceback: _Any,
    ) -> None:
        _C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


def _legacy_rebuild_no_grad(context_type):
    return no_grad.__new__(context_type)


def _no_grad_state(context):
    getstate = getattr(context, "__getstate__", None)
    if getstate is not None:
        return getstate()

    instance_state = getattr(context, "__dict__", None)
    slot_state = {}
    for name in _copyreg._slotnames(type(context)) or ():
        try:
            slot_state[name] = getattr(context, name)
        except AttributeError:
            pass
    if slot_state:
        return instance_state, slot_state
    return instance_state


def _no_grad_newobj(context):
    context_type = type(context)
    getnewargs_ex = getattr(context, "__getnewargs_ex__", None)
    if getnewargs_ex is not None:
        newargs_ex = getnewargs_ex()
        if not isinstance(newargs_ex, tuple):
            raise TypeError(
                "__getnewargs_ex__ should return a tuple, "
                f"not '{type(newargs_ex).__name__}'"
            )
        if len(newargs_ex) != 2:
            raise ValueError(
                "__getnewargs_ex__ should return a tuple of length 2, "
                f"not {len(newargs_ex)}"
            )
        newargs, newkwargs = newargs_ex
        if not isinstance(newargs, tuple):
            raise TypeError(
                "first item of the tuple returned by __getnewargs_ex__ "
                f"must be a tuple, not '{type(newargs).__name__}'"
            )
        if not isinstance(newkwargs, dict):
            raise TypeError(
                "second item of the tuple returned by __getnewargs_ex__ "
                f"must be a dict, not '{type(newkwargs).__name__}'"
            )
        return _copyreg.__newobj_ex__, (context_type, newargs, newkwargs)

    getnewargs = getattr(context, "__getnewargs__", None)
    if getnewargs is not None:
        newargs = getnewargs()
        if not isinstance(newargs, tuple):
            raise TypeError(
                "__getnewargs__ should return a tuple, "
                f"not '{type(newargs).__name__}'"
            )
        return _copyreg.__newobj__, (context_type, *newargs)

    return _copyreg.__newobj__, (context_type,)


def _reduce_no_grad(context, protocol):
    if protocol < 2:
        return (
            _legacy_rebuild_no_grad,
            (type(context),),
            _no_grad_state(context),
        )
    newobj, newargs = _no_grad_newobj(context)
    return newobj, newargs, _no_grad_state(context)


__all__ = ["no_grad", "set_multithreading_enabled"]
