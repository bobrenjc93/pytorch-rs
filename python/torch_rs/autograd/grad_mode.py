"""Gradient-mode context managers."""

import builtins as _builtins
import copyreg as _copyreg
import functools as _functools
import inspect as _inspect
import sys as _sys

from ..torch_rs import _set_grad_enabled as _set_grad_enabled
from ..torch_rs import enable_grad as enable_grad
from ..torch_rs import is_grad_enabled as _is_grad_enabled
from ..torch_rs import no_grad as no_grad


def _grad_mode_type_name(value):
    value_type = _builtins.type(value)
    if value_type.__module__ == "torch_rs":
        if value_type.__name__ == "Tensor":
            return "Tensor"
        if value_type.__name__ in (
            "dtype",
            "device",
            "layout",
            "memory_format",
            "finfo",
        ):
            return f"torch.{value_type.__name__}"
        if value_type.__name__ == "Size":
            return "torch.Size"
    if value_type.__module__ == "numpy":
        return f"numpy.{value_type.__name__}"
    return value_type.__name__


def _require_exact_bool(value):
    if _builtins.type(value) is not _builtins.bool:
        raise TypeError(
            "set_grad_enabled(): argument 'enabled' "
            f"(position 1) must be bool, not {_grad_mode_type_name(value)}"
        )


def _decorate_grad_mode(context_factory, function):
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


class set_grad_enabled:
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: _builtins.bool) -> None:
        _require_exact_bool(mode)
        self.prev = _is_grad_enabled()
        self.mode = mode
        _set_grad_enabled(mode)

    def __call__(self, function):
        _set_grad_enabled(self.prev)
        return _decorate_grad_mode(self.clone, function)

    def __enter__(self) -> None:
        _set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _set_grad_enabled(self.prev)

    def __str__(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self):
        return self.__class__(self.mode)


def _legacy_rebuild_no_grad(context_type):
    return no_grad.__new__(context_type)


def _legacy_rebuild_enable_grad(context_type):
    return enable_grad.__new__(context_type)


def _grad_mode_state(context):
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


def _grad_mode_newobj(context):
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


def _reduce_grad_mode(context, protocol, legacy_rebuild):
    if protocol < 2:
        return (
            legacy_rebuild,
            (type(context),),
            _grad_mode_state(context),
        )
    newobj, newargs = _grad_mode_newobj(context)
    return newobj, newargs, _grad_mode_state(context)


def _reduce_no_grad(context, protocol):
    return _reduce_grad_mode(context, protocol, _legacy_rebuild_no_grad)


def _reduce_enable_grad(context, protocol):
    return _reduce_grad_mode(context, protocol, _legacy_rebuild_enable_grad)


_torch_module = _sys.modules.get("torch_rs")
if _torch_module is not None:
    _torch_module.set_grad_enabled = set_grad_enabled
_autograd_module = _sys.modules.get("torch_rs.autograd")
if _autograd_module is not None:
    _autograd_module.set_grad_enabled = set_grad_enabled


__all__ = ["no_grad", "enable_grad", "set_grad_enabled"]
