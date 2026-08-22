"""Dynamic ``__torch_function__`` override modes."""

import functools as _functools
import types as _types
import warnings
from collections.abc import Callable

from .torch_rs import (
    Tensor,
    _get_function_stack_at,
    _has_torch_function_unary as has_torch_function_unary,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


def _disable_user_warnings(func):
    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*is deprecated, please use.*",
                module="torch",
            )
            return func(*args, **kwargs)

    return wrapper


@_functools.cache
def _get_tensor_methods() -> set[Callable]:
    """Returns a set of the overridable methods on ``torch.Tensor``"""
    names = {
        name
        for tensor_type in Tensor.__mro__
        if tensor_type is not object
        for name in vars(tensor_type)
    }
    methods = {
        method for name in names if callable(method := getattr(Tensor, name))
    }
    # These entry points do not pass themselves to ``__torch_function__``:
    # iteration delegates to ``dim`` and ``unbind``, while ``stride`` is an
    # ignored metadata query in PyTorch's override registry.
    methods.discard(Tensor.__iter__)
    methods.discard(Tensor.stride)
    return methods


@_disable_user_warnings
def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """
    return func in _get_tensor_methods() or func.__name__ == "__get__"


class TorchFunctionMode:
    """Override ``__torch_function__`` operations within a dynamic scope."""

    def __init__(self) -> None:
        pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just "
            "`with Mode()`",
            stacklevel=2,
        )
        return cls(*args, **kwargs)


class BaseTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def _push_mode(mode):
    _push_on_torch_function_stack(mode)


def _pop_mode():
    return _pop_torch_function_stack()


def _get_current_function_mode():
    stack_len = _len_torch_function_stack()
    return _get_function_stack_at(stack_len - 1) if stack_len > 0 else None


def _get_current_function_mode_stack():
    return [
        _get_function_stack_at(index)
        for index in range(_len_torch_function_stack())
    ]


def _is_disabled_torch_function_impl(handler):
    return (
        isinstance(handler, _types.BuiltinFunctionType)
        and handler.__module__ == "torch._C"
        and handler.__name__ == "_disabled_torch_function_impl"
    )


def _overloaded_unary_arguments(input, include_tensor):
    input_type = type(input)
    if input_type is Tensor:
        # A mode-triggered handle_torch_function call includes the ordinary
        # Tensor type even though the unary fast path excludes it otherwise.
        return [input] if include_tensor else []
    if hasattr(input_type, "__torch_function__"):
        handler = input_type.__torch_function__
        if _is_disabled_torch_function_impl(handler):
            return []
        return [input]
    return []


def _dispatch_unary_torch_function(
    public_function,
    implementation,
    input,
    keyword_arguments,
    include_tensor=True,
):
    if not has_torch_function_unary(input):
        return implementation(input, **keyword_arguments)

    mode = _get_current_function_mode()
    overloaded_args = _overloaded_unary_arguments(input, include_tensor)
    types = tuple(type(argument) for argument in overloaded_args)
    if mode is not None:
        popped_mode = _pop_mode()
        try:
            result = popped_mode.__torch_function__(
                public_function,
                types,
                (input,),
                keyword_arguments.copy(),
            )
        finally:
            _push_mode(popped_mode)
        if result is not NotImplemented:
            return result

    for overloaded_arg in overloaded_args:
        if type(overloaded_arg) is Tensor:
            return _dispatch_unary_torch_function(
                public_function,
                implementation,
                input,
                keyword_arguments,
                include_tensor=False,
            )

        torch_func_method = overloaded_arg.__torch_function__
        if (
            hasattr(torch_func_method, "__self__")
            and torch_func_method.__self__ is overloaded_arg
        ):
            warnings.warn(
                "Defining your `__torch_function__ as a plain method is "
                "deprecated and will be an error in future, please define "
                "it as a classmethod.",
                DeprecationWarning,
                stacklevel=2,
            )

        result = torch_func_method(
            public_function,
            types,
            (input,),
            keyword_arguments.copy(),
        )
        if result is not NotImplemented:
            return result

    func_name = f"{public_function.__module__}.{public_function.__name__}"
    message = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__torch_function__: {[type(arg) for arg in overloaded_args]}"
    )
    current_mode = _get_current_function_mode()
    if current_mode is not None:
        message += f" nor in mode {current_mode}"
    raise TypeError(message)


__all__ = ["TorchFunctionMode", "is_tensor_method_or_property"]
