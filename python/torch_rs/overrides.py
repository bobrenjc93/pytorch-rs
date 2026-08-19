"""Dynamic ``__torch_function__`` override modes."""

import types as _types
import warnings

import torch_rs as torch

from .torch_rs import (
    Tensor,
    _get_function_stack_at,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return (
        inp is Tensor
        or type(inp) is Tensor
        or type(inp) is torch.Tensor
        or hasattr(inp, "__torch_function__")
    )


class TorchFunctionMode:
    """Override ``__torch_function__`` operations within a dynamic scope."""

    def __init__(self):
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


def _has_unary_torch_function(input):
    # PyTorch's C-level unary probe treats the Tensor type object as
    # overridable, excludes an ordinary exact Tensor, and suppresses errors
    # raised while looking up user-defined override descriptors.
    if input is Tensor:
        return True
    if type(input) is Tensor:
        return False
    try:
        handler = input.__torch_function__
    except BaseException:
        return False
    return not _is_disabled_torch_function_impl(handler)


def _dispatch_unary_torch_function(
    public_function,
    implementation,
    input,
    keyword_arguments,
    include_tensor=True,
):
    mode = _get_current_function_mode()
    if mode is None and not _has_unary_torch_function(input):
        return implementation(input, **keyword_arguments)

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


__all__ = ["TorchFunctionMode", "is_tensor_like"]
