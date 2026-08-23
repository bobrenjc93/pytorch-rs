"""Dynamic ``__torch_function__`` override modes."""

import types as _types
import warnings

from .torch_rs import (
    Tensor,
    _get_function_stack_at,
    _has_torch_function_unary as has_torch_function_unary,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


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
    *,
    positional_arguments=(),
    include_tensor=True,
):
    if not has_torch_function_unary(input):
        return implementation(input, *positional_arguments, **keyword_arguments)

    mode = _get_current_function_mode()
    overloaded_args = _overloaded_unary_arguments(input, include_tensor)
    types = tuple(type(argument) for argument in overloaded_args)
    call_arguments = (input, *positional_arguments)
    if mode is not None:
        popped_mode = _pop_mode()
        try:
            result = popped_mode.__torch_function__(
                public_function,
                types,
                call_arguments,
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
                positional_arguments=positional_arguments,
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
            call_arguments,
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


__all__ = ["TorchFunctionMode"]
