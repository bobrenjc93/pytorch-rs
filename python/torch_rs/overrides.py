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


def _overloaded_variadic_arguments(inputs, include_tensor=True):
    overloaded_types = set()
    overloaded_args = []
    for input in inputs:
        input_type = type(input)
        if input_type in overloaded_types:
            continue

        if input_type is Tensor:
            if not include_tensor:
                continue
        elif not hasattr(input_type, "__torch_function__"):
            continue
        elif _is_disabled_torch_function_impl(input_type.__torch_function__):
            continue

        if overloaded_types:
            index = len(overloaded_args)
            for old_index, old_arg in enumerate(overloaded_args):
                if issubclass(input_type, type(old_arg)):
                    index = old_index
                    break
            overloaded_args.insert(index, input)
            overloaded_types.add(input_type)
        else:
            overloaded_args = [input]
            overloaded_types = {input_type}
    return overloaded_args


def _has_variadic_torch_function_override(inputs):
    for input in inputs:
        if type(input) is Tensor:
            continue
        try:
            handler = input.__torch_function__
        except BaseException:
            # PyTorch's fast instance probe clears every descriptor failure.
            continue
        if not _is_disabled_torch_function_impl(handler):
            return True
    return False


def _dispatch_variadic_torch_function(
    public_function,
    implementation,
    inputs,
    keyword_arguments,
    include_tensor=True,
):
    """Dispatch variadic inputs through modes and distinct operand overrides."""
    overloaded_args = _overloaded_variadic_arguments(inputs, include_tensor)
    mode = _get_current_function_mode()
    if mode is None and (
        not overloaded_args
        or all(type(argument) is Tensor for argument in overloaded_args)
    ):
        return implementation(inputs)

    types = tuple(type(argument) for argument in overloaded_args)
    dispatch_kwargs = keyword_arguments.copy()
    if mode is not None:
        popped_mode = _pop_mode()
        try:
            result = popped_mode.__torch_function__(
                public_function,
                types,
                inputs,
                dispatch_kwargs,
            )
        finally:
            _push_mode(popped_mode)
        if result is not NotImplemented:
            return result

    for overloaded_arg in overloaded_args:
        if type(overloaded_arg) is Tensor:
            # Tensor's native fallback only re-enters the public wrapper when
            # it is the sole overloaded type. Alongside operand overrides it
            # declines so the next distinct type receives the call.
            if len(overloaded_args) == 1:
                if dispatch_kwargs:
                    return public_function(*inputs, **dispatch_kwargs)
                return _dispatch_variadic_torch_function(
                    public_function,
                    implementation,
                    inputs,
                    keyword_arguments,
                    include_tensor=False,
                )
            continue

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
            inputs,
            dispatch_kwargs,
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
