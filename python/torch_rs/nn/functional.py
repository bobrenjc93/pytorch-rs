"""Functional interface."""

import math
import sys as _sys
import types as _types
import warnings

import torch_rs as torch
from torch_rs import Tensor
from torch_rs.overrides import (
    _get_current_function_mode,
    _pop_mode,
    _push_mode,
)

from ..torch_rs import (
    _nn_functional_dropout,
    _nn_functional_dropout_tensor_autograd_suffix,
)


def _overloaded_dropout_arguments(input, include_tensor):
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


def _is_disabled_torch_function_impl(handler):
    return (
        isinstance(handler, _types.BuiltinFunctionType)
        and handler.__module__ == "torch._C"
        and handler.__name__ == "_disabled_torch_function_impl"
    )


def _has_dropout_torch_function(input):
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


def _handle_dropout_torch_function(
    input, p, training, inplace, include_tensor
):
    overloaded_args = _overloaded_dropout_arguments(input, include_tensor)
    types = tuple(type(argument) for argument in overloaded_args)
    mode = _get_current_function_mode()

    if mode is not None:
        popped_mode = _pop_mode()
        try:
            result = popped_mode.__torch_function__(
                dropout,
                types,
                (input,),
                {"p": p, "training": training, "inplace": inplace},
            )
        finally:
            _push_mode(popped_mode)
        if result is not NotImplemented:
            return result

    for overloaded_arg in overloaded_args:
        if type(overloaded_arg) is Tensor:
            return _dropout_impl(
                input, p, training, inplace, include_tensor=False
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
            dropout,
            types,
            (input,),
            {"p": p, "training": training, "inplace": inplace},
        )
        if result is not NotImplemented:
            return result

    func_name = f"{dropout.__module__}.{dropout.__name__}"
    message = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__torch_function__: {[type(arg) for arg in overloaded_args]}"
    )
    current_mode = _get_current_function_mode()
    if current_mode is not None:
        message += f" nor in mode {current_mode}"
    raise TypeError(message)


def _dropout_impl(input, p, training, inplace, include_tensor):
    mode = _get_current_function_mode()
    if mode is not None or _has_dropout_torch_function(input):
        return _handle_dropout_torch_function(
            input, p, training, inplace, include_tensor
        )

    range_probability, probability_for_error = _dropout_range_probability(p)
    if range_probability < 0.0 or range_probability > 1.0:
        if type(probability_for_error) is Tensor:
            probability_for_error = _format_single_element_tensor(
                probability_for_error, range_probability
            )
        raise ValueError(
            "dropout probability has to be between 0 and 1, but got "
            f"{probability_for_error}"
        )
    if inplace:
        return _nn_functional_dropout(input, p, training, True)
    return _nn_functional_dropout(input, p, training, False)


def _dropout_range_probability(p):
    if type(p) is not Tensor:
        return p, p

    element_count = p.numel()
    if element_count == 0:
        raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
    if element_count != 1:
        raise RuntimeError(
            "Boolean value of Tensor with more than one value is ambiguous"
        )

    value = p.item()
    if len(p.shape) == 0:
        return value, value
    return value, p


def _format_single_element_tensor(tensor, value):
    finite = math.isfinite(value)
    if finite and value != 0.0:
        integer_mode = value == math.ceil(value)
        scientific = abs(value) > 1.0e8 or abs(value) < 1.0e-4
    else:
        integer_mode = True
        scientific = False

    if scientific:
        formatted = f"{value:.4e}"
    elif integer_mode:
        formatted = f"{value:.0f}"
        if finite:
            formatted += "."
    else:
        formatted = f"{value:.4f}"

    try:
        formatted = _format_single_element_tensor_contents(
            len(tensor.shape), formatted
        )
    except RecursionError as error:
        # Before CPython 3.12, the pure-Python recursion check reports the
        # boundary where it happened (for example, ``in comparison``), while
        # PyTorch's native tensor formatter reaches the callable boundary.
        # Preserve the original exception and traceback while matching that
        # public diagnostic.
        if _sys.version_info < (3, 12):
            error.args = (
                "maximum recursion depth exceeded while calling a Python object",
            )
        raise
    suffix = _nn_functional_dropout_tensor_autograd_suffix(tensor)
    return f"tensor({formatted}{suffix})"


def _format_single_element_tensor_contents(
    dimensions, formatted, formatter_frames=6
):
    # PyTorch reaches its recursive tensor printer through six additional
    # Tensor.__format__/repr helper frames. Preserve that recursion headroom so
    # deeply ranked probability diagnostics fail at the same resource limit.
    if formatter_frames:
        return _format_single_element_tensor_contents(
            dimensions, formatted, formatter_frames - 1
        )
    if dimensions == 1:
        return f"[{formatted}]"
    return (
        "["
        + _format_single_element_tensor_contents(
            dimensions - 1, formatted, 0
        )
        + "]"
    )


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        raise NotImplementedError(
            "torch_rs.nn.functional.relu does not support inplace=True"
        )
    return torch.relu(input)


def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    return _dropout_impl(input, p, training, inplace, include_tensor=True)
