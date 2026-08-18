"""Functional interface."""

import warnings

import torch_rs as torch
from torch_rs import Tensor
from torch_rs.overrides import (
    _get_current_function_mode,
    _pop_mode,
    _push_mode,
)

from ..torch_rs import _nn_functional_dropout


def _overloaded_dropout_arguments(input, include_tensor):
    input_type = type(input)
    if input_type is Tensor:
        # A mode-triggered handle_torch_function call includes the ordinary
        # Tensor type even though the unary fast path excludes it otherwise.
        return [input] if include_tensor else []
    if hasattr(input_type, "__torch_function__"):
        return [input]
    return []


def _has_dropout_torch_function(input):
    # PyTorch's C-level unary probe treats the Tensor type object as
    # overridable, excludes an ordinary exact Tensor, and suppresses errors
    # raised while looking up user-defined override descriptors.
    if input is Tensor:
        return True
    if type(input) is Tensor:
        return False
    try:
        input.__torch_function__
    except BaseException:
        return False
    return True


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

    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if inplace:
        return _nn_functional_dropout(input, p, training, True)
    return _nn_functional_dropout(input, p, training, False)


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
