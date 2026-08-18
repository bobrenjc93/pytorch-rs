"""Functional interface."""

import math
import warnings

import torch_rs as torch
from torch_rs import Tensor
from torch_rs._diagnostics import _format_single_element_tensor
from torch_rs.overrides import _dispatch_unary_torch_function

from ..torch_rs import (
    _nn_functional_dropout,
)


def _dropout_impl(input, p, training, inplace):
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
    return _dispatch_unary_torch_function(
        dropout,
        _dropout_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )
