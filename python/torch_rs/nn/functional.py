"""Functional interface."""

import math
import warnings

import torch_rs as torch
from torch_rs import Tensor
from torch_rs._diagnostics import _format_single_element_tensor
from torch_rs.overrides import _dispatch_unary_torch_function

from ..torch_rs import (
    _nn_functional_dropout,
    _nn_functional_linear,
    _nn_functional_mse_loss,
)


_LINEAR_DOC = r"""
linear(input, weight, bias=None) -> Tensor

Applies the rank-1, rank-2, or rank-3 transformation
:math:`\mathrm{output} = \mathrm{input} \, \mathrm{weight}^{T}`, with an
optional bias for rank-1 input.

The current native implementation requires exact ``torch_rs.Tensor`` operands
with CPU ``float32`` storage and shape ``(in_features,)``,
``(rows, in_features)``, or ``(batch, sequence, in_features)`` for ``input``
and ``(out_features, in_features)`` for ``weight``. For rank-2 and rank-3
input, ``bias`` must be ``None``. For rank-1 input, ``bias`` may instead be an
exact rank-1 tensor with shape ``(out_features,)``. The operation returns a
fresh, independent row-major tensor with the corresponding final dimension
replaced by ``out_features``.

Tensor subclasses, active ``TorchFunctionMode`` contexts, and active autograd
recording are not supported. Gradient-requiring input, weight, or supported
bias operands may be used inside ``torch.no_grad()``.
"""


_MSE_LOSS_DOC = r"""
mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

Measures the element-wise mean squared error between ``input`` and ``target``.

The current native implementation requires exact ``torch_rs.Tensor`` operands
with CPU ``float32`` storage, ``reduction='none'``, ``size_average=None``,
``reduce=None``, and ``weight=None``. Operands may have the same shape, or
exactly one operand may be rank zero and broadcast across the other, or one
rank-2 ``(M, N)`` matrix may be paired with a rank-1 ``(N,)`` vector in either
operand order. It fuses subtraction and square into one native pass and
returns a fresh, independent tensor with PyTorch-compatible values, shape,
strides, and broadcast warning.

Other broadcasting, reduced outputs, weights, Tensor subclasses, active
``TorchFunctionMode`` contexts, and active autograd recording are not
supported. Gradient-requiring operands may be used inside ``torch.no_grad()``.
"""


def _validate_dropout_probability(p):
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


def _dropout_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    if inplace:
        return _nn_functional_dropout("dropout", input, p, training, True)
    return _nn_functional_dropout("dropout", input, p, training, False)


def _dropout1d_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    # Preserve PyTorch's pre-native invalid-input validation order.
    input.dim()
    if inplace:
        return _nn_functional_dropout("dropout1d", input, p, training, True)
    return _nn_functional_dropout("dropout1d", input, p, training, False)


def _dropout2d_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    # Preserve PyTorch's pre-native invalid-input validation order.
    input_dimension = input.dim()
    if input_dimension == 2:
        warnings.warn(
            "dropout2d: Received a 2-D input to dropout2d, which is deprecated "
            "and will result in an error in a future release. To retain the "
            "behavior and silence this warning, please use dropout instead. "
            "Note that dropout2d exists to provide channel-wise dropout on "
            "inputs with 2 spatial dimensions, a channel dimension, and an "
            "optional batch dimension (i.e. 3D or 4D inputs).",
            stacklevel=4,
        )
    if input_dimension == 3:
        warnings.warn(
            "dropout2d: Received a 3D input to dropout2d and assuming that "
            "channel-wise 1D dropout behavior is desired - input is interpreted "
            "as shape (N, C, L), where C is the channel dim. This behavior will "
            "change in a future release to interpret the input as one without a "
            "batch dimension, i.e. shape (C, H, W). To maintain the 1D "
            "channel-wise dropout behavior, please switch to using dropout1d "
            "instead.",
            stacklevel=4,
        )
    if inplace:
        return _nn_functional_dropout("dropout2d", input, p, training, True)
    return _nn_functional_dropout("dropout2d", input, p, training, False)


def _dropout3d_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    # Preserve PyTorch's pre-native invalid-input validation order.
    input.dim()
    if inplace:
        return _nn_functional_dropout("dropout3d", input, p, training, True)
    return _nn_functional_dropout("dropout3d", input, p, training, False)


def _alpha_dropout_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    if inplace:
        return _nn_functional_dropout("alpha_dropout", input, p, training, True)
    return _nn_functional_dropout("alpha_dropout", input, p, training, False)


def _feature_alpha_dropout_impl(input, p, training, inplace):
    _validate_dropout_probability(p)
    if inplace:
        return _nn_functional_dropout("feature_alpha_dropout", input, p, training, True)
    return _nn_functional_dropout("feature_alpha_dropout", input, p, training, False)


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


def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    return input.tanh()


def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    return input.sigmoid()


def _softsign_impl(input):
    if isinstance(input, Tensor) and torch.is_grad_enabled() and input.requires_grad:
        raise RuntimeError("softsign(): autograd recording is not supported")
    return input / (input.abs() + 1)


def softsign(input):
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    return _dispatch_unary_torch_function(
        softsign,
        _softsign_impl,
        input,
        {},
    )


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _nn_functional_linear(input, weight, bias)


linear.__doc__ = _LINEAR_DOC


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    return _nn_functional_mse_loss(
        input,
        target,
        size_average,
        reduce,
        reduction,
        weight,
    )


mse_loss.__doc__ = _MSE_LOSS_DOC


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


def dropout1d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 1D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout1d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    return _dispatch_unary_torch_function(
        dropout1d,
        _dropout1d_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )


def dropout2d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 2D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    return _dispatch_unary_torch_function(
        dropout2d,
        _dropout2d_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )


def dropout3d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    return _dispatch_unary_torch_function(
        dropout3d,
        _dropout3d_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )


def alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Apply alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
    return _dispatch_unary_torch_function(
        alpha_dropout,
        _alpha_dropout_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly masks out entire channels (a channel is a feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]` of the input tensor. Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function.

    Each element will be masked independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    See :class:`~torch.nn.FeatureAlphaDropout` for details.

    Args:
        p: dropout probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    return _dispatch_unary_torch_function(
        feature_alpha_dropout,
        _feature_alpha_dropout_impl,
        input,
        {"p": p, "training": training, "inplace": inplace},
    )
