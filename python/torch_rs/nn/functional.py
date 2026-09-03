"""Functional interface."""

import math
import warnings

import torch_rs as torch
from torch_rs import Tensor
from torch_rs._diagnostics import _format_single_element_tensor
from torch_rs.overrides import _dispatch_unary_torch_function

from ..torch_rs import (
    _nn_functional_dropout,
    _nn_functional_l1_loss,
    _nn_functional_linear,
    _nn_functional_mse_loss,
)


_LINEAR_DOC = r"""
linear(input, weight, bias=None) -> Tensor

Applies the rank-1, rank-2, or rank-3 transformation
:math:`\mathrm{output} = \mathrm{input} \, \mathrm{weight}^{T}`, with an
optional rank-1 bias for rank-1 or rank-2 input.

The current native implementation requires exact ``torch_rs.Tensor`` operands
with CPU ``float32`` storage and shape ``(in_features,)``,
``(rows, in_features)``, or ``(batch, sequence, in_features)`` for ``input``
and ``(out_features, in_features)`` for ``weight``. For rank-1 or rank-2
input, ``bias`` may instead be an exact rank-1 tensor with shape
``(out_features,)`` or the PyTorch-compatible singleton shape ``(1,)``. For
rank-3 input, ``bias`` must be ``None``. The operation returns a fresh,
independent row-major tensor with the corresponding final dimension replaced by
``out_features``.

Tensor subclasses, active ``TorchFunctionMode`` contexts, and active autograd
recording are not supported. Gradient-requiring input, weight, or supported
bias operands may be used inside ``torch.no_grad()``.
"""


_L1_LOSS_DOC = r"""
l1_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

Measures the element-wise absolute error between ``input`` and ``target``.

The current native implementation requires exact ``torch_rs.Tensor`` operands
with CPU ``float32`` storage, broadcastable shapes, ``reduction='none'`` or
``reduction='sum'``, ``size_average=None``, ``reduce=None``, and
``weight=None``. It fuses same-shape row-major contiguous operands, same-shape
operands with identical strides and non-overlapping dense storage, non-empty
same-shape rank-4 channels-last-contiguous operands, and rank-0 scalar
broadcasts over row-major contiguous tensors into one native absolute-difference
pass, otherwise preserving the established subtraction and absolute-value
behavior. For ``reduction='sum'``, same-shape row-major contiguous operands use
a direct fused absolute-difference scalar reduction, as does a row-major rank-2
tensor paired with a row-major rank-1 tensor over its trailing dimension; other
supported layouts compose the absolute-difference result with the supported
full-tensor sum.
The operation returns a fresh, independent tensor with PyTorch-compatible
values, shape, strides, scalar metadata, and size-mismatch warning.

Unbroadcastable shapes, ``reduction='mean'``, legacy ``size_average``/``reduce``
behavior, weights, unsupported dtypes or devices, Tensor subclasses, active
``TorchFunctionMode`` contexts, active autograd recording, and module loss
wrappers are not supported. Gradient-requiring operands may be used inside
``torch.no_grad()``.
"""


_MSE_LOSS_DOC = r"""
mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

Measures the element-wise mean squared error between ``input`` and ``target``.

The current native implementation requires exact ``torch_rs.Tensor`` operands
with CPU ``float32`` storage, broadcastable shapes, ``reduction='none'`` or
``reduction='mean'`` or ``reduction='sum'``, ``size_average=None``,
``reduce=None``, and ``weight=None``. It fuses subtraction and square into one
native pass and directly reduces same-shape row-major contiguous no-grad
``reduction='mean'`` inputs to one scalar; other supported ``reduction='mean'``
and ``reduction='sum'`` inputs compose that result with the supported
full-tensor mean or sum.
The operation returns a fresh, independent tensor with PyTorch-compatible
values, shape, strides, scalar metadata, and size-mismatch warning.

Unbroadcastable shapes, legacy ``size_average``/``reduce`` behavior, weights,
unsupported dtypes or devices, Tensor subclasses, active ``TorchFunctionMode``
contexts, active autograd recording for ``reduction='none'``, and module loss
wrappers are not supported. Gradient-requiring operands may be used with
``reduction='mean'`` or ``reduction='sum'`` or inside ``torch.no_grad()``.
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


def _torch_functional_type_name(value):
    value_type = type(value)
    module = value_type.__module__
    name = value_type.__name__
    if module == "numpy":
        return f"numpy.{name}"
    if module in ("torch_rs", "torch_rs.torch_rs") and name in (
        "device",
        "dtype",
        "finfo",
        "layout",
        "memory_format",
        "Size",
    ):
        return f"torch.{name}"
    return name


def _silu_tensor_type_error(input, operation):
    type_name = _torch_functional_type_name(input)
    raise TypeError(
        f"{operation}(): argument 'input' (position 1) must be Tensor, not {type_name}"
    )


def _silu_impl(input, inplace):
    inplace_enabled = bool(inplace)
    operation = "silu_" if inplace_enabled else "silu"
    if type(input) is not Tensor:
        _silu_tensor_type_error(input, operation)
    if inplace_enabled:
        raise NotImplementedError(
            "torch_rs.nn.functional.silu does not support inplace=True"
        )
    return input * input.sigmoid()


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    return _dispatch_unary_torch_function(
        silu,
        _silu_impl,
        input,
        {"inplace": inplace},
    )


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


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    return _nn_functional_l1_loss(
        input,
        target,
        size_average,
        reduce,
        reduction,
        weight,
    )


l1_loss.__doc__ = _L1_LOSS_DOC


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
