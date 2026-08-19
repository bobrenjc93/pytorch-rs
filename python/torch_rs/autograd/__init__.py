"""Automatic differentiation helpers."""

import warnings as _warnings
from collections.abc import Sequence
from typing import Union

from ..overrides import (
    _dispatch_unary_torch_function,
    _is_in_torch_function_handler,
)
from ..torch_rs import Tensor
from . import grad_mode as grad_mode
from .grad_mode import no_grad as no_grad


__all__ = ["backward", "grad_mode", "no_grad"]


_TensorOrOptionalTensors = Tensor | Sequence[Tensor | None]
_TensorOrTensorsOrGradEdge = Union[
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]
_IntegerType = type(0)


def _index_as_int(value):
    index_method = getattr(type(value), "__index__", None)
    if index_method is None:
        raise TypeError
    result = index_method(value)
    if not isinstance(result, _IntegerType):
        raise TypeError
    return _IntegerType.__index__(result)


def _is_false_flag(value):
    try:
        return _index_as_int(value) == 0
    except TypeError:
        return False


def _backward_impl(
    tensors,
    grad_tensors,
    retain_graph,
    create_graph,
    inputs,
):
    if not isinstance(tensors, Tensor):
        raise NotImplementedError(
            "torch_rs.autograd.backward only supports a single Tensor"
        )
    if grad_tensors is not None:
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support explicit gradients"
        )
    if retain_graph is not None and not _is_false_flag(retain_graph):
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support retained graphs"
        )
    if not _is_false_flag(create_graph):
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support higher-order graphs"
        )
    if inputs is not None:
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support input filtering"
        )
    if tensors.requires_grad and tensors.numel() != 1:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
    tensors.backward()


def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrOptionalTensors | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    grad_variables: _TensorOrOptionalTensors | None = None,
    inputs: _TensorOrTensorsOrGradEdge | dict[str, Tensor] | None = None,
) -> None:
    r"""Compute the sum of gradients of given tensors with respect to graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    ``.grad`` attributes or set them to ``None`` before calling it.
    See :ref:`Default gradient layouts<default-grad-layouts>`
    for details on the memory layout of accumulated gradients.

    .. note::
        Using this method with ``create_graph=True`` will create a reference cycle
        between the parameter and its gradient which can cause a memory leak.
        We recommend using ``autograd.grad`` when creating the graph to avoid this.
        If you have to use this function, make sure to reset the ``.grad`` fields of your
        parameters to ``None`` after use to break the cycle and avoid the leak.

    .. note::

        If you run any forward ops, create ``grad_tensors``, and/or call ``backward``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        When ``inputs`` are provided and a given input is not a leaf,
        the current implementation will call its grad_fn (even though it is not strictly needed to get these gradients).
        It is an implementation detail on which the user should not rely.
        See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    Args:
        tensors (Sequence[Tensor] or Tensor or Sequence[GradientEdge] or GradientEdge): Tensors of which
            the derivative will be computed.
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): The "vector" in
            the Jacobian-vector product, usually gradients w.r.t. each element of
            corresponding tensors. None values can be specified for scalar Tensors or
            ones that don't require grad. If a None value would be acceptable for all
            grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge] or dict[str, Tensor], optional):
            Inputs w.r.t. which the gradient will be accumulated into ``.grad``.
            All other Tensors will be ignored. If not provided, the gradient is
            accumulated into all the leaf Tensors that were used to compute the
            :attr:`tensors`. A dict of tensors (e.g.
            ``dict(model.named_parameters())``) is also accepted, in which case
            the values are used as the input tensors.
    """
    if grad_variables is not None:
        _warnings.warn(
            "`grad_variables` is deprecated. Use `grad_tensors` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError(
                "`grad_tensors` and `grad_variables` (deprecated) "
                "arguments both passed to `backward()`. Please only "
                "use `grad_tensors`."
            )

    forwarded_sequence = (
        type(tensors) is tuple
        and len(tensors) == 1
        and isinstance(tensors[0], Tensor)
        and _is_in_torch_function_handler()
    )
    input_tensor = tensors[0] if forwarded_sequence else tensors
    mode_tensors = tensors if forwarded_sequence else (tensors,)
    return _dispatch_unary_torch_function(
        backward,
        _backward_impl,
        input_tensor,
        {
            "grad_tensors": grad_tensors,
            "retain_graph": retain_graph,
            "create_graph": create_graph,
            "inputs": inputs,
        },
        torch_function_args=(mode_tensors,),
        torch_function_kwargs={
            "grad_tensors": grad_tensors,
            "retain_graph": retain_graph,
            "create_graph": create_graph,
            "inputs": inputs,
        },
    )
