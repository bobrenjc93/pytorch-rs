"""Automatic differentiation helpers."""

import operator as _operator
from collections.abc import Sequence as _Sequence
from contextvars import ContextVar as _ContextVar
from typing import Union as _Union

from .. import _C as _C
from ..overrides import _dispatch_unary_torch_function
from . import grad_mode as grad_mode
from .grad_mode import no_grad as no_grad


_Tensor = _C.Tensor
_TensorOrTensorsOrGradEdge = _Union[
    _Tensor,
    _Sequence[_Tensor],
    "GradientEdge",
    _Sequence["GradientEdge"],
]
_TensorOrOptionalTensors = _Tensor | _Sequence[_Tensor | None]
_native_backward = _C._autograd_backward
_mode_redispatch_active = _ContextVar(
    "torch_rs_autograd_backward_mode_redispatch_active",
    default=False,
)


def _require_default_graph_option(name, value, *, allow_none):
    if value is None and allow_none:
        return
    enabled = _operator.index(value)
    if enabled:
        raise NotImplementedError(
            f"torch_rs.autograd.backward does not support {name}=True"
        )


def _single_root_tensor(tensors):
    if type(tensors) is _Tensor:
        return tensors
    if (
        _mode_redispatch_active.get()
        and type(tensors) is tuple
        and len(tensors) == 1
        and type(tensors[0]) is _Tensor
    ):
        return tensors[0]
    raise TypeError(
        "torch_rs.autograd.backward only supports one exact native Tensor"
    )


def _backward_implementation(
    tensor,
    *,
    grad_tensors,
    retain_graph,
    create_graph,
    inputs,
):
    if grad_tensors is not None:
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support explicit gradients"
        )
    _require_default_graph_option("retain_graph", retain_graph, allow_none=True)
    _require_default_graph_option("create_graph", create_graph, allow_none=False)
    if inputs is not None:
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support inputs"
        )

    _native_backward(tensor)


def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrOptionalTensors | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    grad_variables: _TensorOrOptionalTensors | None = None,
    inputs: _TensorOrTensorsOrGradEdge | dict[str, _Tensor] | None = None,
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
    tensor = _single_root_tensor(tensors)
    if grad_variables is not None:
        raise NotImplementedError(
            "torch_rs.autograd.backward does not support grad_variables"
        )

    keyword_arguments = {
        "grad_tensors": grad_tensors,
        "retain_graph": retain_graph,
        "create_graph": create_graph,
        "inputs": inputs,
    }
    token = _mode_redispatch_active.set(True)
    try:
        return _dispatch_unary_torch_function(
            backward,
            _backward_implementation,
            tensor,
            keyword_arguments,
            dispatch_arguments=((tensor,),),
        )
    finally:
        _mode_redispatch_active.reset(token)


is_multithreading_enabled = _C._is_multithreading_enabled
is_view_replay_enabled = _C._is_view_replay_enabled

__all__ = ["backward", "grad_mode", "no_grad"]

del _C
