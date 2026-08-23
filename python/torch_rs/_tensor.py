"""Python-owned methods installed on :class:`torch_rs.Tensor`."""

import operator as _operator

from .overrides import _dispatch_unary_torch_function
from .torch_rs import Tensor as _NativeTensor


# Keep the native no-argument engine reachable after installing the
# Python-owned compatibility wrapper. Preserve it across module reloads too,
# when ``_NativeTensor.backward`` already refers to the wrapper below.
if "_native_backward" not in globals():
    _native_backward = _NativeTensor.backward


def _require_default_backward_graph_option(name, value, *, allow_none):
    if value is None and allow_none:
        return
    enabled = _operator.index(value)
    if enabled:
        raise NotImplementedError(
            f"torch_rs.Tensor.backward does not support {name}=True"
        )


def _is_shared_implementation(input):
    if type(input) is Tensor:
        # Native tensors use process-local owned or mutex-backed gradient
        # storage. Neither representation is operating-system shared memory.
        return False
    # Preserve PyTorch's unbound-call behavior without adding storage methods
    # to the native Tensor API.
    return input._typed_storage()._is_shared()


# Defining the method in a Python class gives it PyTorch's function metadata.
# The module binding is replaced with the native class immediately afterward.
class Tensor:
    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        r"""Computes the gradient of current tensor wrt graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying a ``gradient``.
        It should be a tensor of matching type and shape, that represents
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.
        See :ref:`Default gradient layouts<default-grad-layouts>`
        for details on the memory layout of accumulated gradients.

        .. note::

            If you run any forward ops, create ``gradient``, and/or call ``backward``
            in a user-specified CUDA stream context, see
            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

        .. note::

            When ``inputs`` are provided and a given input is not a leaf,
            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
            It is an implementation detail on which the user should not rely.
            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

        Args:
            gradient (Tensor, optional): The gradient of the function
                being differentiated w.r.t. ``self``.
                This argument can be omitted if ``self`` is a scalar. Defaults to ``None``.
            retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be freed;
                If ``True``, it will be retained. The default is ``None``, in which case the value is inferred from ``create_graph``
                (i.e., the graph is retained only when higher-order derivative tracking is requested). Note that in nearly all cases
                setting this option to True is not needed and often can be worked around in a much more efficient way.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
            inputs (Sequence[Tensor] or dict[str, Tensor], optional): Inputs w.r.t. which
                the gradient will be accumulated into ``.grad``. All other tensors will be
                ignored. If not provided, the gradient is accumulated into all the leaf
                Tensors that were used to compute the :attr:`tensors`. A dict of tensors
                (e.g. ``dict(model.named_parameters())``) is also accepted.
                Defaults to ``None``.
        """
        if gradient is not None:
            raise NotImplementedError(
                "torch_rs.Tensor.backward does not support explicit gradients"
            )
        _require_default_backward_graph_option(
            "retain_graph", retain_graph, allow_none=True
        )
        _require_default_backward_graph_option(
            "create_graph", create_graph, allow_none=False
        )
        if inputs is not None:
            raise NotImplementedError(
                "torch_rs.Tensor.backward does not support inputs"
            )

        return _native_backward(self)

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        return _dispatch_unary_torch_function(
            Tensor.is_shared,
            _is_shared_implementation,
            self,
            {},
        )


_backward = Tensor.backward
_is_shared = Tensor.is_shared
Tensor = _NativeTensor
Tensor.backward = _backward
Tensor.is_shared = _is_shared

del _backward, _is_shared, _NativeTensor
