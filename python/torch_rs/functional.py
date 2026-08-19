"""Functional interface."""

from collections.abc import Sequence

from .overrides import _dispatch_unary_torch_function
from .torch_rs import (
    Size,
    Tensor,
    atleast_1d as _VF_atleast_1d,
    atleast_2d as _VF_atleast_2d,
    atleast_3d as _VF_atleast_3d,
)


__all__ = ["atleast_1d", "atleast_2d", "atleast_3d", "broadcast_shapes"]


_ATLEAST_1D_SEQUENCE_UNSUPPORTED = (
    "atleast_1d() sequence inputs only support an exact tuple or list of "
    "exact Tensors"
)


def _atleast_1d_impl(input):
    if type(input) in (tuple, list):
        if any(type(tensor) is not Tensor for tensor in input):
            raise TypeError(_ATLEAST_1D_SEQUENCE_UNSUPPORTED)
        return tuple(_VF_atleast_1d(tensor) for tensor in input)
    return _VF_atleast_1d(input)


def atleast_1d(*tensors):
    r"""
    Returns a 1-dimensional view of each input tensor with zero dimensions.
    Input tensors with one or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 1-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.arange(2)
        >>> x
        tensor([0, 1])
        >>> torch.atleast_1d(x)
        tensor([0, 1])
        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_1d(x)
        tensor([1.])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_1d((x, y))
        (tensor([0.5000]), tensor([1.]))
        >>> torch.atleast_1d()
        ()
    """
    if len(tensors) != 1:
        raise TypeError("atleast_1d() only supports a single Tensor input")
    return _dispatch_unary_torch_function(
        atleast_1d,
        _atleast_1d_impl,
        tensors[0],
        {},
    )


def _atleast_2d_impl(input):
    return _VF_atleast_2d(input)


def atleast_2d(*tensors):
    r"""
    Returns a 2-dimensional view of each input tensor with zero dimensions.
    Input tensors with two or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 2-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_2d(x)
        tensor([[1.]])
        >>> x = torch.arange(4).view(2, 2)
        >>> x
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_2d(x)
        tensor([[0, 1],
                [2, 3]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_2d((x, y))
        (tensor([[0.5000]]), tensor([[1.]]))
        >>> torch.atleast_2d()
        ()
    """
    if len(tensors) != 1:
        raise TypeError("atleast_2d() only supports a single Tensor input")
    return _dispatch_unary_torch_function(
        atleast_2d,
        _atleast_2d_impl,
        tensors[0],
        {},
    )


def _atleast_3d_impl(input):
    return _VF_atleast_3d(input)


def atleast_3d(*tensors):
    r"""
    Returns a 3-dimensional view of each input tensor with zero dimensions.
    Input tensors with three or more dimensions are returned as-is.

    Args:
        input (Tensor or sequence of Tensors): tensor(s) to be converted to at least 3-dimensional.

    Returns:
        output (Tensor or tuple of Tensors)

    Example:

        >>> x = torch.tensor(0.5)
        >>> x
        tensor(0.5000)
        >>> torch.atleast_3d(x)
        tensor([[[0.5000]]])
        >>> y = torch.arange(4).view(2, 2)
        >>> y
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_3d(y)
        tensor([[[0],
                 [1]],
                <BLANKLINE>
                [[2],
                 [3]]])
        >>> x = torch.tensor(1).view(1, 1, 1)
        >>> x
        tensor([[[1]]])
        >>> torch.atleast_3d(x)
        tensor([[[1]]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.0)
        >>> torch.atleast_3d((x, y))
        (tensor([[[0.5000]]]), tensor([[[1.]]]))
        >>> torch.atleast_3d()
        ()
    """
    if len(tensors) != 1:
        raise TypeError("atleast_3d() only supports a single Tensor input")
    return _dispatch_unary_torch_function(
        atleast_3d,
        _atleast_3d_impl,
        tensors[0],
        {},
    )


def _guard_or_false(condition):
    if not isinstance(condition, bool):
        raise AssertionError(f"Expected bool, got {type(condition)}")
    return condition


def _check(condition, message):
    if not isinstance(condition, bool):
        raise TypeError(f"cond must be a bool, but got {type(condition)}")
    if not condition:
        raise RuntimeError(message())


def broadcast_shapes(*shapes):
    r"""broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need to create intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
    normalized_shapes = tuple(
        (shape,) if isinstance(shape, int) else shape
        for shape in shapes
        if shape is not None
    )
    if not normalized_shapes:
        return Size([])

    for shape in normalized_shapes:
        if not isinstance(shape, Sequence):
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, or a "
                f"list of ints, got {shape}"
            )

    common_shape = [1] * max(len(shape) for shape in normalized_shapes)
    for argument_index, shape in enumerate(normalized_shapes):
        for index in range(-1, -1 - len(shape), -1):
            dimension = shape[index]
            if _guard_or_false(dimension == common_shape[index]):
                continue

            if _guard_or_false(common_shape[index] == 1):
                if dimension < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[index] = dimension

            if _guard_or_false(dimension == 1):
                continue

            _check(
                common_shape[index] == dimension,
                lambda: (
                    "Attempting to broadcast a dimension of length "
                    f"{dimension} at {index}! Mismatching argument at index "
                    f"{argument_index} had {shape}; but expected shape should "
                    f"be broadcastable to {common_shape}"
                ),
            )

    return Size(common_shape)
