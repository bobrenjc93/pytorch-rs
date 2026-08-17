"""Functional interface."""

from collections.abc import Sequence

from .torch_rs import Size


__all__ = ["broadcast_shapes"]


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
