"""Leaf-only conversion helpers for :mod:`torch_rs.utils.data`."""

import collections.abc

from torch_rs import Tensor


_NUMPY_ERROR = "default_convert(): NumPy arrays and scalars are not supported"
_CONTAINER_ERROR = (
    "default_convert(): recursive Mapping, sequence, and named-tuple inputs "
    "are not supported"
)


def _is_numpy_array_or_scalar(data):
    return any(
        base.__module__ == "numpy" and base.__name__ in {"generic", "ndarray"}
        for base in type(data).__mro__
    )


def default_convert(data):
    r"""
    Return a supported data leaf unchanged.

    Exact native :class:`torch.Tensor` objects and non-container values such as
    Python scalars, strings, bytes, and arbitrary object leaves preserve their
    identity. Tensor storage, view metadata, and autograd state are therefore
    unchanged.

    NumPy arrays and scalars are not supported. Mappings, named tuples, and
    non-string sequences are also not supported because recursive conversion
    is outside this leaf-only implementation. Unsupported inputs raise
    :class:`TypeError` without traversing their contents.

    Args:
        data: a single data point to be converted

    Returns:
        The original input object.

    Raises:
        TypeError: If ``data`` is a NumPy array or scalar, mapping, named tuple,
            or non-string sequence.

    Examples:
        >>> default_convert(0)
        0
        >>> value = object()
        >>> default_convert(value) is value
        True
    """
    if type(data) is Tensor:
        return data
    if _is_numpy_array_or_scalar(data):
        raise TypeError(_NUMPY_ERROR)
    if isinstance(data, collections.abc.Mapping):
        raise TypeError(_CONTAINER_ERROR)
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        raise TypeError(_CONTAINER_ERROR)
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        raise TypeError(_CONTAINER_ERROR)
    return data
