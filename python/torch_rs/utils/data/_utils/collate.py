"""Conversion helpers used by :mod:`torch_rs.utils.data`."""

import collections
import copy

from torch_rs import Tensor


_NUMPY_ERROR = "default_convert does not support NumPy arrays or scalars"


def _is_numpy_array_or_scalar(data):
    return any(
        base.__module__ == "numpy" and base.__name__ in {"generic", "ndarray"}
        for base in type(data).__mro__
    )


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not a NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """
    if isinstance(data, Tensor):
        return data
    if _is_numpy_array_or_scalar(data):
        raise TypeError(_NUMPY_ERROR)

    elem_type = type(data)
    if isinstance(data, collections.abc.Mapping):
        if isinstance(data, collections.abc.MutableMapping):
            try:
                clone = copy.copy(data)
            except TypeError:
                return {key: default_convert(data[key]) for key in data}

            converted = {key: default_convert(data[key]) for key in data}
            try:
                clone.update(converted)
            except TypeError:
                return converted
            return clone

        converted = {key: default_convert(data[key]) for key in data}
        try:
            return elem_type(converted)
        except TypeError:
            return converted
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return elem_type(*(default_convert(value) for value in data))
    if isinstance(data, tuple):
        return [default_convert(value) for value in data]
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        if isinstance(data, collections.abc.MutableSequence):
            try:
                clone = copy.copy(data)
            except TypeError:
                return [default_convert(value) for value in data]

            converted = [default_convert(value) for value in data]
            try:
                for index, value in enumerate(converted):
                    clone[index] = value
            except TypeError:
                return converted
            return clone

        converted = [default_convert(value) for value in data]
        try:
            return elem_type(converted)
        except TypeError:
            return converted
    return data
