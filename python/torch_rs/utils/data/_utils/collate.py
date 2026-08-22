import collections.abc
import copy

from torch_rs import Tensor


def default_convert(data):
    r"""
    Recursively copy supported containers without converting their leaves.

    Tensors, Python scalars, strings, bytes, and other non-container values are
    returned unchanged. Mappings, named tuples, tuples, and non-string
    sequences are traversed using :func:`torch.utils.data.default_convert`'s
    container-copying rules. Named tuples retain their type, while ordinary
    tuples are converted to lists.

    NumPy arrays and scalars are deliberately unsupported and raise
    :class:`TypeError` instead of being converted to tensors.

    Args:
        data: a single data point to be converted

    Examples:
        >>> default_convert(0)
        0
        >>> from collections import namedtuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert((1, [2, 3]))
        [1, [2, 3]]
    """
    elem_type = type(data)
    if isinstance(data, Tensor):
        return data
    if elem_type.__module__ == "numpy":
        raise TypeError("default_convert(): NumPy arrays and scalars are not supported")
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
