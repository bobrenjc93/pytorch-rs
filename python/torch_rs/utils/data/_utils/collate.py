import collections.abc
import copy

from torch_rs import Tensor


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
    elem_type = type(data)
    if isinstance(data, Tensor):
        return data
    if elem_type.__module__ == "numpy":
        raise TypeError("default_convert(): NumPy arrays and scalars are not supported")
    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            return {key: default_convert(data[key]) for key in data}
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return elem_type(*(default_convert(value) for value in data))
    if isinstance(data, tuple):
        return [default_convert(value) for value in data]
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                clone = copy.copy(data)
                for index, value in enumerate(data):
                    clone[index] = default_convert(value)
                return clone
            return elem_type([default_convert(value) for value in data])
        except TypeError:
            return [default_convert(value) for value in data]
    return data
